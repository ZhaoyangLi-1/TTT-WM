"""Stage 2 training for TTT-WM.

Two sub-steps, both driven by Hydra config ``configs/stage2_idm_config.yaml``:

* **2.1** — fine-tune the Stage-1 ``ARVideoPatchTransformer`` on the three
  held-out test tasks (``/scr2/zhaoyang/libero_wm/meta/test_tasks.json``).
  Episodes from those tasks are split into train/val at the episode level
  with ``val_fraction=0.01`` (configurable via ``data.stage2_val_fraction``).
  Training objective / inputs / outputs / hyper-parameters are identical to
  Stage 1 training.

* **2.2** — freeze the Stage-2.1 backbone (``ARVideoPatchTransformer``) and
  train **only** the IDM action head (``InverseDynamicsModel`` or
  ``InverseDynamicsModelDP`` from ``idm_model.py``, with ``freeze_stage1=True``).

Select the sub-step with ``train.substep=2.1`` (default) or ``train.substep=2.2``.

Additional config options:

* ``train.stage1_ckpt`` — optional warm start for 2.1 (path to a Stage-1 ckpt).
* ``train.stage2_1_ckpt`` — required for 2.2 (path to a Stage-2.1 ckpt).
* ``data.stage2_val_fraction`` — val split ratio for the 3 test tasks (default 0.01).
"""

from __future__ import annotations

import logging

import hydra
import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf

from idm_model import (
    InverseDynamicsModel as _IDMModel,
    InverseDynamicsModelDP as _IDMModelDP,
)
from train_stage1 import (
    Trainer,
    VideoFrameDataset,
    _clean_state_dict,
    unwrap_model,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset: only the 3 held-out test tasks, split 99/1 at the episode level.
# ---------------------------------------------------------------------------


class HeldoutTaskSplitDataset(VideoFrameDataset):
    """VideoFrameDataset restricted to the held-out test tasks.

    The parent class already loads all episodes whose task is in
    ``meta/test_tasks.json`` when ``split != "train"``. We then reshuffle
    those episodes with a fixed seed and carve off ``val_fraction`` of them
    for validation; the remainder becomes the train split.
    """

    def __init__(
        self,
        data_cfg: DictConfig,
        model_cfg: DictConfig,
        split: str,
        *,
        val_fraction: float = 0.01,
        seed: int = 42,
        is_main: bool = True,
    ):
        if split not in {"train", "val"}:
            raise ValueError(
                f"HeldoutTaskSplitDataset split must be 'train' or 'val', got {split!r}"
            )

        # Ask the parent for the "val" (held-out-task) episode pool.
        super().__init__(data_cfg, model_cfg, "val", is_main=is_main)

        all_eps = sorted(self.episode_indices)
        if not all_eps:
            if is_main:
                log.warning("HeldoutTaskSplitDataset: no episodes found for the configured test tasks.")
            return

        rng = np.random.default_rng(seed)
        shuffled = list(all_eps)
        rng.shuffle(shuffled)

        if len(shuffled) > 1:
            n_val = max(1, int(round(len(shuffled) * val_fraction)))
            n_val = min(n_val, len(shuffled) - 1)
        else:
            n_val = 0

        val_eps = set(shuffled[:n_val])
        train_eps = set(shuffled[n_val:])
        selected = val_eps if split == "val" else train_eps

        allowed_paths = {self.episode_files[ep] for ep in selected}
        self.episode_indices = sorted(selected)
        self.samples = [s for s in self.samples if s[0] in allowed_paths]

        if is_main:
            log.info(
                f"[stage2.{split}] {len(self.samples)} windows / "
                f"{len(self.episode_indices)} episodes "
                f"(val_fraction={val_fraction}, seed={seed})"
            )


def build_heldout_task_datasets(
    data_cfg: DictConfig,
    model_cfg: DictConfig,
    is_main: bool,
    *,
    val_fraction: float,
    seed: int,
):
    train_ds = HeldoutTaskSplitDataset(
        data_cfg, model_cfg, "train",
        val_fraction=val_fraction, seed=seed, is_main=is_main,
    )
    val_ds = HeldoutTaskSplitDataset(
        data_cfg, model_cfg, "val",
        val_fraction=val_fraction, seed=seed, is_main=is_main,
    )
    return train_ds, val_ds, None


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------


def _extract_backbone_state_dict(ckpt: dict) -> tuple[dict, str]:
    """Return (state_dict, source_label). Prefers EMA shadow weights."""
    if "ema" in ckpt:
        ema_state = ckpt["ema"]
        sd = ema_state["shadow"] if "shadow" in ema_state else ema_state
        return _clean_state_dict(sd), "EMA"
    return _clean_state_dict(ckpt["model"]), "live"


# ---------------------------------------------------------------------------
# Stage 2.1 — Fine-tune ARVideoPatchTransformer on 3 test tasks.
# ---------------------------------------------------------------------------


class Stage2Part1Trainer(Trainer):
    """Stage 2.1: same objective/inputs/outputs as Stage 1, restricted to the
    three held-out test tasks with a 99/1 episode-level split."""

    def _stage_tag(self) -> str:
        return "2.1"

    def _build_datasets(self):
        cfg = self.cfg
        val_fraction = float(
            OmegaConf.select(cfg, "data.stage2_val_fraction", default=0.01)
        )
        seed = int(OmegaConf.select(cfg, "seed", default=42))
        return build_heldout_task_datasets(
            cfg.data, cfg.model, self.is_main,
            val_fraction=val_fraction, seed=seed,
        )

    def _build_trainable_model(self, raw_wm: nn.Module, has_goal: bool) -> nn.Module:
        stage1_ckpt = str(self.cfg.train.get("stage1_ckpt", "") or "")
        if stage1_ckpt:
            ckpt = torch.load(stage1_ckpt, map_location=self.device, weights_only=False)
            sd, src = _extract_backbone_state_dict(ckpt)
            raw_wm.load_state_dict(sd)
            if self.is_main:
                log.info(f"Stage 2.1 warm-start from {stage1_ckpt} ({src} weights)")
        raw_wm.prebuild_mask(device=self.device, has_goal=has_goal)
        return raw_wm


# ---------------------------------------------------------------------------
# Stage 2.2 — Freeze backbone, train only the action head via IDM.
# ---------------------------------------------------------------------------


class Stage2Part2Trainer(Trainer):
    """Stage 2.2: freeze the (Stage 2.1) ARVideoPatchTransformer and only
    train the IDM action head."""

    def __init__(self, cfg: DictConfig):
        # Read idm_type up front because the hooks below depend on it.
        self.idm_type = str(cfg.train.get("idm_type", "mlp")).lower()
        super().__init__(cfg)

    def _stage_tag(self) -> str:
        return "2.2"

    def _build_datasets(self):
        cfg = self.cfg
        val_fraction = float(
            OmegaConf.select(cfg, "data.stage2_val_fraction", default=0.01)
        )
        seed = int(OmegaConf.select(cfg, "seed", default=42))
        return build_heldout_task_datasets(
            cfg.data, cfg.model, self.is_main,
            val_fraction=val_fraction, seed=seed,
        )

    def _build_trainable_model(self, raw_wm: nn.Module, has_goal: bool) -> nn.Module:
        cfg = self.cfg
        stage2_1_ckpt = str(cfg.train.get("stage2_1_ckpt", "") or "")
        if not stage2_1_ckpt:
            raise ValueError(
                "Stage 2.2 requires train.stage2_1_ckpt pointing to the Stage 2.1 checkpoint."
            )
        ckpt = torch.load(stage2_1_ckpt, map_location=self.device, weights_only=False)
        sd, src = _extract_backbone_state_dict(ckpt)
        raw_wm.load_state_dict(sd)
        if self.is_main:
            log.info(f"Stage 2.2 loaded backbone from {stage2_1_ckpt} ({src} weights)")

        if self.idm_type in {"dp", "diffusion", "diffusion_policy"}:
            wrapped = _IDMModelDP(
                raw_wm,
                n_actions=int(cfg.data.get("frame_gap", 0)),
                freeze_stage1=True,
                **OmegaConf.to_container(cfg.train.get("idm_dp", {}), resolve=True),
            ).to(self.device)
        else:
            wrapped = _IDMModel(
                raw_wm,
                n_actions=int(cfg.data.get("frame_gap", 0)),
                freeze_stage1=True,
            ).to(self.device)
        wrapped.prebuild_mask(device=self.device, has_goal=has_goal)

        if self.is_main:
            n_trainable = sum(p.numel() for p in wrapped.parameters() if p.requires_grad) / 1e6
            n_total = sum(p.numel() for p in wrapped.parameters()) / 1e6
            log.info(
                f"Stage 2.2 backbone frozen — trainable {n_trainable:.2f}M / total {n_total:.2f}M"
            )
        return wrapped

    def _should_skip_compile(self) -> bool:
        if self.idm_type in {"dp", "diffusion", "diffusion_policy"}:
            if self.is_main:
                log.info(
                    "Skipping torch.compile for stage 2.2 diffusion_policy IDM "
                    "(robomimic/diffusion_policy vision stack is not compile-safe)."
                )
            return True
        return False

    def _ddp_broadcast_buffers(self) -> bool:
        if (
            self.world_size > 1
            and self.idm_type in {"dp", "diffusion", "diffusion_policy"}
        ):
            if hasattr(torch, "_dynamo"):
                torch._dynamo.config.optimize_ddp = False
            if self.is_main:
                log.info(
                    "Stage 2.2 DP IDM: disabling DDP buffer broadcasts and "
                    "TorchDynamo DDP optimizer to avoid collective/compile mismatches."
                )
            return False
        return True

    def _post_data_setup(self, train_ds) -> None:
        raw = unwrap_model(self.model)
        if hasattr(raw, "set_action_stats") and hasattr(train_ds, "get_action_stats"):
            stats = train_ds.get_action_stats()
            raw.set_action_stats(stats)
            if self.ema is not None and hasattr(self.ema.shadow, "set_action_stats"):
                self.ema.shadow.set_action_stats(stats)
            if self.is_main:
                log.info("Loaded action stats into Stage 2.2 diffusion-policy IDM.")

    def _forward(self, model, context, target, actions, goal):
        pred, _, loss = model(context, target, actions, goal=goal)
        return pred, loss


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


@hydra.main(config_path="configs", config_name="stage2_idm_config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    substep = str(cfg.train.get("substep", "2.1")).strip()
    if substep in {"2.1", "1", "part1"}:
        Stage2Part1Trainer(cfg).train()
    elif substep in {"2.2", "2", "part2"}:
        Stage2Part2Trainer(cfg).train()
    else:
        raise ValueError(
            f"Unknown train.substep={substep!r}. Expected '2.1' or '2.2'."
        )


if __name__ == "__main__":
    main()
