"""Train the pure inverse-dynamics model on one held-out task.

This mirrors the task-filtered split style from ``train_dp.py`` and reuses the
episode-level split builder from ``train_stage2.py``:

* train on exactly one selected held-out task
* split that task's episodes into train/val at the episode level
* predict intermediate actions from ``(context_frame, gt_next_frame)``
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from pathlib import Path

import hydra
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf

from pure_idm import PureInverseDynamicsModel, PureInverseDynamicsModelDP
from train_stage1 import Trainer, unwrap_model
from train_stage2 import build_heldout_task_datasets

log = logging.getLogger(__name__)


def _build_wandb_safe_task_tag(task_name: str, *, max_len: int = 64) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "_", task_name).strip("._-")
    if not normalized:
        normalized = "task"
    if len(normalized) <= max_len:
        return normalized

    digest = hashlib.sha1(task_name.encode("utf-8")).hexdigest()[:8]
    keep = max_len - len(digest) - 1
    keep = max(keep, 8)
    shortened = normalized[:keep].rstrip("._-")
    if not shortened:
        shortened = "task"
    return f"{shortened}_{digest}"


def _resolve_dataset_root(cfg: DictConfig) -> Path:
    dataset_root = OmegaConf.select(cfg, "data.root", default=None)
    if dataset_root in (None, "", "None"):
        dataset_root = OmegaConf.select(cfg, "dataset_root", default=None)
    if dataset_root in (None, "", "None"):
        raise ValueError(
            "Dataset root is missing. Set `data.root` or `dataset_root`."
        )
    return Path(str(dataset_root)).expanduser()


def _load_heldout_tasks(dataset_root: Path) -> tuple[Path, list[str]]:
    meta_path = dataset_root / "meta" / "test_tasks.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"Held-out task metadata not found: {meta_path}")

    with meta_path.open() as f:
        payload = json.load(f)

    heldout_tasks: list[str] = []
    if isinstance(payload, dict):
        if isinstance(payload.get("tasks"), list):
            heldout_tasks.extend(
                str(task) for task in payload["tasks"] if task not in (None, "")
            )
        if isinstance(payload.get("records"), list):
            heldout_tasks.extend(
                str(record["task"])
                for record in payload["records"]
                if isinstance(record, dict) and record.get("task") not in (None, "")
            )
    elif isinstance(payload, list):
        heldout_tasks.extend(str(task) for task in payload if task not in (None, ""))

    seen = set()
    heldout_tasks = [
        task for task in heldout_tasks if not (task in seen or seen.add(task))
    ]
    if not heldout_tasks:
        raise ValueError(f"No held-out tasks were found in {meta_path}.")

    return meta_path, heldout_tasks


def _resolve_selected_heldout_task(
    selected_task: object,
    heldout_tasks: list[str],
) -> str:
    selector = str(selected_task).strip()
    if selector in ("", "None"):
        raise ValueError("`data.selected_task` must not be empty.")
    if selector in heldout_tasks:
        return selector
    raise ValueError(
        "Configured `data.selected_task` does not match any held-out task in "
        f"meta/test_tasks.json: {selector!r}. Available held-out tasks: {heldout_tasks}"
    )


def _apply_selected_task_overrides(cfg: DictConfig) -> None:
    selected_task = OmegaConf.select(cfg, "data.selected_task", default=None)
    if selected_task in (None, "", "None"):
        raise ValueError(
            "Pure IDM training requires `data.selected_task` so each run trains one held-out task."
        )

    dataset_root = _resolve_dataset_root(cfg)
    meta_path, heldout_tasks = _load_heldout_tasks(dataset_root)
    resolved_task = _resolve_selected_heldout_task(selected_task, heldout_tasks)

    cfg.data.selected_task = resolved_task
    cfg.data.task_tag = _build_wandb_safe_task_tag(resolved_task)
    cfg.data.test_tasks = [resolved_task]
    if int(OmegaConf.select(cfg, "data.test_task_count", default=1)) != 1:
        cfg.data.test_task_count = 1

    rank = int(os.environ.get("RANK", "0"))
    if rank == 0:
        log.info(
            "Using held-out task-filtered split for pure IDM training: "
            f"meta={meta_path}, data.selected_task={resolved_task!r}, "
            f"data.task_tag={cfg.data.task_tag!r}, "
            f"data.stage2_val_fraction={float(cfg.data.stage2_val_fraction):.3f}"
        )


def _apply_image_resolution_overrides(cfg: DictConfig) -> None:
    image_resolution = OmegaConf.select(cfg, "data.image_resolution", default=None)
    if image_resolution not in (None, "", "None"):
        resolution = int(image_resolution)
        if resolution <= 0:
            raise ValueError(
                f"`data.image_resolution` must be positive, got {resolution}."
            )
        cfg.model.resolution = resolution

    resolution = int(cfg.model.resolution)
    patch_size = int(cfg.model.patch_size)
    if resolution % patch_size != 0:
        raise ValueError(
            f"`model.resolution` ({resolution}) must be divisible by "
            f"`model.patch_size` ({patch_size})."
        )

    crop_shape = OmegaConf.select(cfg, "train.idm_dp.crop_shape", default=None)
    if crop_shape not in (None, "", "None"):
        crop_h, crop_w = (int(crop_shape[0]), int(crop_shape[1]))
        if crop_h > resolution or crop_w > resolution:
            raise ValueError(
                "Diffusion-policy crop_shape cannot exceed model.resolution: "
                f"crop_shape={[crop_h, crop_w]}, resolution={resolution}."
            )

    rank = int(os.environ.get("RANK", "0"))
    if rank == 0:
        log.info(
            f"Pure IDM image resolution={resolution}, patch_size={patch_size}"
        )


class PureIDMTrainer(Trainer):
    def __init__(self, cfg: DictConfig):
        self.idm_type = str(cfg.train.get("idm_type", "mlp")).lower()
        super().__init__(cfg)

    def _stage_tag(self) -> str:
        return "pure_idm"

    def _build_datasets(self):
        configured_tasks = list(self.cfg.data.get("test_tasks", []))
        if len(configured_tasks) != 1:
            raise ValueError(
                "train_pure.py expects exactly one task in data.test_tasks after "
                "resolving data.selected_task."
            )

        val_fraction = float(
            OmegaConf.select(self.cfg, "data.stage2_val_fraction", default=0.01)
        )
        seed = int(OmegaConf.select(self.cfg, "seed", default=42))
        return build_heldout_task_datasets(
            self.cfg.data,
            self.cfg.model,
            self.is_main,
            val_fraction=val_fraction,
            seed=seed,
        )

    def _build_raw_model(self) -> nn.Module:
        n_actions = int(self.cfg.data.get("frame_gap", 0))
        if n_actions <= 0:
            raise ValueError(
                f"data.frame_gap must be positive for PureInverseDynamicsModel, got {n_actions}."
            )

        if self.is_main and bool(self.cfg.data.get("use_goal", False)):
            log.info(
                "Pure IDM ignores goal conditioning; set data.use_goal=false to avoid "
                "loading unused goal frames."
            )

        if self.idm_type in {"dp", "diffusion", "diffusion_policy"}:
            return PureInverseDynamicsModelDP(
                self.model_cfg,
                n_actions=n_actions,
                **OmegaConf.to_container(
                    self.cfg.train.get("idm_dp", {}),
                    resolve=True,
                ),
            ).to(self.device)

        return PureInverseDynamicsModel(self.model_cfg, n_actions=n_actions).to(
            self.device
        )

    def _should_skip_compile(self) -> bool:
        if self.idm_type in {"dp", "diffusion", "diffusion_policy"}:
            if self.is_main:
                log.info(
                    "Skipping torch.compile for pure diffusion-policy IDM "
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
                    "Pure DP IDM: disabling DDP buffer broadcasts and TorchDynamo "
                    "DDP optimizer to avoid collective/compile mismatches."
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
                log.info("Loaded action stats into pure diffusion-policy IDM.")

    def _forward(self, model, context, target, actions, goal):
        pred, _, loss = model(context, target, actions, goal=goal)
        return pred, loss

    @torch.no_grad()
    def _log_val_videos(self, n_samples=3, val_loss=None):
        del n_samples, val_loss

    @torch.no_grad()
    def _log_train_samples(self, n_samples=3):
        del n_samples


@hydra.main(config_path="configs", config_name="pure_idm", version_base="1.3")
def main(cfg: DictConfig) -> None:
    _apply_selected_task_overrides(cfg)
    _apply_image_resolution_overrides(cfg)
    OmegaConf.resolve(cfg)
    PureIDMTrainer(cfg).train()


if __name__ == "__main__":
    main()
