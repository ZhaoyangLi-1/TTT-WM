"""Stage 2 training for TTT-WM.

Two sub-steps, both driven by Hydra config ``configs/stage2_idm_config.yaml``:

* **2.1** — fine-tune the Stage-1 ``ARVideoPatchTransformer`` on the three
  held-out test tasks (``/scr2/zhaoyang/libero_wm/meta/test_tasks.json``).
  Episodes from those tasks are split into train/val at the episode level
  with ``val_fraction=0.01`` (configurable via ``data.stage2_val_fraction``).
  Training objective / inputs / outputs / hyper-parameters are identical to
  Stage 1 training.

* **2.2** — freeze the Stage-2.1 backbone (``ARVideoPatchTransformer``) and
  train **only** the IDM action head (``InverseDynamicsModelDP`` from
  ``idm_model.py``, with ``freeze_stage1=True``).

Select the sub-step with ``train.substep=2.1`` (default) or ``train.substep=2.2``.

Additional config options:

* ``train.stage1_ckpt``    — optional warm start for 2.1 (path to a Stage-1 ckpt).
* ``train.stage2_1_ckpt``  — required for 2.2 (path to a Stage-2.1 ckpt).
* ``data.stage2_val_fraction`` — val split ratio for the 3 test tasks (default 0.01).

Changes vs original
--------------------
Bug fixes inherited from train_stage1.py fixes:

  1. _eval_loader_loss: ddp_barrier() added before entering val loop.
     Stage 2 datasets are tiny (3 tasks, 99/1 split), so the val set may be
     as small as 1-2 episodes. Without the barrier, fast ranks enter the val
     DataLoader while slow ranks are still finishing the last train micro-batch,
     causing the ddp_all_reduce_scalar at the end of _run_epoch to stall.
     Stage 2 is MORE vulnerable to this than Stage 1 because the val set is
     orders of magnitude smaller and rank skew is proportionally worse.

  2. val DataLoader persistent_workers=False: same worker-state-pollution
     risk as Stage 1, amplified here because the val DataLoader has so few
     batches that a single stalled worker blocks the entire val pass.

  3. _post_data_setup called before model is fully wrapped (DDP / compile):
     In Stage2Part2Trainer, _post_data_setup calls unwrap_model(self.model)
     to set action stats. But _post_data_setup is called from __init__ of
     the base Trainer BEFORE DDP wrapping, so unwrap_model is a no-op and
     the set_action_stats call lands on the correct raw model. No bug here,
     but added a comment to make the ordering explicit.

  4. clip_grad_norm_ foreach=True: small speedup for multi-param-group
     optimizers, consistent with Stage 1 alignment.

  5. Stage2Part2Trainer._forward signature: original returns (pred, _, loss)
     from model(context, target, actions, goal=goal). The IDM model's
     __call__ returns a 3-tuple. _forward in the base class expects a 2-tuple
     (pred, loss). The override correctly unpacks this — no bug, but added
     an assertion comment for clarity.

Acceleration alignment with Stage 1 (without changing Stage 2 logic):

  - clip_grad_norm_ foreach=True (same as Stage 1 recommendation)
  - No other Stage 1 speed changes apply: Stage 2.1 uses the same training
    loop as Stage 1 (inherits Trainer unchanged). Stage 2.2 freezes the
    backbone so compile is intentionally skipped (the diffusion_policy
    vision stack is not torch.compile-safe).
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import re
from pathlib import Path

import hydra
import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf, open_dict
from torch.utils.data import Dataset

from idm_model import InverseDynamicsModelDP as _IDMModelDP
from train_stage1 import (
    Trainer,
    VideoFrameDataset,
    _clean_state_dict,
    ddp_barrier,
    unwrap_model,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Single-task selection helpers (mirrors train_pure_idm.py)
# ---------------------------------------------------------------------------


def _build_wandb_safe_task_tag(task_name: str, *, max_len: int = 64) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "_", task_name).strip("._-")
    if not normalized:
        normalized = "task"
    if len(normalized) <= max_len:
        return normalized
    digest = hashlib.sha1(task_name.encode("utf-8")).hexdigest()[:8]
    keep = max(max_len - len(digest) - 1, 8)
    shortened = normalized[:keep].rstrip("._-") or "task"
    return f"{shortened}_{digest}"


def _build_task_dirname(task_name: str) -> str:
    normalized = task_name.replace(":", "").replace(" ", "_").strip("_")
    return normalized or "task"


OmegaConf.register_new_resolver("task_slug", _build_task_dirname, replace=True)


def _resolve_dataset_root(cfg: DictConfig) -> Path:
    dataset_root = OmegaConf.select(cfg, "data.root", default=None)
    if dataset_root in (None, "", "None"):
        raise ValueError("Dataset root is missing. Set `data.root`.")
    return Path(str(dataset_root)).expanduser()


def _load_heldout_tasks(dataset_root: Path) -> tuple[Path, list[str]]:
    meta_path = dataset_root / "meta" / "test_tasks.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"Held-out task metadata not found: {meta_path}")
    with meta_path.open() as f:
        payload = json.load(f)

    heldout: list[str] = []
    if isinstance(payload, dict):
        if isinstance(payload.get("tasks"), list):
            heldout.extend(str(t) for t in payload["tasks"] if t not in (None, ""))
        if isinstance(payload.get("records"), list):
            heldout.extend(
                str(r["task"]) for r in payload["records"]
                if isinstance(r, dict) and r.get("task") not in (None, "")
            )
    elif isinstance(payload, list):
        heldout.extend(str(t) for t in payload if t not in (None, ""))

    seen: set[str] = set()
    heldout = [t for t in heldout if not (t in seen or seen.add(t))]
    if not heldout:
        raise ValueError(f"No held-out tasks were found in {meta_path}.")
    return meta_path, heldout


def _resolve_selected_heldout_task(selected_task: object, heldout_tasks: list[str]) -> str:
    selector = str(selected_task).strip()
    if selector in ("", "None"):
        raise ValueError("`data.selected_task` must not be empty.")
    if selector in heldout_tasks:
        return selector
    raise ValueError(
        "Configured `data.selected_task` does not match any held-out task in "
        f"meta/test_tasks.json: {selector!r}. Available: {heldout_tasks}"
    )


def _apply_selected_task_overrides(cfg: DictConfig) -> None:
    """Enforce that Stage 2 trains on exactly one held-out task.

    Populates ``data.test_tasks``/``task_tag``/``task_dirname`` from
    ``data.selected_task`` so the downstream dataset filter (which keys off
    ``data.test_tasks``) restricts training to that single task, and so wandb
    / hydra output paths carry the task name.
    """
    selected_task = OmegaConf.select(cfg, "data.selected_task", default=None)
    if selected_task in (None, "", "None"):
        raise ValueError(
            "Stage 2 training requires `data.selected_task` (one of the 3 "
            "held-out tasks from meta/test_tasks.json)."
        )

    dataset_root = _resolve_dataset_root(cfg)
    meta_path, heldout_tasks = _load_heldout_tasks(dataset_root)
    resolved_task = _resolve_selected_heldout_task(selected_task, heldout_tasks)

    with open_dict(cfg.data):
        cfg.data.selected_task = resolved_task
        cfg.data.task_tag = _build_wandb_safe_task_tag(resolved_task)
        cfg.data.task_dirname = _build_task_dirname(resolved_task)
        cfg.data.test_tasks = [resolved_task]
        cfg.data.test_task_count = 1

    rank = int(os.environ.get("RANK", "0"))
    if rank == 0:
        log.info(
            "Stage 2 single-task filter: "
            f"meta={meta_path}, data.selected_task={resolved_task!r}, "
            f"data.task_tag={cfg.data.task_tag!r}, "
            f"data.task_dirname={cfg.data.task_dirname!r}, "
            f"data.stage2_val_fraction={float(OmegaConf.select(cfg, 'data.stage2_val_fraction', default=0.01)):.3f}"
        )


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
                log.warning(
                    "HeldoutTaskSplitDataset: no episodes found for the configured test tasks."
                )
            return

        rng = np.random.default_rng(seed)
        shuffled = list(all_eps)
        rng.shuffle(shuffled)

        if len(shuffled) > 1:
            n_val = max(1, int(round(len(shuffled) * val_fraction)))
            n_val = min(n_val, len(shuffled) - 1)
        else:
            n_val = 0

        val_eps   = set(shuffled[:n_val])
        train_eps = set(shuffled[n_val:])
        selected  = val_eps if split == "val" else train_eps

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


def compute_backbone_sha1(state_dict: dict) -> str:
    """Content hash of a backbone state_dict. Used to bind a prebaked cache
    (scripts/prebake_stage1_pred.py) to the exact backbone weights that
    produced it — mismatched hashes at Stage 2.2 startup = raise or warn.
    """
    buf = io.BytesIO()
    # Sort keys for stability across Python dict-insert orders.
    ordered = {k: state_dict[k] for k in sorted(state_dict.keys())}
    torch.save(ordered, buf)
    return hashlib.sha1(buf.getvalue()).hexdigest()


# ---------------------------------------------------------------------------
# Cached Stage-1 prediction dataset (A1): wraps a HeldoutTaskSplitDataset so
# each __getitem__ also returns the prebaked pred_frames for that window,
# letting Stage 2.2 skip the frozen backbone's AR decode in the hot path.
# ---------------------------------------------------------------------------


class CachedStage1PredDataset(Dataset):
    """Read-only wrapper over a HeldoutTaskSplitDataset that attaches, as the
    5th item of each sample, the Stage-1 predicted frames that the frozen
    backbone would produce for ``(ctx, goal)`` of that window.

    Cache layout (produced by scripts/prebake_stage1_pred.py):

        <cache_dir>/cache_meta.json
            { "backbone_sha1": "...", "fin": int, "fout": int,
              "resolution": int, "num_channels": int, "use_goal": bool,
              "frame_gap": int, "task": "..." }

        <cache_dir>/episode_{ep_idx:06d}.stage1pred.pt
            {
              "pred_frames_u8":  uint8 Tensor (N, fout, H, W, C),
              "window_starts":   int64 Tensor (N,),
            }

    uint8 quantization is `((pred + 1) * 127.5).clamp(0, 255).byte()`, inverted
    at read time to float32 in [-1, 1] — the same range the IDM expects.
    """

    _CACHE_MAX = 8192  # matches VideoFrameDataset._CACHE_MAX

    def __init__(
        self,
        base: VideoFrameDataset,
        cache_dir: Path,
        expected_sha1: str,
        *,
        require_sha1_match: bool = True,
        is_main: bool = True,
    ):
        self.base = base
        self.cache_dir = Path(cache_dir)
        self._is_main = bool(is_main)
        self._require_sha1 = bool(require_sha1_match)

        meta_path = self.cache_dir / "cache_meta.json"
        if not meta_path.is_file():
            raise FileNotFoundError(
                f"Stage-1 pred cache not found at {meta_path}. "
                f"Run scripts/prebake_stage1_pred.py to populate it."
            )
        with meta_path.open() as f:
            self.meta = json.load(f)

        self._check_compat(expected_sha1)

        self._pt_cache: dict[str, dict] = {}
        # Map each base sample's (ep_path, start) to (cache_pt_path, row_idx)
        # by reading the window_starts index of each episode's pt once.
        self._sample_to_row = self._build_index()

    # ── compatibility & index build ──────────────────────────────────────

    def _check_compat(self, expected_sha1: str) -> None:
        got = str(self.meta.get("backbone_sha1", ""))
        mismatch = got and expected_sha1 and got != expected_sha1
        if mismatch:
            msg = (
                f"Stage-1 pred cache backbone_sha1 mismatch:\n"
                f"  cache: {got}\n"
                f"  trainer: {expected_sha1}\n"
                f"  cache_dir: {self.cache_dir}\n"
                f"Re-run scripts/prebake_stage1_pred.py with the new backbone."
            )
            if self._require_sha1:
                raise RuntimeError(msg)
            if self._is_main:
                log.warning(msg)

        base_cfg_ok = (
            int(self.meta.get("fin", -1)) == int(self.base.fin)
            and int(self.meta.get("fout", -1)) == int(self.base.fout)
            and int(self.meta.get("frame_gap", -1)) == int(self.base.gap)
        )
        if not base_cfg_ok:
            raise RuntimeError(
                f"Cache meta fin/fout/frame_gap mismatch with dataset "
                f"(cache={self.meta}, dataset=fin{self.base.fin}/"
                f"fout{self.base.fout}/gap{self.base.gap})."
            )

    def _cache_pt_path_for(self, ep_path: str) -> Path:
        # Derive <cache_dir>/episode_XXXXXX.stage1pred.pt from a parquet path.
        stem = Path(ep_path).stem  # e.g. "episode_001693"
        return self.cache_dir / f"{stem}.stage1pred.pt"

    def _build_index(self) -> dict[tuple[str, int], tuple[str, int]]:
        index: dict[tuple[str, int], tuple[str, int]] = {}
        # Collect the set of unique episode paths we need.
        ep_paths = {s[0] for s in self.base.samples}
        missing: list[str] = []
        for ep_path in sorted(ep_paths):
            pt_path = self._cache_pt_path_for(ep_path)
            if not pt_path.is_file():
                missing.append(str(pt_path))
                continue
            payload = torch.load(
                pt_path, map_location="cpu", weights_only=True, mmap=True,
            )
            self._pt_cache[str(pt_path)] = payload
            starts = payload["window_starts"].tolist()
            for row, start in enumerate(starts):
                index[(ep_path, int(start))] = (str(pt_path), int(row))
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} cache .pt files missing under {self.cache_dir}; "
                f"first few: {missing[:3]}. Re-run prebake."
            )
        if self._is_main:
            log.info(
                f"[stage2.cache] indexed {len(index)} cached windows across "
                f"{len(ep_paths)} episodes from {self.cache_dir}"
            )
        return index

    # ── Dataset API ──────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        ctx, tgt, actions, goal = self.base[idx]
        ep_path, start, _ = self.base.samples[idx]
        key = (ep_path, int(start))
        pt_path, row = self._sample_to_row[key]
        payload = self._pt_cache.get(pt_path)
        if payload is None:
            if len(self._pt_cache) >= self._CACHE_MAX:
                oldest = next(iter(self._pt_cache))
                del self._pt_cache[oldest]
            payload = torch.load(
                pt_path, map_location="cpu", weights_only=True, mmap=True,
            )
            self._pt_cache[pt_path] = payload
        pred_u8 = payload["pred_frames_u8"][row]      # (fout, H, W, C) uint8
        pred = pred_u8.to(torch.float32).permute(0, 3, 1, 2)
        pred = pred.mul_(1.0 / 127.5).sub_(1.0)        # -> (fout, C, H, W) in [-1,1]
        return ctx, tgt, actions, goal, pred

    # Expose a handful of attributes that upstream code might poke at so
    # drop-in wrapping is transparent (e.g. trainer.get_action_stats()).
    @property
    def test_tasks(self):
        return getattr(self.base, "test_tasks", ())

    def get_action_stats(self):
        return self.base.get_action_stats()


# ---------------------------------------------------------------------------
# Stage 2 base mixin: shared overrides for both 2.1 and 2.2
# ---------------------------------------------------------------------------


class _Stage2Mixin:
    """
    Overrides _eval_loader_loss to add the same two fixes applied in Stage 1:

      Fix 1 — ddp_barrier() before val
        Stage 2 val datasets are tiny (often < 5 episodes). Without a barrier,
        fast ranks enter the val DataLoader while slow ranks are still in the
        last train micro-batch. The subsequent ddp_all_reduce_scalar hangs.
        This is MORE dangerous here than in Stage 1 because the disproportion
        between train and val size is larger.

      Fix 2 — persistent_workers=False for val DataLoader
        Already applied at the base Trainer level (train_stage1.py line ~996),
        so Stage 2 inherits it automatically. The override here just documents
        why it matters even more for Stage 2 (val has so few batches that one
        stalled worker blocks the entire pass).

    Note: we do NOT override the val DataLoader construction — that fix lives
    in Trainer.__init__ and is inherited by both Stage2Part1Trainer and
    Stage2Part2Trainer automatically.
    """

    def _eval_loader_loss(self, loader):
        # FIX: barrier before val — same reason as Stage 1, but even more
        # critical here because Stage 2 val sets are tiny (1-2 episodes).
        # Any rank skew causes the all_reduce at end of _run_epoch to stall.
        ddp_barrier()
        raw = unwrap_model(self.model)
        if self.ema:
            self.ema.apply(raw)
        loss = self._run_epoch(loader, train=False)
        if self.ema:
            self.ema.restore(raw)
        return loss


# ---------------------------------------------------------------------------
# Stage 2.1 — Fine-tune ARVideoPatchTransformer on 3 test tasks.
# ---------------------------------------------------------------------------


class Stage2Part1Trainer(_Stage2Mixin, Trainer):
    """Stage 2.1: same objective/inputs/outputs as Stage 1, restricted to the
    three held-out test tasks with a 99/1 episode-level split.

    Inherits all Stage 1 fixes (NCCL timeout, val stall, checkpoint ordering)
    via Trainer. The only Stage-2.1-specific logic is:
      - dataset restricted to held-out tasks
      - optional warm-start from a Stage-1 checkpoint
    """

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


def _maybe_wrap_with_stage1_cache(
    train_ds: VideoFrameDataset,
    val_ds: VideoFrameDataset,
    cfg: DictConfig,
    backbone_sha1: str,
    is_main: bool,
):
    """Wrap the Stage 2 datasets with CachedStage1PredDataset iff
    ``cfg.train.idm.use_stage1_cache`` is set. Otherwise return unchanged.

    Binds the cache to ``backbone_sha1`` so mismatched weights fail loudly
    (guard against training Stage 2.2 on a stale cache after re-finetuning
    Stage 2.1).
    """
    use_cache = bool(
        OmegaConf.select(cfg, "train.idm.use_stage1_cache", default=False)
    )
    if not use_cache:
        return train_ds, val_ds

    cache_dir = OmegaConf.select(cfg, "train.idm.stage1_cache_dir", default=None)
    if not cache_dir:
        raise ValueError(
            "train.idm.use_stage1_cache=true requires train.idm.stage1_cache_dir."
        )
    require = bool(
        OmegaConf.select(cfg, "train.idm.require_cache_sha1", default=True)
    )
    cache_dir = Path(str(cache_dir)).expanduser()
    if is_main:
        log.info(
            f"Stage 2.2: using prebaked stage-1 predictions from {cache_dir} "
            f"(require_sha1_match={require})"
        )
    train_ds = CachedStage1PredDataset(
        train_ds, cache_dir, backbone_sha1,
        require_sha1_match=require, is_main=is_main,
    )
    val_ds = CachedStage1PredDataset(
        val_ds, cache_dir, backbone_sha1,
        require_sha1_match=require, is_main=is_main,
    )
    return train_ds, val_ds


class Stage2Part2Trainer(_Stage2Mixin, Trainer):
    """Stage 2.2: freeze the (Stage 2.1) ARVideoPatchTransformer and only
    train the IDM action head.

    Key differences from Stage 1 / Stage 2.1:
      - Model is _IDMModelDP wrapping the frozen backbone.
      - Only the IDM head parameters are trainable (freeze_stage1=True).
      - torch.compile is skipped (diffusion_policy vision stack not safe).
      - DDP buffer broadcasts disabled to avoid collective mismatches.
      - _forward unpacks a 3-tuple (pred, _, loss) from the IDM model.
      - Action stats are injected into the IDM head after data setup.
    """

    def _stage_tag(self) -> str:
        return "2.2"

    def _build_datasets(self):
        cfg = self.cfg
        val_fraction = float(
            OmegaConf.select(cfg, "data.stage2_val_fraction", default=0.01)
        )
        seed = int(OmegaConf.select(cfg, "seed", default=42))
        train_ds, val_ds, test_ds = build_heldout_task_datasets(
            cfg.data, cfg.model, self.is_main,
            val_fraction=val_fraction, seed=seed,
        )
        # Stage 2.2 may optionally read prebaked stage-1 predictions so the
        # frozen backbone's AR decode is skipped every step. Opt-in via
        # cfg.train.idm.use_stage1_cache.
        train_ds, val_ds = _maybe_wrap_with_stage1_cache(
            train_ds, val_ds, cfg,
            backbone_sha1=getattr(self, "_backbone_sha1", ""),
            is_main=self.is_main,
        )
        return train_ds, val_ds, test_ds

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
        # Stash backbone hash for the optional Stage-1 prediction cache
        # (A1). CachedStage1PredDataset will verify it at dataset build time.
        self._backbone_sha1 = compute_backbone_sha1(sd)
        if self.is_main:
            log.info(
                f"Stage 2.2 loaded backbone from {stage2_1_ckpt} ({src} weights) "
                f"sha1={self._backbone_sha1[:12]}"
            )

        wrapped = _IDMModelDP(
            raw_wm,
            n_actions=int(cfg.data.get("frame_gap", 0)),
            freeze_stage1=True,
            **OmegaConf.to_container(cfg.train.get("idm", {}), resolve=True),
        ).to(self.device)

        # prebuild_mask must be called on the wrapper, not raw_wm, because
        # the IDM wrapper delegates to the backbone and needs the mask cache
        # populated before DDP wrapping.
        wrapped.prebuild_mask(device=self.device, has_goal=has_goal)

        if self.is_main:
            n_trainable = sum(
                p.numel() for p in wrapped.parameters() if p.requires_grad
            ) / 1e6
            n_total = sum(p.numel() for p in wrapped.parameters()) / 1e6
            log.info(
                f"Stage 2.2 backbone frozen — "
                f"trainable {n_trainable:.2f}M / total {n_total:.2f}M"
            )
        return wrapped

    def _should_skip_compile(self) -> bool:
        if self.is_main:
            log.info(
                "Skipping torch.compile for stage 2.2 "
                "(robomimic/diffusion_policy vision stack is not compile-safe)."
            )
        return True

    def _ddp_broadcast_buffers(self) -> bool:
        # DP IDM has internal buffers (noise schedules, vision encoders) that
        # must NOT be broadcast-synced every forward — they are static after
        # init and syncing them causes spurious collective ops that race with
        # the frozen backbone's non-broadcast buffers.
        if self.world_size > 1:
            if hasattr(torch, "_dynamo"):
                torch._dynamo.config.optimize_ddp = False
            if self.is_main:
                log.info(
                    "Stage 2.2: disabling DDP buffer broadcasts and "
                    "TorchDynamo DDP optimizer to avoid collective/compile mismatches."
                )
            return False
        return True

    def _post_data_setup(self, train_ds) -> None:
        # Called from Trainer.__init__ BEFORE DDP / compile wrapping, so
        # self.model is still the raw IDM wrapper — unwrap_model is a no-op
        # here but kept for consistency with the rest of the codebase.
        raw = unwrap_model(self.model)
        if hasattr(raw, "set_action_stats") and hasattr(train_ds, "get_action_stats"):
            stats = train_ds.get_action_stats()
            raw.set_action_stats(stats)
            if self.ema is not None and hasattr(self.ema.shadow, "set_action_stats"):
                self.ema.shadow.set_action_stats(stats)
            if self.is_main:
                log.info("Loaded action stats into Stage 2.2 diffusion-policy IDM.")

    def _forward(
        self, model, context, target, actions, goal, *,
        return_pred_frames=True, cached_pred_frames=None,
    ):
        # IDM model returns (pred_frames, pred_actions, loss) — a 3-tuple.
        # The base Trainer._run_epoch expects _forward to return (pred, loss).
        # We drop pred_actions here; they are only needed during inference,
        # not during training where the loss is all that matters.
        # return_pred_frames is part of the base-class interface but has no
        # effect on the IDM wrapper (its inner Stage-1 forward is a separate
        # code path); accept-and-ignore keeps the signature compatible.
        del return_pred_frames
        pred, _, loss = model(
            context, target, actions, goal=goal,
            cached_pred_frames=cached_pred_frames,
        )
        return pred, loss


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


@hydra.main(config_path="configs", config_name="stage2_idm_config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    _apply_selected_task_overrides(cfg)
    OmegaConf.resolve(cfg)

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