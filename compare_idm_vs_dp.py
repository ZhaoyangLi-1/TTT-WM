"""Offline action-error comparison: Pure IDM vs original Diffusion Policy.

Both models are evaluated on the same held-out task with the same episode-level
val split (same seed + same val fraction), so every sample drawn from the Pure
IDM loader corresponds one-to-one with a sample from the DP loader (same
(episode, start_frame, action chunk)).

Pure IDM consumes the ground-truth next frame, so its action error should be
interpreted as an upper bound — the gap between DP and Pure IDM is the headroom
a perfect world model could still help compress.

Metrics reported per model
--------------------------
  * mean action MSE / L1 over all evaluated samples
  * per-dim MSE / L1 (shape = [action_dim])
  * per-step MSE / L1 across the action chunk (shape = [n_action_steps])

Outputs (per task, under <output-dir>/<task_slug>/)
---------------------------------------------------
  * summary.json              — scalar + per-dim + per-episode rollout metrics
  * per_dim.png               — per-action-dim MSE bars (IDM vs DP)
  * rollout_per_step.png      — per-episode-timestep MSE curve (IDM vs DP)
  * rollout_per_episode.png   — per-episode mean-MSE bars (IDM vs DP)

Root-level outputs
------------------
  * overall_summary.json      — cross-task aggregate
  * combined_rollout.png      — per-task rollout curves side-by-side

Example
-------
    python compare_idm_vs_dp.py \
        --idm-checkpoint  /.../pure_idm/.../checkpoints/best.pt \
        --dp-checkpoint   /.../original_diffusion_policy/.../checkpoints/latest.ckpt \
        --dataset-root    /scr2/zhaoyang/libero_wm \
        --task            "KITCHEN_SCENE10: put the butter at the back in the top drawer of the cabinet and close it" \
        --output-dir      /scr2/zhaoyang/TTT-WM-outputs/compare/scene10 \
        --batch-size 64 --num-batches 200
"""

from __future__ import annotations

import argparse
import copy
import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import dill
import hydra
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from cosmos_model import ARPatchConfig
from dp.common import (
    align_action_tensors,
    dict_apply,
    load_state_dict_flexible,
    resolve_checkpoint_path,
    resolve_device,
    strip_state_dict_prefixes,
)
from dp.runtime import configure_diffusion_policy_path, register_omegaconf_resolvers
from pure_idm import PureInverseDynamicsModel, PureInverseDynamicsModelDP
from train_stage1 import _clean_state_dict
from train_stage2 import build_heldout_task_datasets

register_omegaconf_resolvers()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline action-error comparison between Pure IDM and DP."
    )
    parser.add_argument(
        "--idm-checkpoint",
        type=str,
        nargs="+",
        required=True,
        help="One or more Pure IDM checkpoints (one per held-out task).",
    )
    parser.add_argument(
        "--dp-checkpoint",
        type=str,
        nargs="+",
        required=True,
        help="DP checkpoints matching --idm-checkpoint 1:1 by position.",
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        default="/scr2/zhaoyang/libero_wm",
        help="Root of the LIBERO-WM dataset (must contain meta/test_tasks.json).",
    )
    parser.add_argument(
        "--task",
        type=str,
        nargs="+",
        default=None,
        help=(
            "Held-out task names (optional). If omitted, auto-read from each ckpt's "
            "cfg.data.selected_task. If supplied, length must match --idm-checkpoint."
        ),
    )
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--num-batches",
        type=int,
        default=0,
        help="0 = iterate the full val split (recommended for small val splits).",
    )
    parser.add_argument(
        "--eval-val-fraction",
        type=float,
        default=None,
        help=(
            "Override the val split fraction used for rollout. Overrides both "
            "data.stage2_val_fraction (IDM) and task.dataset.val_ratio (DP); "
            "both share seed+shuffle so the val sets stay nested and aligned. "
            "E.g. 0.2 on a 50-episode task gives ~10 val episodes. WARNING: the "
            "extra episodes were in the model's training set — numbers will be "
            "optimistic compared to the default held-out val split. Default: "
            "inherit from ckpt cfg."
        ),
    )
    parser.add_argument(
        "--eval-all-episodes",
        action="store_true",
        help=(
            "Shortcut: set --eval-val-fraction to 1.0 so every episode of the "
            "task is rolled out (includes training episodes)."
        ),
    )
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--use-ema", dest="use_ema", action="store_true")
    parser.add_argument("--no-ema", dest="use_ema", action="store_false")
    parser.set_defaults(use_ema=True)
    parser.add_argument("--diffusion-policy-src", type=str, default=None)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Pure IDM loading
# ---------------------------------------------------------------------------


def _build_idm_model_cfg(cfg: DictConfig) -> ARPatchConfig:
    mcfg = cfg.model
    kwargs = dict(
        resolution=int(mcfg.resolution),
        num_channels=int(mcfg.num_channels),
        patch_size=int(mcfg.patch_size),
        d_model=int(mcfg.d_model),
        n_heads=int(mcfg.n_heads),
        n_layers=int(mcfg.n_layers),
        mlp_ratio=float(mcfg.mlp_ratio),
        dropout=float(mcfg.dropout),
        frames_in=int(mcfg.frames_in),
        frames_out=int(mcfg.frames_out),
        action_dim=int(cfg.data.get("action_dim", 7)),
    )
    if "qk_norm" in mcfg:
        kwargs["qk_norm"] = bool(mcfg.qk_norm)
    if "parallel_attn" in mcfg:
        kwargs["parallel_attn"] = bool(mcfg.parallel_attn)
    return ARPatchConfig(**kwargs)


def read_idm_payload(ckpt_path: Path) -> tuple[dict, DictConfig, str]:
    print(f"[compare] reading Pure IDM ckpt: {ckpt_path}", flush=True)
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg_dict = payload.get("cfg")
    if cfg_dict is None:
        raise KeyError(f"Pure IDM checkpoint {ckpt_path} is missing embedded `cfg`.")
    cfg = OmegaConf.create(cfg_dict)
    task = OmegaConf.select(cfg, "data.selected_task", default=None)
    if task in (None, "", "None"):
        raise ValueError(
            f"Pure IDM checkpoint {ckpt_path} does not record cfg.data.selected_task."
        )
    return payload, cfg, str(task)


def load_pure_idm(
    payload: dict,
    cfg: DictConfig,
    device: torch.device,
    use_ema: bool,
    task: str,
    dataset_root_override: str,
) -> tuple[torch.nn.Module, DictConfig, str]:

    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    cfg.data.root = str(dataset_root_override)
    cfg.data.selected_task = task
    cfg.data.test_tasks = [task]
    cfg.data.test_task_count = 1

    print("[compare] building Pure IDM model", flush=True)
    arpatch_cfg = _build_idm_model_cfg(cfg)
    n_actions = int(cfg.data.get("frame_gap", 0))
    if n_actions <= 0:
        raise ValueError(f"cfg.data.frame_gap must be positive, got {n_actions}.")

    idm_type = str(cfg.train.get("idm_type", "mlp")).lower()
    is_dp = idm_type in {"dp", "diffusion", "diffusion_policy"}
    if is_dp:
        idm_dp_kwargs = OmegaConf.to_container(
            cfg.train.get("idm_dp", {}), resolve=True
        )
        model = PureInverseDynamicsModelDP(
            arpatch_cfg, n_actions=n_actions, **idm_dp_kwargs
        )
    else:
        model = PureInverseDynamicsModel(arpatch_cfg, n_actions=n_actions)
    model.to(device)
    model.prebuild_mask(device=device, has_goal=False)

    if is_dp:
        # Register normalizer parameters so load_state_dict can populate them.
        # Values will be overwritten by the checkpoint; we only need shape.
        from train_stage2 import HeldoutTaskSplitDataset  # local import for clarity

        print("[compare] computing/loading IDM action stats", flush=True)
        val_fraction = float(
            OmegaConf.select(cfg, "data.stage2_val_fraction", default=0.02)
        )
        seed = int(OmegaConf.select(cfg, "seed", default=42))
        train_ds = HeldoutTaskSplitDataset(
            cfg.data,
            cfg.model,
            "train",
            val_fraction=val_fraction,
            seed=seed,
            is_main=True,
        )
        stats = train_ds.get_action_stats()
        model.set_action_stats(stats)

    source = "live"
    sd = None
    if use_ema and isinstance(payload.get("ema"), dict):
        shadow = payload["ema"].get("shadow", payload["ema"])
        if isinstance(shadow, dict) and shadow:
            sd = shadow
            source = "ema"
    if sd is None:
        sd = payload["model"]
    sd = _clean_state_dict(sd)

    print(f"[compare] loading Pure IDM weights ({source})", flush=True)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"[pure_idm] missing keys ({len(missing)}): e.g. {missing[:3]}")
    if unexpected:
        print(f"[pure_idm] unexpected keys ({len(unexpected)}): e.g. {unexpected[:3]}")

    # The DP policy's normalizer uses a custom _load_from_state_dict that REPLACES
    # its internal ParameterDict with tensors read from the checkpoint (which was
    # loaded with map_location="cpu"). This silently reverts the prior .to(device)
    # on the normalizer, causing obs to be pulled back to CPU inside _normalize
    # and crashing at the first GPU conv. Re-apply .to(device) to recover.
    model.to(device)
    model.eval()
    return model, cfg, source


def build_idm_val_loader(
    cfg: DictConfig,
    batch_size: int,
    num_workers: int,
    val_fraction_override: float | None = None,
) -> DataLoader:
    val_fraction = float(
        OmegaConf.select(cfg, "data.stage2_val_fraction", default=0.02)
    )
    if val_fraction_override is not None:
        val_fraction = float(val_fraction_override)
    seed = int(OmegaConf.select(cfg, "seed", default=42))
    _train_ds, val_ds, _ = build_heldout_task_datasets(
        cfg.data, cfg.model, is_main=True, val_fraction=val_fraction, seed=seed
    )
    loader = DataLoader(
        val_ds,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=True,
        drop_last=False,
        persistent_workers=False,
    )
    return loader


# ---------------------------------------------------------------------------
# DP loading (mirrors test_dp.py)
# ---------------------------------------------------------------------------


def read_dp_payload(ckpt_path: Path) -> tuple[dict, Any, str]:
    print(f"[compare] reading DP ckpt: {ckpt_path}", flush=True)
    payload = torch.load(
        ckpt_path, map_location="cpu", pickle_module=dill, weights_only=False
    )
    cfg = payload.get("cfg")
    if cfg is None:
        raise KeyError(f"DP checkpoint {ckpt_path} is missing embedded `cfg`.")
    task = OmegaConf.select(cfg, "data.selected_task", default=None)
    if task in (None, "", "None"):
        task = OmegaConf.select(cfg, "task.dataset.task_filter", default=None)
    if task in (None, "", "None"):
        raise ValueError(
            f"DP checkpoint {ckpt_path} does not record a training task "
            "(cfg.data.selected_task / cfg.task.dataset.task_filter)."
        )
    return payload, cfg, str(task)


def load_dp_policy(
    payload: dict,
    cfg: Any,
    device: torch.device,
    use_ema: bool,
    task: str,
    dataset_root_override: str,
) -> tuple[Any, DictConfig, str]:
    cfg = copy.deepcopy(cfg)

    cfg.dataset_root = str(dataset_root_override)
    if OmegaConf.select(cfg, "task.dataset.dataset_root") is not None:
        cfg.task.dataset.dataset_root = str(dataset_root_override)
    cfg.data.selected_task = task
    if OmegaConf.select(cfg, "task.dataset.task_filter") is not None:
        cfg.task.dataset.task_filter = task
    OmegaConf.resolve(cfg)

    print("[compare] building DP policy", flush=True)
    policy = hydra.utils.instantiate(cfg.policy)

    state_dicts = payload.get("state_dicts", {})
    source = None
    sd = None
    if use_ema and isinstance(state_dicts.get("ema_model"), dict):
        sd = state_dicts["ema_model"]
        source = "ema_model"
    if sd is None and isinstance(state_dicts.get("model"), dict):
        sd = state_dicts["model"]
        source = "model"
    if sd is None:
        raise KeyError(
            "DP checkpoint does not contain `state_dicts['model']` or `state_dicts['ema_model']`."
        )
    print(f"[compare] loading DP weights ({source})", flush=True)
    load_state_dict_flexible(policy, strip_state_dict_prefixes(sd))

    policy.to(device)
    policy.eval()
    return policy, cfg, source


def build_dp_val_loader(
    cfg: DictConfig,
    batch_size: int,
    num_workers: int,
    val_ratio_override: float | None = None,
) -> DataLoader:
    if val_ratio_override is not None:
        cfg = copy.deepcopy(cfg)
        if OmegaConf.select(cfg, "task.dataset.val_ratio") is not None:
            cfg.task.dataset.val_ratio = float(val_ratio_override)
        if OmegaConf.select(cfg, "data.val_ratio") is not None:
            cfg.data.val_ratio = float(val_ratio_override)
    dataset = hydra.utils.instantiate(cfg.task.dataset)
    dataset = dataset.get_validation_dataset()
    loader = DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=True,
        drop_last=False,
        persistent_workers=False,
    )
    return loader


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------


@torch.no_grad()
def idm_predict(model: torch.nn.Module, batch: tuple, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    ctx, tgt, actions, goal = batch
    ctx = ctx.to(device, non_blocking=True)
    tgt = tgt.to(device, non_blocking=True)
    actions = actions.to(device, non_blocking=True)
    _pred_frames, pred_actions = model.generate(ctx, tgt, goal=None)
    if pred_actions is None:
        raise RuntimeError("Pure IDM generate() returned no actions.")
    return pred_actions.float(), actions.float()


@torch.no_grad()
def dp_predict(policy: Any, batch: dict, device: torch.device, n_obs_steps: int) -> tuple[torch.Tensor, torch.Tensor]:
    batch = dict_apply(batch, lambda tensor: tensor.to(device, non_blocking=True))
    result = policy.predict_action(batch["obs"])
    pred_action = result.get("action_pred", result["action"])
    pred_action, gt_action = align_action_tensors(pred_action, batch["action"], n_obs_steps)
    return pred_action.float(), gt_action.float()


# ---------------------------------------------------------------------------
# Metric accumulator
# ---------------------------------------------------------------------------


class ActionMetricAccumulator:
    """Tracks action error decomposed per-sample / per-step / per-dim."""

    def __init__(self, n_action_steps: int, action_dim: int):
        self.n_action_steps = int(n_action_steps)
        self.action_dim = int(action_dim)
        self._se_step_dim = np.zeros((self.n_action_steps, self.action_dim), dtype=np.float64)
        self._ae_step_dim = np.zeros((self.n_action_steps, self.action_dim), dtype=np.float64)
        self._count_samples = 0  # B across all batches
        self._gt_se = 0.0        # for normalizing by GT variance if desired

    def update(self, pred: torch.Tensor, gt: torch.Tensor) -> None:
        if pred.shape != gt.shape:
            raise ValueError(f"Shape mismatch: pred {pred.shape} vs gt {gt.shape}")
        if pred.ndim != 3:
            raise ValueError(f"Expected (B, T, D), got {tuple(pred.shape)}")

        diff = (pred - gt).detach().cpu().double().numpy()  # (B, T, D)
        self._se_step_dim += np.sum(diff ** 2, axis=0)
        self._ae_step_dim += np.sum(np.abs(diff), axis=0)
        self._count_samples += int(pred.shape[0])
        self._gt_se += float(np.sum(gt.detach().cpu().double().numpy() ** 2))

    def compute(self) -> dict[str, Any]:
        if self._count_samples == 0:
            raise RuntimeError("No samples accumulated.")
        n = float(self._count_samples)
        per_step_dim_mse = self._se_step_dim / n
        per_step_dim_l1 = self._ae_step_dim / n
        per_step_mse = per_step_dim_mse.mean(axis=1)
        per_step_l1 = per_step_dim_l1.mean(axis=1)
        per_dim_mse = per_step_dim_mse.mean(axis=0)
        per_dim_l1 = per_step_dim_l1.mean(axis=0)
        return {
            "num_samples": self._count_samples,
            "mean_mse": float(per_step_dim_mse.mean()),
            "mean_l1": float(per_step_dim_l1.mean()),
            "per_step_mse": per_step_mse.tolist(),
            "per_step_l1": per_step_l1.tolist(),
            "per_dim_mse": per_dim_mse.tolist(),
            "per_dim_l1": per_dim_l1.tolist(),
        }


# ---------------------------------------------------------------------------
# Episode-level rollout
# ---------------------------------------------------------------------------


_EP_IDX_RE = re.compile(r"episode_(\d+)\.parquet")


def _ep_idx_from_sample(sample: Any) -> int:
    """Normalize IDM (ep_path, start, length) / DP (ep_idx, start) to ep_idx."""
    if isinstance(sample, tuple) and len(sample) == 2:
        return int(sample[0])
    if isinstance(sample, tuple) and len(sample) >= 3:
        m = _EP_IDX_RE.search(str(sample[0]))
        if m is None:
            raise ValueError(f"Cannot parse episode idx from IDM sample {sample!r}")
        return int(m.group(1))
    raise ValueError(f"Unexpected sample structure: {sample!r}")


def _sample_start(sample: Any) -> int:
    if isinstance(sample, tuple) and len(sample) >= 2:
        return int(sample[1])
    raise ValueError(f"Unexpected sample structure: {sample!r}")


def rollout_per_episode(
    loader: DataLoader,
    predict_fn: Callable[[Any, torch.device], tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
    max_batches: int | None,
    desc: str,
) -> dict[int, dict[str, np.ndarray]]:
    """Run model over each window in loader order (shuffle=False assumed).

    For each chunk prediction we keep the full per-chunk-step MSE vector (not
    averaged over the chunk) so downstream aggregation can produce a true
    per-timestep curve where overlapping chunks get averaged.

    Returns
    -------
    {episode_idx: {
        "starts":     np.ndarray[N]       window start frame in the episode
        "chunk_mse":  np.ndarray[N, T]    per-chunk-step MSE, dim-averaged
        "chunk_l1":   np.ndarray[N, T]    per-chunk-step L1, dim-averaged
    }} sorted by start.
    """
    dataset = loader.dataset
    samples = dataset.samples  # list of tuples, matches DataLoader iteration order

    per_ep: dict[int, dict[str, list]] = defaultdict(
        lambda: {"starts": [], "chunk_mse": [], "chunk_l1": []}
    )
    sample_cursor = 0
    batches_done = 0
    total_batches = len(loader) if max_batches is None else min(max_batches, len(loader))
    pbar = tqdm(total=total_batches, desc=desc, unit="batch", dynamic_ncols=True)
    with torch.no_grad():
        for batch in loader:
            pred, gt = predict_fn(batch, device)  # (B, T, D)
            diff = (pred.float() - gt.float()).detach().cpu().double().numpy()
            chunk_mse = (diff ** 2).mean(axis=2)  # (B, T)
            chunk_l1 = np.abs(diff).mean(axis=2)  # (B, T)
            B = chunk_mse.shape[0]
            for i in range(B):
                s = samples[sample_cursor + i]
                ep = _ep_idx_from_sample(s)
                per_ep[ep]["starts"].append(_sample_start(s))
                per_ep[ep]["chunk_mse"].append(chunk_mse[i])
                per_ep[ep]["chunk_l1"].append(chunk_l1[i])
            sample_cursor += B
            batches_done += 1
            pbar.update(1)
            if max_batches is not None and batches_done >= max_batches:
                break
    pbar.close()

    out: dict[int, dict[str, np.ndarray]] = {}
    for ep, rec in per_ep.items():
        order = np.argsort(rec["starts"])
        out[ep] = {
            "starts": np.asarray(rec["starts"], dtype=np.int64)[order],
            "chunk_mse": np.stack(rec["chunk_mse"], axis=0)[order],  # (N, T)
            "chunk_l1": np.stack(rec["chunk_l1"], axis=0)[order],    # (N, T)
        }
    return out


def _scatter_chunks_to_timesteps(
    starts: np.ndarray,
    chunk_vals: np.ndarray,  # (N, T)
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Place each chunk's per-step values at their absolute episode timesteps
    (s, s+1, ..., s+T-1). Overlapping chunks at the same timestep are averaged.

    Returns (timesteps, mean, count) where timesteps is 0..max_t inclusive,
    and mean entries for uncovered timesteps are NaN.
    """
    N, T = chunk_vals.shape
    if N == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.float64), np.array([], dtype=np.int64)
    max_t = int(starts[-1] + T - 1)
    if max_t < 0:
        max_t = 0
    length = max_t + 1
    vsum = np.zeros(length, dtype=np.float64)
    vcount = np.zeros(length, dtype=np.int64)
    # vectorised scatter-add
    for j in range(T):
        idx = starts + j
        # clip in the unlikely case of padding beyond episode end
        mask = (idx >= 0) & (idx < length)
        np.add.at(vsum, idx[mask], chunk_vals[:, j][mask])
        np.add.at(vcount, idx[mask], 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(vcount > 0, vsum / np.maximum(vcount, 1), np.nan)
    return np.arange(length, dtype=np.int64), mean, vcount


def summarize_rollout(
    per_ep: dict[int, dict[str, np.ndarray]],
) -> dict[str, Any]:
    """Aggregate per-episode rollout into per-timestep curves (averaged over
    overlapping chunks), episode-level scalars, and a cross-episode per-step
    mean ± std curve.
    """
    if not per_ep:
        return {
            "num_episodes": 0,
            "num_windows": 0,
            "mean_mse": float("nan"),
            "mean_l1": float("nan"),
            "per_episode": {},
            "per_step_mean_mse": [],
            "per_step_std_mse": [],
            "per_step_counts": [],
        }
    per_ep_stats: dict[int, dict[str, Any]] = {}
    per_ep_timestep_mse: dict[int, np.ndarray] = {}
    max_len = 0
    all_mse_vals: list[float] = []
    all_l1_vals: list[float] = []
    total_windows = 0
    for ep, rec in per_ep.items():
        starts = rec["starts"]
        chunk_mse = rec["chunk_mse"]  # (N, T)
        chunk_l1 = rec["chunk_l1"]
        total_windows += chunk_mse.shape[0]
        _, mse_curve, _ = _scatter_chunks_to_timesteps(starts, chunk_mse)
        _, l1_curve, _ = _scatter_chunks_to_timesteps(starts, chunk_l1)
        per_ep_timestep_mse[int(ep)] = mse_curve

        # per-episode scalar = mean over covered timesteps
        valid = ~np.isnan(mse_curve)
        ep_mse_mean = float(mse_curve[valid].mean()) if valid.any() else float("nan")
        ep_l1_mean = float(l1_curve[~np.isnan(l1_curve)].mean()) if np.any(~np.isnan(l1_curve)) else float("nan")

        per_ep_stats[int(ep)] = {
            "num_windows": int(chunk_mse.shape[0]),
            "num_timesteps": int(valid.sum()),
            "mean_mse": ep_mse_mean,
            "mean_l1": ep_l1_mean,
            "per_step_mse": mse_curve.tolist(),
            "per_step_l1": l1_curve.tolist(),
            "window_starts": starts.tolist(),
        }
        if valid.any():
            all_mse_vals.extend(mse_curve[valid].tolist())
            valid_l1 = ~np.isnan(l1_curve)
            all_l1_vals.extend(l1_curve[valid_l1].tolist())
        max_len = max(max_len, mse_curve.size)

    # Cross-episode per-timestep mean ± std (aligned to t=0..max_len-1).
    step_sum = np.zeros(max_len, dtype=np.float64)
    step_sqsum = np.zeros(max_len, dtype=np.float64)
    step_count = np.zeros(max_len, dtype=np.int64)
    for mse_curve in per_ep_timestep_mse.values():
        n = mse_curve.size
        valid = ~np.isnan(mse_curve)
        step_sum[:n][valid] += mse_curve[valid]
        step_sqsum[:n][valid] += mse_curve[valid] ** 2
        step_count[:n][valid] += 1
    with np.errstate(invalid="ignore", divide="ignore"):
        step_mean = np.where(step_count > 0, step_sum / np.maximum(step_count, 1), np.nan)
        step_var = np.where(
            step_count > 0,
            step_sqsum / np.maximum(step_count, 1) - step_mean ** 2,
            np.nan,
        )
        step_std = np.sqrt(np.maximum(step_var, 0.0))

    return {
        "num_episodes": len(per_ep),
        "num_windows": int(total_windows),
        "mean_mse": float(np.mean(all_mse_vals)) if all_mse_vals else float("nan"),
        "mean_l1": float(np.mean(all_l1_vals)) if all_l1_vals else float("nan"),
        "per_episode": per_ep_stats,
        "per_step_mean_mse": step_mean.tolist(),
        "per_step_std_mse": step_std.tolist(),
        "per_step_counts": step_count.tolist(),
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _task_title(task: str) -> str:
    """Strip the `SCENE_NAME:` prefix from a task string for plot titles."""
    _, _, rest = task.partition(":")
    return rest.strip() if rest else task.strip()


def plot_comparison(
    idm_metrics: dict[str, Any],
    dp_metrics: dict[str, Any],
    output_dir: Path,
) -> list[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[WARN] skipping plots: {exc}")
        return []

    saved: list[str] = []

    # NOTE: the chunk-indexed per-step plot is superseded by
    # `rollout_per_step.png`, which plots per episode timestep. Only the
    # per-dim plot is still emitted here.

    # Per-dim MSE
    dims = np.arange(len(idm_metrics["per_dim_mse"]))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(dims - width / 2, idm_metrics["per_dim_mse"], width, label="IDM")
    ax.bar(dims + width / 2, dp_metrics["per_dim_mse"], width, label="Diffusion Policy")
    ax.set_xlabel("action dim")
    ax.set_ylabel("MSE")
    ax.set_title("Per-dim action MSE (lower is better)")
    ax.set_xticks(dims)
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend()
    fig.tight_layout()
    dim_path = output_dir / "per_dim.png"
    fig.savefig(dim_path, dpi=150)
    plt.close(fig)
    saved.append(str(dim_path))

    return saved


def plot_rollout_per_step(
    task: str,
    idm_roll: dict[str, Any],
    dp_roll: dict[str, Any],
    output_dir: Path,
) -> list[str]:
    """Per-task figure: x=window index in episode, y=MSE; mean ± std across eps."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[WARN] skipping rollout plot: {exc}")
        return []

    saved: list[str] = []

    def _plot(ax, roll, color, label):
        mean = np.asarray(roll["per_step_mean_mse"], dtype=np.float64)
        std = np.asarray(roll["per_step_std_mse"], dtype=np.float64)
        if mean.size == 0:
            return
        x = np.arange(mean.size)
        ax.plot(x, mean, "-", color=color, label=f"{label} (mean)")
        ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.15)
        for ep_id, ep_stat in roll.get("per_episode", {}).items():
            y = np.asarray(ep_stat["per_step_mse"], dtype=np.float64)
            ax.plot(np.arange(y.size), y, "-", color=color, alpha=0.25, linewidth=0.8)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    _plot(ax, idm_roll, "tab:blue", "IDM")
    _plot(ax, dp_roll, "tab:orange", "Diffusion Policy")
    ax.set_xlabel("episode timestep (frame)")
    ax.set_ylabel("per-step action MSE")
    ax.set_title(f"Per-timestep action error along episode — {_task_title(task)}")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    p = output_dir / "rollout_per_step.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    saved.append(str(p))

    # Per-episode bar chart
    ep_ids = sorted(
        set(idm_roll.get("per_episode", {}).keys())
        | set(dp_roll.get("per_episode", {}).keys())
    )
    if ep_ids:
        width = 0.4
        x = np.arange(len(ep_ids))
        idm_vals = [idm_roll["per_episode"].get(e, {}).get("mean_mse", np.nan) for e in ep_ids]
        dp_vals = [dp_roll["per_episode"].get(e, {}).get("mean_mse", np.nan) for e in ep_ids]
        fig, ax = plt.subplots(figsize=(max(6, 0.6 * len(ep_ids) + 3), 4))
        ax.bar(x - width / 2, idm_vals, width, label="IDM", color="tab:blue")
        ax.bar(x + width / 2, dp_vals, width, label="Diffusion Policy", color="tab:orange")
        ax.set_xticks(x)
        ax.set_xticklabels([str(e) for e in ep_ids], rotation=45, ha="right")
        ax.set_xlabel("episode index")
        ax.set_ylabel("episode-level mean MSE")
        ax.set_title(f"Episode-level MSE — {_task_title(task)}")
        ax.grid(True, alpha=0.3, axis="y")
        ax.legend()
        fig.tight_layout()
        p2 = output_dir / "rollout_per_episode.png"
        fig.savefig(p2, dpi=150)
        plt.close(fig)
        saved.append(str(p2))

    return saved


def plot_combined_rollout(
    per_task: list[dict[str, Any]],
    output_path: Path,
) -> str | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[WARN] skipping combined rollout plot: {exc}")
        return None
    if not per_task:
        return None

    n = len(per_task)
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 4.2), sharey=False)
    if n == 1:
        axes = [axes]
    for ax, tr in zip(axes, per_task):
        for roll, color, label in (
            (tr["idm_rollout"], "tab:blue", "IDM"),
            (tr["dp_rollout"], "tab:orange", "Diffusion Policy"),
        ):
            mean = np.asarray(roll["per_step_mean_mse"], dtype=np.float64)
            std = np.asarray(roll["per_step_std_mse"], dtype=np.float64)
            if mean.size == 0:
                continue
            x = np.arange(mean.size)
            ax.plot(x, mean, "-", color=color, label=label)
            ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.15)
        ax.set_xlabel("episode timestep (frame)")
        ax.set_ylabel("per-step action MSE")
        title = _task_title(tr["task"])
        if len(title) > 60:
            title = title[:57] + "..."
        ax.set_title(title, fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Per-episode rollout across held-out tasks")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return str(output_path)


# ---------------------------------------------------------------------------
# Per-task evaluation
# ---------------------------------------------------------------------------


def evaluate_task(
    idm_ckpt_arg: str,
    dp_ckpt_arg: str,
    task_override: str | None,
    args: argparse.Namespace,
    device: torch.device,
    output_root: Path,
) -> dict[str, Any]:
    idm_ckpt = resolve_checkpoint_path(idm_ckpt_arg)
    dp_ckpt = resolve_checkpoint_path(dp_ckpt_arg)

    idm_payload, idm_cfg_raw, idm_trained_task = read_idm_payload(idm_ckpt)
    dp_payload, dp_cfg_raw, dp_trained_task = read_dp_payload(dp_ckpt)

    if idm_trained_task != dp_trained_task:
        raise ValueError(
            f"Checkpoint task mismatch: Pure IDM trained on {idm_trained_task!r}, "
            f"DP trained on {dp_trained_task!r}. Pair ckpts from the same task."
        )
    if task_override is not None and task_override != idm_trained_task:
        raise ValueError(
            f"--task={task_override!r} does not match the task both ckpts were "
            f"trained on ({idm_trained_task!r})."
        )
    task = idm_trained_task

    task_slug = task.replace(":", "").replace(" ", "_")
    output_dir = output_root / task_slug
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[compare] idm ckpt = {idm_ckpt}", flush=True)
    print(f"[compare]  dp ckpt = {dp_ckpt}", flush=True)
    print(f"[compare]    task  = {task}", flush=True)
    print(f"[compare] output   = {output_dir}", flush=True)

    idm_model, idm_cfg, idm_source = load_pure_idm(
        idm_payload, idm_cfg_raw, device, args.use_ema, task, args.dataset_root
    )
    dp_policy, dp_cfg, dp_source = load_dp_policy(
        dp_payload, dp_cfg_raw, device, args.use_ema, task, args.dataset_root
    )

    frame_gap_idm = int(idm_cfg.data.get("frame_gap", 0))
    frame_gap_dp = int(OmegaConf.select(dp_cfg, "horizon", default=0))
    if frame_gap_idm != frame_gap_dp:
        raise ValueError(
            f"frame_gap mismatch: IDM={frame_gap_idm}, DP horizon={frame_gap_dp}. "
            "The two models must be trained with the same action-chunk length."
        )
    val_frac_idm = float(
        OmegaConf.select(idm_cfg, "data.stage2_val_fraction", default=0.02)
    )
    val_ratio_dp = float(OmegaConf.select(dp_cfg, "data.val_ratio", default=0.02))
    if abs(val_frac_idm - val_ratio_dp) > 1e-6:
        print(
            f"[WARN] val fractions differ (IDM={val_frac_idm}, DP={val_ratio_dp}) — "
            "sample alignment across the two loaders may break."
        )
    seed_idm = int(OmegaConf.select(idm_cfg, "seed", default=42))
    seed_dp = int(OmegaConf.select(dp_cfg, "training.seed", default=42))
    if seed_idm != seed_dp:
        print(
            f"[WARN] seeds differ (IDM={seed_idm}, DP={seed_dp}) — "
            "val episode shuffles may not match."
        )

    val_fraction_override: float | None = None
    if args.eval_all_episodes:
        val_fraction_override = 1.0
    elif args.eval_val_fraction is not None:
        val_fraction_override = float(args.eval_val_fraction)
    if val_fraction_override is not None:
        print(
            f"[compare] overriding val fraction -> {val_fraction_override} "
            "(expands rollout set; includes episodes the model trained on)",
            flush=True,
        )

    print("[compare] building IDM val loader", flush=True)
    idm_loader = build_idm_val_loader(
        idm_cfg, args.batch_size, args.num_workers,
        val_fraction_override=val_fraction_override,
    )
    print(
        f"[compare]   IDM val: {len(idm_loader.dataset)} windows / {len(idm_loader)} batches",
        flush=True,
    )
    print("[compare] building DP val loader", flush=True)
    dp_loader = build_dp_val_loader(
        dp_cfg, args.batch_size, args.num_workers,
        val_ratio_override=val_fraction_override,
    )
    print(
        f"[compare]   DP  val: {len(dp_loader.dataset)} windows / {len(dp_loader)} batches",
        flush=True,
    )
    if len(idm_loader.dataset) != len(dp_loader.dataset):
        print(
            f"[WARN] val size mismatch — IDM windows={len(idm_loader.dataset)}, "
            f"DP windows={len(dp_loader.dataset)}. Metrics computed independently."
        )

    action_dim = int(idm_cfg.data.get("action_dim", 7))
    n_obs_steps_dp = int(OmegaConf.select(dp_cfg, "n_obs_steps", default=1))
    max_batches = int(args.num_batches) if args.num_batches > 0 else None

    # --- Aggregate per-step-within-chunk metrics (original behaviour) ---
    idm_acc = ActionMetricAccumulator(frame_gap_idm, action_dim)
    dp_acc = ActionMetricAccumulator(frame_gap_dp, action_dim)

    # --- Per-episode rollout: group windows by episode, track per-step MSE ---
    def _idm_pred_fn(batch, dev):
        pred, gt = idm_predict(idm_model, batch, dev)
        idm_acc.update(pred, gt)
        return pred, gt

    def _dp_pred_fn(batch, dev):
        pred, gt = dp_predict(dp_policy, batch, dev, n_obs_steps_dp)
        dp_acc.update(pred, gt)
        return pred, gt

    idm_per_ep = rollout_per_episode(
        idm_loader, _idm_pred_fn, device, max_batches, "[compare:IDM rollout]"
    )
    dp_per_ep = rollout_per_episode(
        dp_loader, _dp_pred_fn, device, max_batches, "[compare:DP  rollout]"
    )

    idm_metrics = idm_acc.compute() if idm_acc._count_samples else None
    dp_metrics = dp_acc.compute() if dp_acc._count_samples else None
    idm_rollout = summarize_rollout(idm_per_ep)
    dp_rollout = summarize_rollout(dp_per_ep)

    gap = None
    if idm_metrics is not None and dp_metrics is not None:
        gap_mse = [d - i for d, i in zip(dp_metrics["per_step_mse"], idm_metrics["per_step_mse"])]
        gap_l1 = [d - i for d, i in zip(dp_metrics["per_step_l1"], idm_metrics["per_step_l1"])]
        gap_dim_mse = [d - i for d, i in zip(dp_metrics["per_dim_mse"], idm_metrics["per_dim_mse"])]
        gap = {
            "mean_mse": dp_metrics["mean_mse"] - idm_metrics["mean_mse"],
            "mean_l1": dp_metrics["mean_l1"] - idm_metrics["mean_l1"],
            "per_step_mse": gap_mse,
            "per_step_l1": gap_l1,
            "per_dim_mse": gap_dim_mse,
        }

    summary: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "task": task,
        "dataset_root": args.dataset_root,
        "frame_gap": frame_gap_idm,
        "action_dim": action_dim,
        "idm_checkpoint": str(idm_ckpt),
        "dp_checkpoint": str(dp_ckpt),
        "idm_weight_source": idm_source,
        "dp_weight_source": dp_source,
        "idm": idm_metrics,
        "dp": dp_metrics,
        "gap_dp_minus_idm": gap,
        "idm_rollout": idm_rollout,
        "dp_rollout": dp_rollout,
        "visualizations": [],
    }

    vis = []
    if idm_metrics is not None and dp_metrics is not None:
        vis.extend(plot_comparison(idm_metrics, dp_metrics, output_dir))
    vis.extend(plot_rollout_per_step(task, idm_rollout, dp_rollout, output_dir))
    summary["visualizations"] = vis

    out_json = output_dir / "summary.json"
    out_json.write_text(json.dumps(summary, indent=2))

    print()
    print("=" * 70)
    print(f"task: {task}")
    if idm_metrics is not None and dp_metrics is not None:
        print(f"samples: IDM={idm_metrics['num_samples']}  DP={dp_metrics['num_samples']}")
        print(
            f"chunk-mean MSE: IDM={idm_metrics['mean_mse']:.6f}  "
            f"DP={dp_metrics['mean_mse']:.6f}  "
            f"gap={gap['mean_mse']:+.6f}"
        )
    print(
        f"episode rollout: IDM eps={idm_rollout['num_episodes']} windows={idm_rollout['num_windows']} "
        f"meanMSE={idm_rollout['mean_mse']:.6f}  |  "
        f"DP eps={dp_rollout['num_episodes']} windows={dp_rollout['num_windows']} "
        f"meanMSE={dp_rollout['mean_mse']:.6f}"
    )
    print(f"wrote: {out_json}")
    for v in vis:
        print(f"wrote: {v}")
    print("=" * 70)

    # Free GPU memory before moving on to next task.
    del idm_model, dp_policy, idm_payload, dp_payload
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    configure_diffusion_policy_path(args.diffusion_policy_src)

    device = resolve_device(args.device)
    print(f"[compare] device = {device}", flush=True)

    output_root = Path(args.output_dir).expanduser()
    output_root.mkdir(parents=True, exist_ok=True)

    idm_ckpts = list(args.idm_checkpoint)
    dp_ckpts = list(args.dp_checkpoint)
    if len(idm_ckpts) != len(dp_ckpts):
        raise ValueError(
            f"--idm-checkpoint ({len(idm_ckpts)}) and --dp-checkpoint "
            f"({len(dp_ckpts)}) must have the same number of entries."
        )
    task_overrides: list[str | None]
    if args.task is None:
        task_overrides = [None] * len(idm_ckpts)
    else:
        if len(args.task) != len(idm_ckpts):
            raise ValueError(
                f"--task has {len(args.task)} entries but {len(idm_ckpts)} ckpt pairs."
            )
        task_overrides = list(args.task)

    print(f"[compare] evaluating {len(idm_ckpts)} task(s)", flush=True)

    per_task_summaries: list[dict[str, Any]] = []
    for i, (idm_c, dp_c, task_ov) in enumerate(zip(idm_ckpts, dp_ckpts, task_overrides)):
        print()
        print("#" * 70)
        print(f"# Task {i + 1}/{len(idm_ckpts)}")
        print("#" * 70)
        summary = evaluate_task(idm_c, dp_c, task_ov, args, device, output_root)
        per_task_summaries.append(summary)

    combined_path = plot_combined_rollout(
        per_task_summaries, output_root / "combined_rollout.png"
    )
    overall = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset_root": args.dataset_root,
        "num_tasks": len(per_task_summaries),
        "tasks": [s["task"] for s in per_task_summaries],
        "per_task": [
            {
                "task": s["task"],
                "idm_checkpoint": s["idm_checkpoint"],
                "dp_checkpoint": s["dp_checkpoint"],
                "chunk_mean_mse": {
                    "idm": (s["idm"] or {}).get("mean_mse"),
                    "dp": (s["dp"] or {}).get("mean_mse"),
                },
                "rollout_mean_mse": {
                    "idm": s["idm_rollout"]["mean_mse"],
                    "dp": s["dp_rollout"]["mean_mse"],
                    "num_episodes_idm": s["idm_rollout"]["num_episodes"],
                    "num_episodes_dp": s["dp_rollout"]["num_episodes"],
                },
            }
            for s in per_task_summaries
        ],
        "combined_plot": combined_path,
    }
    overall_path = output_root / "overall_summary.json"
    overall_path.write_text(json.dumps(overall, indent=2))
    print(f"\nwrote: {overall_path}")
    if combined_path:
        print(f"wrote: {combined_path}")


if __name__ == "__main__":
    main()