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

Outputs
-------
  * <output-dir>/summary.json         — all scalar + per-dim + per-step metrics
  * <output-dir>/per_step.png         — per-step MSE curve (chunk index)
  * <output-dir>/per_dim.png          — per-dim MSE bars

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
from datetime import datetime
from pathlib import Path
from typing import Any

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
    parser.add_argument("--idm-checkpoint", type=str, required=True)
    parser.add_argument("--dp-checkpoint", type=str, required=True)
    parser.add_argument(
        "--dataset-root",
        type=str,
        default="/scr2/zhaoyang/libero_wm",
        help="Root of the LIBERO-WM dataset (must contain meta/test_tasks.json).",
    )
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help=(
            "Held-out task name. If omitted, auto-read from both ckpts' cfg.data.selected_task "
            "(must agree). If passed, must match the task both ckpts were trained on."
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

    model.eval()
    return model, cfg, source


def build_idm_val_loader(
    cfg: DictConfig,
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    val_fraction = float(
        OmegaConf.select(cfg, "data.stage2_val_fraction", default=0.02)
    )
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
) -> DataLoader:
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
# Plotting
# ---------------------------------------------------------------------------


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

    # Per-step MSE
    steps = np.arange(len(idm_metrics["per_step_mse"]))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(steps, idm_metrics["per_step_mse"], "o-", label="Pure IDM (oracle upper bound)")
    ax.plot(steps, dp_metrics["per_step_mse"], "s-", label="Diffusion Policy")
    ax.set_xlabel("action chunk index")
    ax.set_ylabel("MSE")
    ax.set_title("Per-step action MSE (lower is better)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    step_path = output_dir / "per_step.png"
    fig.savefig(step_path, dpi=150)
    plt.close(fig)
    saved.append(str(step_path))

    # Per-dim MSE
    dims = np.arange(len(idm_metrics["per_dim_mse"]))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(dims - width / 2, idm_metrics["per_dim_mse"], width, label="Pure IDM")
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    configure_diffusion_policy_path(args.diffusion_policy_src)

    device = resolve_device(args.device)
    print(f"[compare] device = {device}", flush=True)
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    idm_ckpt = resolve_checkpoint_path(args.idm_checkpoint)
    dp_ckpt = resolve_checkpoint_path(args.dp_checkpoint)

    # Read each ckpt ONCE (this is the slow part — several seconds each). The
    # embedded cfg + recorded training task are extracted here; the same
    # payload is then passed into the builders to avoid a second torch.load.
    idm_payload, idm_cfg_raw, idm_trained_task = read_idm_payload(idm_ckpt)
    dp_payload, dp_cfg_raw, dp_trained_task = read_dp_payload(dp_ckpt)

    if idm_trained_task != dp_trained_task:
        raise ValueError(
            f"Checkpoint task mismatch: Pure IDM trained on {idm_trained_task!r}, "
            f"DP trained on {dp_trained_task!r}. Pair ckpts from the same task."
        )
    if args.task is None:
        task = idm_trained_task
    else:
        if args.task != idm_trained_task:
            raise ValueError(
                f"--task={args.task!r} does not match the task both ckpts were "
                f"trained on ({idm_trained_task!r})."
            )
        task = args.task

    print(f"[compare] idm ckpt = {idm_ckpt}", flush=True)
    print(f"[compare]  dp ckpt = {dp_ckpt}", flush=True)
    print(f"[compare]    task  = {task}", flush=True)

    # --- Build models ---
    idm_model, idm_cfg, idm_source = load_pure_idm(
        idm_payload, idm_cfg_raw, device, args.use_ema, task, args.dataset_root
    )
    dp_policy, dp_cfg, dp_source = load_dp_policy(
        dp_payload, dp_cfg_raw, device, args.use_ema, task, args.dataset_root
    )

    # --- Cross-check alignment of the two val splits ---
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

    # --- Build val loaders ---
    print("[compare] building IDM val loader", flush=True)
    idm_loader = build_idm_val_loader(idm_cfg, args.batch_size, args.num_workers)
    print(f"[compare]   IDM val: {len(idm_loader.dataset)} windows / {len(idm_loader)} batches", flush=True)
    print("[compare] building DP val loader", flush=True)
    dp_loader = build_dp_val_loader(dp_cfg, args.batch_size, args.num_workers)
    print(f"[compare]   DP  val: {len(dp_loader.dataset)} windows / {len(dp_loader)} batches", flush=True)

    if len(idm_loader.dataset) != len(dp_loader.dataset):
        print(
            f"[WARN] val size mismatch — IDM windows={len(idm_loader.dataset)}, "
            f"DP windows={len(dp_loader.dataset)}. Metrics are still computed "
            "independently per model but sample-level alignment is broken."
        )

    action_dim = int(idm_cfg.data.get("action_dim", 7))
    idm_acc = ActionMetricAccumulator(frame_gap_idm, action_dim)
    dp_acc = ActionMetricAccumulator(frame_gap_dp, action_dim)
    n_obs_steps_dp = int(OmegaConf.select(dp_cfg, "n_obs_steps", default=1))

    # --- Iterate (two loaders have identical sample order when splits match) ---
    max_batches = int(args.num_batches) if args.num_batches > 0 else None
    loader_max = max(len(idm_loader), len(dp_loader))
    total_batches = loader_max if max_batches is None else min(max_batches, loader_max)

    n_batches = 0
    idm_iter = iter(idm_loader)
    dp_iter = iter(dp_loader)

    pbar = tqdm(total=total_batches, desc="[compare]", unit="batch", dynamic_ncols=True)
    while True:
        try:
            idm_batch = next(idm_iter)
        except StopIteration:
            idm_batch = None
        try:
            dp_batch = next(dp_iter)
        except StopIteration:
            dp_batch = None

        if idm_batch is None and dp_batch is None:
            break

        if idm_batch is not None:
            pred_idm, gt_idm = idm_predict(idm_model, idm_batch, device)
            idm_acc.update(pred_idm, gt_idm)

        if dp_batch is not None:
            pred_dp, gt_dp = dp_predict(dp_policy, dp_batch, device, n_obs_steps_dp)
            dp_acc.update(pred_dp, gt_dp)

        n_batches += 1
        idm_running = idm_acc._se_step_dim.sum() / max(idm_acc._count_samples, 1) / (frame_gap_idm * action_dim)
        dp_running = dp_acc._se_step_dim.sum() / max(dp_acc._count_samples, 1) / (frame_gap_dp * action_dim)
        pbar.set_postfix(idm_mse=f"{idm_running:.4f}", dp_mse=f"{dp_running:.4f}")
        pbar.update(1)

        if max_batches is not None and n_batches >= max_batches:
            break
    pbar.close()

    idm_metrics = idm_acc.compute()
    dp_metrics = dp_acc.compute()

    # --- Gap = DP − IDM (headroom for a perfect world model) ---
    gap_mse = [d - i for d, i in zip(dp_metrics["per_step_mse"], idm_metrics["per_step_mse"])]
    gap_l1 = [d - i for d, i in zip(dp_metrics["per_step_l1"], idm_metrics["per_step_l1"])]
    gap_dim_mse = [d - i for d, i in zip(dp_metrics["per_dim_mse"], idm_metrics["per_dim_mse"])]

    summary = {
        "timestamp_utc": datetime.utcnow().isoformat(timespec="seconds"),
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
        "gap_dp_minus_idm": {
            "mean_mse": dp_metrics["mean_mse"] - idm_metrics["mean_mse"],
            "mean_l1": dp_metrics["mean_l1"] - idm_metrics["mean_l1"],
            "per_step_mse": gap_mse,
            "per_step_l1": gap_l1,
            "per_dim_mse": gap_dim_mse,
        },
        "visualizations": [],
    }

    summary["visualizations"] = plot_comparison(idm_metrics, dp_metrics, output_dir)

    out_json = output_dir / "summary.json"
    out_json.write_text(json.dumps(summary, indent=2))

    print()
    print("=" * 70)
    print(f"task: {task}")
    print(f"samples: IDM={idm_metrics['num_samples']}  DP={dp_metrics['num_samples']}")
    print(
        f"mean MSE: IDM={idm_metrics['mean_mse']:.6f}  "
        f"DP={dp_metrics['mean_mse']:.6f}  "
        f"gap={summary['gap_dp_minus_idm']['mean_mse']:+.6f}"
    )
    print(
        f"mean L1:  IDM={idm_metrics['mean_l1']:.6f}  "
        f"DP={dp_metrics['mean_l1']:.6f}  "
        f"gap={summary['gap_dp_minus_idm']['mean_l1']:+.6f}"
    )
    print(f"wrote: {out_json}")
    for vis in summary["visualizations"]:
        print(f"wrote: {vis}")
    print("=" * 70)


if __name__ == "__main__":
    main()