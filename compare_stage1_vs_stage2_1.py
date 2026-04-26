#!/usr/bin/env python3
"""Compare Stage 1 vs Stage 2.1 effect on a single held-out task.

Takes the two ckpts + a task string + a dataset root, picks an episode of
that task, and produces:

  <output-dir>/
    examples/example_00.png    — ctx | GT_target | S1_pred | S2.1_pred | goal
    examples/example_01.png    (N=4 by default; windows evenly spaced)
    ...
    quantitative/
      tf_mse_curve.png         — teacher-forcing MSE per window (S1 vs S2.1)
      rollout_mse_curve.png    — AR rollout MSE per step (S1 vs S2.1)
      per_row_drift.png        — per-patch-row gen-vs-forward MSE (S1 vs S2.1)
      summary.png              — 3-panel figure combining the above
    metrics.json               — scalar summary of both models

Usage:
    python compare_stage1_vs_stage2_1.py \\
        --stage1-ckpt /scr2/zhaoyang/latest_stage1.pt \\
        --stage2-1-ckpt /path/to/stage2.1_best.pt \\
        --dataset /scr2/zhaoyang/libero_wm \\
        --task "KITCHEN_SCENE10: put the butter at the back in the top drawer of the cabinet and close it" \\
        --output-dir /scr2/zhaoyang/compare_s1_s21/KITCHEN_SCENE10 \\
        [--n-examples 4] [--n-rollout-steps 8] [--n-tf-windows 64]
"""
from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torchvision import transforms

_REPO = Path(__file__).resolve().parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from cosmos_model import ARPatchConfig, ARVideoPatchTransformer, patchify  # noqa: E402
from eval import save_video  # noqa: E402


# ---------------------------------------------------------------------------
# Checkpoint loading (mirrors train_stage2's helper)
# ---------------------------------------------------------------------------


def _clean_sd(sd: dict) -> dict:
    return {
        k.removeprefix("_orig_mod.").removeprefix("module."): v
        for k, v in sd.items()
    }


def _extract_state_dict(ckpt: dict) -> tuple[dict, str]:
    ema = (ckpt.get("ema") or {}).get("shadow") or ckpt.get("ema") or {}
    if ema:
        return _clean_sd(ema), "EMA"
    return _clean_sd(ckpt["model"]), "live"


def _build_cfg(ckpt: dict) -> ARPatchConfig:
    saved = ckpt["cfg"]
    m = saved["model"]
    return ARPatchConfig(
        resolution=int(m["resolution"]),
        num_channels=int(m["num_channels"]),
        patch_size=int(m["patch_size"]),
        d_model=int(m["d_model"]),
        n_heads=int(m["n_heads"]),
        n_layers=int(m["n_layers"]),
        mlp_ratio=float(m["mlp_ratio"]),
        dropout=float(m.get("dropout", 0.0)),
        frames_in=int(m["frames_in"]),
        frames_out=int(m["frames_out"]),
        action_dim=int(saved["data"].get("action_dim", 7)),
        qk_norm=bool(m.get("qk_norm", True)),
        parallel_attn=bool(m.get("parallel_attn", False)),
    )


def _load_model(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = _build_cfg(ckpt)
    model = ARVideoPatchTransformer(cfg).to(device)
    sd, src = _extract_state_dict(ckpt)
    model.load_state_dict(sd, strict=True)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  loaded {src} weights ({n_params:.2f}M params) from {ckpt_path}")
    return model, cfg, ckpt


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _find_episode_path(dataset_root: Path, task: str) -> tuple[int, Path, int]:
    with (dataset_root / "meta" / "info.json").open() as f:
        info = json.load(f)
    chunks_size = int(info["chunks_size"])
    with (dataset_root / "meta" / "episodes.jsonl").open() as f:
        for line in f:
            rec = json.loads(line)
            if (rec.get("tasks") or [None])[0] == task:
                ep = int(rec["episode_index"])
                pq = (
                    dataset_root / "data"
                    / f"chunk-{ep // chunks_size:03d}"
                    / f"episode_{ep:06d}.parquet"
                )
                return ep, pq, int(rec["length"])
    raise SystemExit(f"No episode for task {task!r}")


def _reconstruct_stage2_split(
    dataset_root: Path,
    task: str,
    val_fraction: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    """Bit-identical reproduction of HeldoutTaskSplitDataset's episode split.

    Mirrors the 5 lines inside train_stage2.HeldoutTaskSplitDataset.__init__
    (see train_stage2.py:231-250):

        all_eps = sorted(episodes_where_task == selected_task)
        rng     = np.random.default_rng(seed)
        shuffled = list(all_eps);  rng.shuffle(shuffled)
        n_val   = max(1, round(len * val_fraction))  clamped to len-1
        val_eps = shuffled[:n_val];  train_eps = shuffled[n_val:]

    Returns (val_eps, train_eps) as sorted ascending episode-index lists,
    identical to what Stage 2.1 training saw. Runs without loading any
    parquet file — just reads meta/episodes.jsonl.
    """
    eps_for_task: list[int] = []
    with (dataset_root / "meta" / "episodes.jsonl").open() as f:
        for line in f:
            rec = json.loads(line)
            if (rec.get("tasks") or [None])[0] == task:
                eps_for_task.append(int(rec["episode_index"]))
    if not eps_for_task:
        return [], []

    all_eps = sorted(eps_for_task)
    rng = np.random.default_rng(int(seed))
    shuffled = list(all_eps)
    rng.shuffle(shuffled)

    if len(shuffled) > 1:
        n_val = max(1, int(round(len(shuffled) * float(val_fraction))))
        n_val = min(n_val, len(shuffled) - 1)
    else:
        n_val = 0

    val_eps = sorted(shuffled[:n_val])
    train_eps = sorted(shuffled[n_val:])
    return val_eps, train_eps


def _read_stage2_split_cfg(ckpt: dict) -> tuple[Optional[str], float, int]:
    """Pull (selected_task, stage2_val_fraction, seed) out of a Stage 2.1 ckpt
    so the val/train split can be reproduced bit-identically without needing
    the user to remember the exact training-time hyperparameters.
    """
    saved = ckpt.get("cfg", {})
    data = saved.get("data", {})
    task = data.get("selected_task") or None
    vfrac = float(data.get("stage2_val_fraction", 0.01))
    # Seed can appear at top-level (cfg.seed) or inside training.
    seed_val = saved.get("seed")
    if seed_val is None:
        seed_val = saved.get("training", {}).get("seed", 42)
    return task, vfrac, int(seed_val)


def _resolve_parquet_path(dataset_root: Path, episode_idx: int) -> Path:
    with (dataset_root / "meta" / "info.json").open() as f:
        chunks_size = int(json.load(f)["chunks_size"])
    return (
        dataset_root / "data"
        / f"chunk-{episode_idx // chunks_size:03d}"
        / f"episode_{episode_idx:06d}.parquet"
    )


def _load_frames(parquet_path: Path, resolution: int) -> list[torch.Tensor]:
    df = pd.read_parquet(parquet_path, columns=["image"])
    tr = transforms.Compose([
        transforms.Resize((resolution, resolution)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])
    def _decode(row):
        raw = row["bytes"] if isinstance(row, dict) else row
        return tr(Image.open(io.BytesIO(raw)).convert("RGB"))
    return [_decode(r) for r in df["image"].tolist()]


# ---------------------------------------------------------------------------
# Model evaluation primitives (work identically on S1 and S2.1)
# ---------------------------------------------------------------------------


@torch.no_grad()
def _run_teacher_forcing_sweep(
    model: ARVideoPatchTransformer,
    frames: list[torch.Tensor],
    cfg: ARPatchConfig,
    frame_gap: int,
    device: torch.device,
    n_windows: int,
) -> dict:
    """Evaluate teacher-forcing MSE on up to n_windows evenly-spaced windows."""
    fin, fout = cfg.frames_in, cfg.frames_out
    span = fin + frame_gap + fout - 1
    target_offset = fin - 1 + frame_gap
    total_windows = max(0, len(frames) - span + 1)
    if total_windows == 0:
        return {"n_windows": 0, "window_starts": [], "mse_per_window": []}
    idxs = (
        [0] if n_windows >= total_windows
        else np.linspace(0, total_windows - 1, num=n_windows, dtype=int).tolist()
    )
    if n_windows >= total_windows:
        idxs = list(range(total_windows))
    goal = frames[-1].unsqueeze(0).to(device)
    mses: list[float] = []
    for s in idxs:
        ctx = torch.stack(frames[s:s + fin]).unsqueeze(0).to(device)
        tgt = torch.stack(
            frames[s + target_offset:s + target_offset + fout]
        ).unsqueeze(0).to(device)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            pred, _ = model(ctx, tgt, goal)
        m = F.mse_loss(pred.float(), tgt.float()).item()
        mses.append(float(m))
    return {
        "n_windows": len(idxs),
        "window_starts": [int(x) for x in idxs],
        "mse_per_window": mses,
    }


@torch.no_grad()
def _run_rollout(
    model: ARVideoPatchTransformer,
    frames: list[torch.Tensor],
    cfg: ARPatchConfig,
    frame_gap: int,
    device: torch.device,
    n_steps: int,
    *,
    start_frame: int = 0,
) -> dict:
    """Open-loop AR rollout from `frames[start_frame]`, comparing against GT
    every `frame_gap` frames. Returns per-step MSE plus uint8 GT/pred lists
    ready for video composition.

    start_frame must satisfy 0 <= start_frame <= len(frames)-1-frame_gap
    (at least one future GT frame available for step 0). If the remaining
    tail is too short, n_steps is silently truncated.
    """
    if cfg.frames_in != 1 or cfg.frames_out != 1:
        raise NotImplementedError("Rollout path only supports fin=fout=1.")
    start_frame = int(start_frame)
    if start_frame < 0 or start_frame >= len(frames) - 1:
        raise SystemExit(
            f"--rollout-start-frame={start_frame} out of range for episode of "
            f"length {len(frames)} (valid: 0..{len(frames) - 2})."
        )
    available = (len(frames) - 1 - start_frame) // frame_gap
    n_steps = min(int(n_steps), available)
    if n_steps <= 0:
        return {
            "n_steps": 0, "step_mse": [],
            "gt_frames": [], "pred_frames": [],
            "start_frame": start_frame,
        }
    goal = frames[-1].unsqueeze(0).to(device)
    window = frames[start_frame].unsqueeze(0).unsqueeze(0).to(device)
    step_mse: list[float] = []
    gt_u8  = [_to_uint8(frames[start_frame].unsqueeze(0))[0]]
    pred_u8 = [_to_uint8(frames[start_frame].unsqueeze(0))[0]]
    for step in range(n_steps):
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            pred = model.generate(window, goal=goal)                 # (1,1,3,H,W)
        gt_frame = frames[
            start_frame + (step + 1) * frame_gap
        ].unsqueeze(0).unsqueeze(0).to(device)
        m = F.mse_loss(pred.float(), gt_frame.float()).item()
        step_mse.append(float(m))
        pred_u8.append(_to_uint8(pred[0])[0])
        gt_u8.append(_to_uint8(gt_frame[0])[0])
        window = pred
    return {
        "n_steps": len(step_mse),
        "step_mse": step_mse,
        "gt_frames": gt_u8,
        "pred_frames": pred_u8,
        "start_frame": start_frame,
    }


@torch.no_grad()
def _per_row_gen_vs_forward_drift(
    model: ARVideoPatchTransformer,
    frames: list[torch.Tensor],
    cfg: ARPatchConfig,
    frame_gap: int,
    device: torch.device,
    window_start: int,
) -> np.ndarray:
    """Return (H/P,) array of MSE(gen - forward) per patch row."""
    fin, fout, P = cfg.frames_in, cfg.frames_out, cfg.patch_size
    target_offset = fin - 1 + frame_gap
    ctx = torch.stack(frames[window_start:window_start + fin]).unsqueeze(0).to(device)
    tgt = torch.stack(
        frames[window_start + target_offset:window_start + target_offset + fout]
    ).unsqueeze(0).to(device)
    goal = frames[-1].unsqueeze(0).to(device)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        pred_tf, _ = model(ctx, tgt, goal)
        pred_gen   = model.generate(ctx, goal=goal)
    gen_p = patchify(pred_gen.float(), P)[0]      # (N, patch_dim)
    tf_p  = patchify(pred_tf.float(), P)[0]
    per_patch = (gen_p - tf_p).pow(2).mean(dim=-1)
    h = cfg.resolution // P
    if h * h != per_patch.numel():
        # Should never happen for square frames; be safe.
        return per_patch.cpu().numpy()
    return per_patch.view(h, h).mean(dim=1).cpu().numpy()


@torch.no_grad()
def _generate_one_step(
    model: ARVideoPatchTransformer,
    ctx: torch.Tensor,
    goal: torch.Tensor,
) -> torch.Tensor:
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        return model.generate(ctx, goal=goal).float()


# ---------------------------------------------------------------------------
# Imaging helpers
# ---------------------------------------------------------------------------


def _to_uint8(frames: torch.Tensor) -> np.ndarray:
    """(T,3,H,W) in [-1,1] → (T,H,W,3) uint8. NaN/Inf safe.

    LIBERO's agentview parquet stores frames upside-down; the model trains
    on them directly (no flip) and visualization code in train_stage1.py
    (`_frames_to_uint8`) rot90-twice before showing to a human. Mirror that
    convention here so saved PNGs are right-side-up.
    """
    x = frames.float()
    x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
    x = (x.clamp(-1, 1) * 0.5 + 0.5) * 255.0
    arr = x.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
    arr = np.rot90(arr, 2, axes=(1, 2)).copy()      # 180° rotation per frame
    return arr


def _hstack_with_captions(
    panels: list[np.ndarray],
    captions: list[str],
    scale: int = 2,
) -> Image.Image:
    """Concatenate panels horizontally with a caption strip above each."""
    assert len(panels) == len(captions)
    panels = [
        np.asarray(Image.fromarray(p).resize(
            (p.shape[1] * scale, p.shape[0] * scale), resample=Image.NEAREST,
        ))
        for p in panels
    ]
    h, w = panels[0].shape[:2]
    sep = np.full((h, 4, 3), 180, dtype=np.uint8)
    pieces: list[np.ndarray] = []
    for i, p in enumerate(panels):
        pieces.append(p)
        if i < len(panels) - 1:
            pieces.append(sep)
    strip = np.concatenate(pieces, axis=1)
    # caption strip
    caption_h = 18
    canvas = Image.new("RGB", (strip.shape[1], strip.shape[0] + caption_h),
                       color=(255, 255, 255))
    canvas.paste(Image.fromarray(strip), (0, caption_h))
    draw = ImageDraw.Draw(canvas)
    # Captions are positioned above the center of each panel.
    x = 0
    for i, cap in enumerate(captions):
        draw.text((x + 4, 2), cap, fill=(0, 0, 0))
        x += w
        if i < len(captions) - 1:
            x += 4
    return canvas


# ---------------------------------------------------------------------------
# Quantitative plots
# ---------------------------------------------------------------------------


def _plot_tf_curve(tf_s1: dict, tf_s2: dict, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 3.6), dpi=120)
    if tf_s1["n_windows"]:
        ax.plot(tf_s1["window_starts"], tf_s1["mse_per_window"],
                label=f"Stage 1  mean={np.mean(tf_s1['mse_per_window']):.4f}",
                color="#1f77b4", marker="o", markersize=3, linewidth=1.1)
    if tf_s2["n_windows"]:
        ax.plot(tf_s2["window_starts"], tf_s2["mse_per_window"],
                label=f"Stage 2.1  mean={np.mean(tf_s2['mse_per_window']):.4f}",
                color="#2ca02c", marker="o", markersize=3, linewidth=1.1)
    ax.set_xlabel("Window start frame")
    ax.set_ylabel("Teacher-forcing MSE")
    ax.set_title("Teacher-forcing MSE per window (lower is better)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _plot_rollout_curve(roll_s1: dict, roll_s2: dict, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 3.6), dpi=120)
    if roll_s1["n_steps"]:
        ys = np.asarray(roll_s1["step_mse"])
        ax.plot(range(len(ys)), ys,
                label=f"Stage 1  mean={ys.mean():.4f}",
                color="#1f77b4", marker="o", markersize=3, linewidth=1.2)
    if roll_s2["n_steps"]:
        ys = np.asarray(roll_s2["step_mse"])
        ax.plot(range(len(ys)), ys,
                label=f"Stage 2.1  mean={ys.mean():.4f}",
                color="#2ca02c", marker="o", markersize=3, linewidth=1.2)
    ax.set_xlabel("Rollout step")
    ax.set_ylabel("MSE (pred vs GT)")
    ax.set_title("AR rollout MSE per step (lower is better)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _plot_per_row_drift(
    drift_s1: np.ndarray, drift_s2: np.ndarray, out_path: Path,
) -> None:
    n_rows = len(drift_s1)
    xs = np.arange(n_rows)
    fig, ax = plt.subplots(figsize=(8.0, 3.8), dpi=120)
    w = 0.4
    ax.bar(xs - w / 2, drift_s1, width=w, label="Stage 1", color="#1f77b4")
    ax.bar(xs + w / 2, drift_s2, width=w, label="Stage 2.1", color="#2ca02c")
    ax.set_yscale("log")
    ax.set_xlabel("Patch row (0 = top of frame, 15 = bottom)")
    ax.set_ylabel("mean(gen - forward)^2  [log scale]")
    ax.set_title("Per-patch-row generate vs forward drift (lower is better)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _plot_summary(
    tf_s1: dict, tf_s2: dict,
    roll_s1: dict, roll_s2: dict,
    drift_s1: np.ndarray, drift_s2: np.ndarray,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.0), dpi=120)
    # TF panel
    ax = axes[0]
    if tf_s1["n_windows"]:
        ax.plot(tf_s1["window_starts"], tf_s1["mse_per_window"],
                color="#1f77b4", label="Stage 1", marker="o", markersize=3)
    if tf_s2["n_windows"]:
        ax.plot(tf_s2["window_starts"], tf_s2["mse_per_window"],
                color="#2ca02c", label="Stage 2.1", marker="o", markersize=3)
    ax.set_title("Teacher-forcing MSE / window")
    ax.set_xlabel("Window start")
    ax.set_ylabel("MSE")
    ax.grid(True, alpha=0.3)
    ax.legend()
    # Rollout panel
    ax = axes[1]
    if roll_s1["n_steps"]:
        ax.plot(roll_s1["step_mse"], color="#1f77b4",
                label="Stage 1", marker="o", markersize=3)
    if roll_s2["n_steps"]:
        ax.plot(roll_s2["step_mse"], color="#2ca02c",
                label="Stage 2.1", marker="o", markersize=3)
    ax.set_title("Rollout MSE / step")
    ax.set_xlabel("Step")
    ax.set_ylabel("MSE")
    ax.grid(True, alpha=0.3)
    ax.legend()
    # Drift panel
    ax = axes[2]
    xs = np.arange(len(drift_s1))
    w = 0.4
    ax.bar(xs - w / 2, drift_s1, width=w, color="#1f77b4", label="Stage 1")
    ax.bar(xs + w / 2, drift_s2, width=w, color="#2ca02c", label="Stage 2.1")
    ax.set_yscale("log")
    ax.set_title("Per-row gen vs fwd drift")
    ax.set_xlabel("Patch row")
    ax.set_ylabel("MSE (log)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Overlay: compare val-episode vs train-episode metrics in one figure
# ---------------------------------------------------------------------------


def _make_split_overlay(val_dir: Path, train_dir: Path, out_dir: Path) -> None:
    """Load per-split metrics.json and produce a 2-panel overlay figure +
    a small combined_metrics.json summarizing the train/val generalization gap.
    """
    def _load(d: Path) -> dict:
        return json.loads((d / "metrics.json").read_text())
    m_val = _load(val_dir)
    m_tr  = _load(train_dir)

    tf_val, tf_tr = m_val["teacher_forcing"], m_tr["teacher_forcing"]
    ro_val, ro_tr = m_val["rollout"], m_tr["rollout"]

    # ── Plot: 2 panels — TF per window; Rollout per step ──────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2), dpi=120)

    # Teacher-forcing overlay
    ax = axes[0]
    if tf_val["n_windows"]:
        ax.plot(tf_val["window_starts"], tf_val["stage1_mse_per_window"],
                color="#1f77b4", linestyle="-",  linewidth=1.2,
                label=f"S1 · val   (ep {m_val['episode_index']})")
        ax.plot(tf_val["window_starts"], tf_val["stage2_1_mse_per_window"],
                color="#2ca02c", linestyle="-",  linewidth=1.2,
                label=f"S2.1 · val")
    if tf_tr["n_windows"]:
        ax.plot(tf_tr["window_starts"], tf_tr["stage1_mse_per_window"],
                color="#1f77b4", linestyle="--", linewidth=1.2,
                label=f"S1 · train (ep {m_tr['episode_index']})")
        ax.plot(tf_tr["window_starts"], tf_tr["stage2_1_mse_per_window"],
                color="#2ca02c", linestyle="--", linewidth=1.2,
                label="S2.1 · train")
    ax.set_xlabel("Window start frame")
    ax.set_ylabel("Teacher-forcing MSE")
    ax.set_title("Teacher-forcing MSE — val (solid) vs train (dashed)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)

    # Rollout overlay
    ax = axes[1]
    if ro_val["n_steps"]:
        xs = np.arange(ro_val["n_steps"])
        ax.plot(xs, ro_val["stage1_step_mse"],
                color="#1f77b4", linestyle="-",  marker="o", markersize=3, linewidth=1.2,
                label="S1 · val")
        ax.plot(xs, ro_val["stage2_1_step_mse"],
                color="#2ca02c", linestyle="-",  marker="o", markersize=3, linewidth=1.2,
                label="S2.1 · val")
    if ro_tr["n_steps"]:
        xs = np.arange(ro_tr["n_steps"])
        ax.plot(xs, ro_tr["stage1_step_mse"],
                color="#1f77b4", linestyle="--", marker="s", markersize=3, linewidth=1.2,
                label="S1 · train")
        ax.plot(xs, ro_tr["stage2_1_step_mse"],
                color="#2ca02c", linestyle="--", marker="s", markersize=3, linewidth=1.2,
                label="S2.1 · train")
    ax.set_xlabel("Rollout step")
    ax.set_ylabel("MSE")
    ax.set_title("Rollout MSE — val (solid/circle) vs train (dashed/square)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_dir / "overlay.png", bbox_inches="tight")
    plt.close(fig)

    # ── combined_metrics.json — scalar summary of the generalization gap ─
    def _pct_delta(old: float, new: float) -> float | None:
        if old is None or new is None:
            return None
        old = float(old); new = float(new)
        if old <= 0:
            return None
        return (old - new) / old * 100.0

    tf_val_s1 = tf_val.get("stage1_mean_mse")
    tf_val_s2 = tf_val.get("stage2_1_mean_mse")
    tf_tr_s1  = tf_tr.get("stage1_mean_mse")
    tf_tr_s2  = tf_tr.get("stage2_1_mean_mse")
    ro_val_s1 = ro_val.get("stage1_mean_mse")
    ro_val_s2 = ro_val.get("stage2_1_mean_mse")
    ro_tr_s1  = ro_tr.get("stage1_mean_mse")
    ro_tr_s2  = ro_tr.get("stage2_1_mean_mse")

    combined = {
        "task": m_val["task"],
        "val_episode":   m_val["episode_index"],
        "train_episode": m_tr["episode_index"],
        "teacher_forcing_mean_mse": {
            "val":   {"stage1": tf_val_s1, "stage2_1": tf_val_s2,
                      "s2_1_improvement_pct": _pct_delta(tf_val_s1, tf_val_s2)},
            "train": {"stage1": tf_tr_s1,  "stage2_1": tf_tr_s2,
                      "s2_1_improvement_pct": _pct_delta(tf_tr_s1,  tf_tr_s2)},
        },
        "rollout_mean_mse": {
            "val":   {"stage1": ro_val_s1, "stage2_1": ro_val_s2,
                      "s2_1_improvement_pct": _pct_delta(ro_val_s1, ro_val_s2)},
            "train": {"stage1": ro_tr_s1,  "stage2_1": ro_tr_s2,
                      "s2_1_improvement_pct": _pct_delta(ro_tr_s1,  ro_tr_s2)},
        },
        "generalization_gap_pct": {
            # Positive number = S2.1 improves train more than val = overfit-ish.
            "teacher_forcing": (
                None if None in (tf_val_s1, tf_val_s2, tf_tr_s1, tf_tr_s2)
                else _pct_delta(tf_tr_s1, tf_tr_s2) - _pct_delta(tf_val_s1, tf_val_s2)
            ),
            "rollout": (
                None if None in (ro_val_s1, ro_val_s2, ro_tr_s1, ro_tr_s2)
                else _pct_delta(ro_tr_s1, ro_tr_s2) - _pct_delta(ro_val_s1, ro_val_s2)
            ),
        },
    }
    (out_dir / "combined_metrics.json").write_text(json.dumps(combined, indent=2))

    # ── Pretty-print summary ───────────────────────────────────────────
    print("\n── val vs train summary ────────────────────────────────────")
    def _fmt(x):
        return f"{x:.5f}" if isinstance(x, (int, float)) else "n/a"
    def _fmtpct(x):
        return f"{x:+.1f}%" if isinstance(x, (int, float)) else "n/a"
    print(f"  Teacher-forcing mean MSE:")
    print(f"    val   — S1={_fmt(tf_val_s1)}  S2.1={_fmt(tf_val_s2)}  "
          f"(Δ={_fmtpct(_pct_delta(tf_val_s1, tf_val_s2))})")
    print(f"    train — S1={_fmt(tf_tr_s1)}  S2.1={_fmt(tf_tr_s2)}  "
          f"(Δ={_fmtpct(_pct_delta(tf_tr_s1, tf_tr_s2))})")
    print(f"  Rollout mean MSE:")
    print(f"    val   — S1={_fmt(ro_val_s1)}  S2.1={_fmt(ro_val_s2)}  "
          f"(Δ={_fmtpct(_pct_delta(ro_val_s1, ro_val_s2))})")
    print(f"    train — S1={_fmt(ro_tr_s1)}  S2.1={_fmt(ro_tr_s2)}  "
          f"(Δ={_fmtpct(_pct_delta(ro_tr_s1, ro_tr_s2))})")
    gap_tf = combined["generalization_gap_pct"]["teacher_forcing"]
    gap_ro = combined["generalization_gap_pct"]["rollout"]
    print(f"\n  Generalization gap (train improvement − val improvement):")
    print(f"    TF      : {_fmtpct(gap_tf)}  (positive = overfit tendency)")
    print(f"    Rollout : {_fmtpct(gap_ro)}")


# ---------------------------------------------------------------------------
# Rollout video (GT | Stage 1 | Stage 2.1 side-by-side)
# ---------------------------------------------------------------------------


def _save_rollout_comparison_video(
    gt_frames: list[np.ndarray],
    s1_frames: list[np.ndarray],
    s2_frames: list[np.ndarray],
    output_path: Path,
    fps: int = 6,
    image_scale: int = 2,
) -> None:
    """Save a 3-column side-by-side mp4: GT | Stage 1 pred | Stage 2.1 pred.

    Each row uses _to_uint8's convention (rot180-ed; display orientation).
    A thin caption strip at the top labels each panel so the video is
    self-describing when shared.
    """
    n = min(len(gt_frames), len(s1_frames), len(s2_frames))
    if n == 0:
        return

    # Upscale each frame (nearest-neighbor) for a more readable video.
    def _scale(x: np.ndarray) -> np.ndarray:
        if int(image_scale) <= 1:
            return x
        from PIL import Image as _Img
        pil = _Img.fromarray(x).resize(
            (x.shape[1] * int(image_scale), x.shape[0] * int(image_scale)),
            resample=_Img.NEAREST,
        )
        return np.asarray(pil)

    gt_s  = [_scale(x) for x in gt_frames[:n]]
    s1_s  = [_scale(x) for x in s1_frames[:n]]
    s2_s  = [_scale(x) for x in s2_frames[:n]]
    H, W = gt_s[0].shape[:2]
    sep_w = max(4, 2 * int(image_scale))
    sep_v = np.full((H, sep_w, 3), 180, dtype=np.uint8)

    # Build a constant caption strip once; blend into every frame.
    caption_h = max(18, 10 * int(image_scale))
    from PIL import Image as _Img, ImageDraw as _Draw
    strip = _Img.new("RGB",
                     (W * 3 + 2 * sep_w, caption_h),
                     color=(255, 255, 255))
    draw = _Draw.Draw(strip)
    draw.text((8,           2), "GT",         fill=(0, 0, 0))
    draw.text((W + sep_w + 8, 2), "Stage 1",  fill=(0x1f, 0x77, 0xb4))
    draw.text((2 * (W + sep_w) + 8, 2), "Stage 2.1", fill=(0x2c, 0xa0, 0x2c))
    caption_np = np.asarray(strip)

    video_frames = []
    for t in range(n):
        row = np.concatenate([gt_s[t], sep_v, s1_s[t], sep_v, s2_s[t]], axis=1)
        full = np.concatenate([caption_np, row], axis=0)
        video_frames.append(full)

    save_video(np.stack(video_frames, axis=0), str(output_path), fps=int(fps))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage1-ckpt", required=True)
    ap.add_argument("--stage2-1-ckpt", required=True)
    ap.add_argument("--dataset", default="/scr2/zhaoyang/libero_wm")
    ap.add_argument("--task", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--n-examples", type=int, default=10,
                    help="Number of qualitative example windows to save.")
    ap.add_argument("--n-rollout-steps", type=int, default=50)
    ap.add_argument("--n-tf-windows", type=int, default=64,
                    help="Windows sampled for teacher-forcing MSE sweep.")
    ap.add_argument("--episode-index", type=int, default=-1,
                    help="Explicit episode index (overrides --split).")
    ap.add_argument(
        "--split", choices=("val", "train", "both", "first-match"), default="val",
        help=(
            "Which episode to evaluate on. 'val' (default) reproduces Stage "
            "2.1 training's val/train split from the ckpt cfg and uses the "
            "first val episode. 'train' uses the first train episode. "
            "'both' runs the full pipeline TWICE — once on val, once on train "
            "— saving to <output-dir>/val and <output-dir>/train, plus a "
            "<output-dir>/val_vs_train/overlay.png combining all 4 curves. "
            "'first-match' = original behavior (first matching in episodes.jsonl)."
        ),
    )
    ap.add_argument(
        "--val-fraction", type=float, default=-1.0,
        help=(
            "Override stage2_val_fraction for split reconstruction "
            "(default: read from Stage 2.1 ckpt cfg)."
        ),
    )
    ap.add_argument(
        "--split-seed", type=int, default=-1,
        help=(
            "Override seed for split reconstruction "
            "(default: read from Stage 2.1 ckpt cfg)."
        ),
    )
    ap.add_argument(
        "--verify-split", action="store_true",
        help=(
            "Print the reconstructed Stage 2.1 val/train episode lists and "
            "exit without running any evaluation. Use this to sanity-check "
            "that the split matches what training actually saw."
        ),
    )
    ap.add_argument("--image-scale", type=int, default=2,
                    help="Nearest-neighbor upscale for saved PNGs.")
    ap.add_argument("--rollout-fps", type=int, default=6,
                    help="fps for the rollout comparison mp4.")
    ap.add_argument(
        "--rollout-start-frame", type=int, default=0,
        help=(
            "Frame index within the episode at which the AR rollout begins. "
            "Default 0 (start of episode). Both Stage 1 and Stage 2.1 use the "
            "same start for a fair comparison. Valid: 0..len(frames)-2. "
            "The video's first frame will be frames[start]."
        ),
    )
    ap.add_argument("--skip-rollout-video", action="store_true",
                    help="Do not save the 3-column rollout comparison video.")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "examples").mkdir(exist_ok=True)
    (out_dir / "quantitative").mkdir(exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("\n── Loading Stage 1 model ───────────────────────────────────")
    model_s1, cfg_s1, ckpt_s1 = _load_model(args.stage1_ckpt, device)
    print("\n── Loading Stage 2.1 model ─────────────────────────────────")
    model_s2, cfg_s2, ckpt_s2 = _load_model(args.stage2_1_ckpt, device)

    # Both ckpts should share the same patch/frame config for a fair comparison.
    for attr in ("resolution", "patch_size", "frames_in", "frames_out"):
        if getattr(cfg_s1, attr) != getattr(cfg_s2, attr):
            raise SystemExit(
                f"cfg mismatch on '{attr}': S1={getattr(cfg_s1, attr)} vs "
                f"S2.1={getattr(cfg_s2, attr)}"
            )
    cfg = cfg_s1
    frame_gap = int(ckpt_s1["cfg"]["data"].get("frame_gap", 4))
    print(f"\nShared cfg: res={cfg.resolution} patch={cfg.patch_size} "
          f"fin={cfg.frames_in} fout={cfg.frames_out} gap={frame_gap}")

    # ── Reconstruct Stage 2.1's train/val split from the ckpt cfg ──────
    # This matches train_stage2.HeldoutTaskSplitDataset bit-identically.
    dataset_root = Path(args.dataset)
    ckpt_task, ckpt_vfrac, ckpt_seed = _read_stage2_split_cfg(ckpt_s2)
    vfrac = args.val_fraction if args.val_fraction >= 0 else ckpt_vfrac
    seed = args.split_seed if args.split_seed >= 0 else ckpt_seed
    if ckpt_task and ckpt_task != args.task:
        print(
            f"\n[warn] --task='{args.task}' differs from Stage 2.1 ckpt's "
            f"data.selected_task='{ckpt_task}'. Using --task for the split "
            f"reconstruction; the ckpt's backbone was trained on the other. "
            f"If this is intentional, ignore; otherwise pass the matching --task."
        )
    val_eps, train_eps = _reconstruct_stage2_split(
        dataset_root, args.task, val_fraction=vfrac, seed=seed,
    )
    print("\n── Reconstructed Stage 2.1 split ───────────────────────────")
    print(f"  task          : {args.task}")
    print(f"  val_fraction  : {vfrac}")
    print(f"  seed          : {seed}")
    print(f"  total episodes: {len(val_eps) + len(train_eps)}")
    print(f"  val   ({len(val_eps)}): {val_eps}")
    print(f"  train ({len(train_eps)}): "
          f"{train_eps[:6]}{' …' if len(train_eps) > 6 else ''}")

    if args.verify_split:
        print("\n--verify-split was set; exiting without running evaluation.")
        return

    # ── --split both: run twice (val + train) and overlay the results ──
    if args.split == "both":
        if not val_eps:
            raise SystemExit("Cannot run --split both: reconstructed val set is empty.")
        if not train_eps:
            raise SystemExit("Cannot run --split both: reconstructed train set is empty.")
        import subprocess
        sub_dirs: dict[str, Path] = {}
        for sub_split in ("val", "train"):
            sub_out = out_dir / sub_split
            # Re-exec this script with --split <sub_split> and a sub output-dir.
            new_args = [sys.executable, sys.argv[0]]
            skip_next = False
            for tok in sys.argv[1:]:
                if skip_next:
                    skip_next = False
                    continue
                if tok in ("--split", "--output-dir"):
                    skip_next = True
                    continue
                if tok.startswith("--split=") or tok.startswith("--output-dir="):
                    continue
                new_args.append(tok)
            new_args += ["--split", sub_split, "--output-dir", str(sub_out)]
            print(f"\n=== Sub-run: --split {sub_split} → {sub_out} ===")
            subprocess.run(new_args, check=True)
            sub_dirs[sub_split] = sub_out

        # Aggregate: load both metrics.json, make an overlay figure.
        overlay_dir = out_dir / "val_vs_train"
        overlay_dir.mkdir(parents=True, exist_ok=True)
        _make_split_overlay(sub_dirs["val"], sub_dirs["train"], overlay_dir)

        print(f"\nAll outputs under: {out_dir.resolve()}")
        print(f"  val  : {sub_dirs['val'].resolve()}")
        print(f"  train: {sub_dirs['train'].resolve()}")
        print(f"  overlay: {overlay_dir.resolve()}")
        return

    # Choose which episode to evaluate on.
    if args.episode_index >= 0:
        ep_idx = int(args.episode_index)
        source_label = "manual"
    elif args.split == "val":
        if not val_eps:
            raise SystemExit(
                "Stage 2.1 split has no val episodes (val_fraction × n_eps < 1). "
                "Use --split train or --episode-index ..."
            )
        ep_idx = val_eps[0]
        source_label = "stage2.1-val"
    elif args.split == "train":
        if not train_eps:
            raise SystemExit("Reconstructed split has no train episodes.")
        ep_idx = train_eps[0]
        source_label = "stage2.1-train"
    else:  # "first-match"
        ep_idx, _, _ = _find_episode_path(dataset_root, args.task)
        source_label = "first-match"

    pq = _resolve_parquet_path(dataset_root, ep_idx)

    # Tag the chosen episode with its role in the split (for logging + JSON).
    if ep_idx in val_eps:
        episode_role = "val"
    elif ep_idx in train_eps:
        episode_role = "train"
    else:
        episode_role = "unknown"  # e.g. --episode-index pointing elsewhere

    print(f"\nTask        : {args.task}")
    print(f"Chosen ep   : {ep_idx}  ({source_label}, role={episode_role})")
    print(f"Parquet     : {pq.name}")
    frames = _load_frames(pq, cfg.resolution)
    print(f"Loaded      : {len(frames)} frames at {cfg.resolution}x{cfg.resolution}")

    # ── 1. Qualitative examples ────────────────────────────────────────
    print("\n── Saving qualitative examples ─────────────────────────────")
    fin, fout = cfg.frames_in, cfg.frames_out
    span = fin + frame_gap + fout - 1
    target_offset = fin - 1 + frame_gap
    total_windows = max(0, len(frames) - span + 1)
    n_ex = min(int(args.n_examples), total_windows)
    if n_ex <= 0:
        raise SystemExit("Episode too short for any valid window.")
    example_starts = (
        [0] if n_ex == 1
        else np.linspace(0, total_windows - 1, num=n_ex, dtype=int).tolist()
    )
    goal = frames[-1].unsqueeze(0).to(device)

    example_rows: list[dict] = []
    for i, s in enumerate(example_starts):
        ctx_t = torch.stack(frames[s:s + fin]).unsqueeze(0).to(device)
        tgt_t = torch.stack(
            frames[s + target_offset:s + target_offset + fout]
        ).unsqueeze(0).to(device)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            pred_s1 = model_s1.generate(ctx_t, goal=goal).float()
            pred_s2 = model_s2.generate(ctx_t, goal=goal).float()
        mse_s1 = F.mse_loss(pred_s1, tgt_t.float()).item()
        mse_s2 = F.mse_loss(pred_s2, tgt_t.float()).item()

        ctx_u8  = _to_uint8(ctx_t[0, -1:])[0]
        tgt_u8  = _to_uint8(tgt_t[0, :1])[0]
        s1_u8   = _to_uint8(pred_s1[0, :1])[0]
        s2_u8   = _to_uint8(pred_s2[0, :1])[0]
        goal_u8 = _to_uint8(goal[0:1])[0]

        img = _hstack_with_captions(
            [ctx_u8, tgt_u8, s1_u8, s2_u8, goal_u8],
            [
                f"ctx (frame {s + fin - 1})",
                f"GT target",
                f"Stage 1   mse={mse_s1:.4f}",
                f"Stage 2.1 mse={mse_s2:.4f}",
                "goal",
            ],
            scale=int(args.image_scale),
        )
        path = out_dir / "examples" / f"example_{i:02d}_start_{int(s):04d}.png"
        img.save(path)
        rel = mse_s1 / max(mse_s2, 1e-9)
        print(f"  [{i+1}/{n_ex}] window_start={int(s):>4d}  "
              f"S1 MSE={mse_s1:.4f}  S2.1 MSE={mse_s2:.4f}  "
              f"S1/S2.1 ratio={rel:.2f}x  → {path.name}")
        example_rows.append({
            "slot": i,
            "window_start": int(s),
            "mse_stage1": mse_s1,
            "mse_stage2_1": mse_s2,
            "ratio_s1_over_s2_1": rel,
            "path": str(path.resolve()),
        })

    # ── 2. Teacher-forcing sweep ───────────────────────────────────────
    print("\n── Teacher-forcing sweep (S1 vs S2.1) ──────────────────────")
    tf_s1 = _run_teacher_forcing_sweep(
        model_s1, frames, cfg, frame_gap, device, int(args.n_tf_windows),
    )
    tf_s2 = _run_teacher_forcing_sweep(
        model_s2, frames, cfg, frame_gap, device, int(args.n_tf_windows),
    )
    print(f"  S1   : n={tf_s1['n_windows']:>3d}  "
          f"mean MSE = {np.mean(tf_s1['mse_per_window']):.5f}")
    print(f"  S2.1 : n={tf_s2['n_windows']:>3d}  "
          f"mean MSE = {np.mean(tf_s2['mse_per_window']):.5f}")
    _plot_tf_curve(tf_s1, tf_s2, out_dir / "quantitative" / "tf_mse_curve.png")

    # ── 3. Open-loop rollout ───────────────────────────────────────────
    rollout_start = int(args.rollout_start_frame)
    print(f"\n── AR rollout (S1 vs S2.1, start_frame={rollout_start}) ────")
    roll_s1 = _run_rollout(
        model_s1, frames, cfg, frame_gap, device, int(args.n_rollout_steps),
        start_frame=rollout_start,
    )
    roll_s2 = _run_rollout(
        model_s2, frames, cfg, frame_gap, device, int(args.n_rollout_steps),
        start_frame=rollout_start,
    )
    print(f"  S1   : n_steps={roll_s1['n_steps']:>3d}  "
          f"mean={np.mean(roll_s1['step_mse']) if roll_s1['step_mse'] else float('nan'):.5f}  "
          f"last={roll_s1['step_mse'][-1] if roll_s1['step_mse'] else float('nan'):.5f}")
    print(f"  S2.1 : n_steps={roll_s2['n_steps']:>3d}  "
          f"mean={np.mean(roll_s2['step_mse']) if roll_s2['step_mse'] else float('nan'):.5f}  "
          f"last={roll_s2['step_mse'][-1] if roll_s2['step_mse'] else float('nan'):.5f}")
    _plot_rollout_curve(roll_s1, roll_s2, out_dir / "quantitative" / "rollout_mse_curve.png")

    # ── 3b. Rollout side-by-side video (GT | Stage 1 | Stage 2.1) ──────
    rollout_video_path: Optional[Path] = None
    if (
        not args.skip_rollout_video
        and roll_s1["n_steps"] > 0
        and roll_s2["n_steps"] > 0
    ):
        # Encode start_frame in the filename so multiple runs with different
        # start frames don't clobber each other in the same output-dir.
        vid_name = (
            "rollout_comparison.mp4"
            if rollout_start == 0
            else f"rollout_comparison_start{rollout_start:04d}.mp4"
        )
        rollout_video_path = out_dir / "quantitative" / vid_name
        # Both rollouts share the same GT frames (same episode, same gap,
        # same start_frame).
        _save_rollout_comparison_video(
            gt_frames=roll_s1["gt_frames"],
            s1_frames=roll_s1["pred_frames"],
            s2_frames=roll_s2["pred_frames"],
            output_path=rollout_video_path,
            fps=int(args.rollout_fps),
            image_scale=int(args.image_scale),
        )

    # ── 4. Per-row generate-vs-forward drift ───────────────────────────
    print("\n── Per-row gen-vs-forward drift (first window) ─────────────")
    drift_s1 = _per_row_gen_vs_forward_drift(
        model_s1, frames, cfg, frame_gap, device, window_start=int(example_starts[0]),
    )
    drift_s2 = _per_row_gen_vs_forward_drift(
        model_s2, frames, cfg, frame_gap, device, window_start=int(example_starts[0]),
    )
    print(f"  S1   drift per row: "
          f"min={drift_s1.min():.2e}  max={drift_s1.max():.2e}  "
          f"mean={drift_s1.mean():.2e}")
    print(f"  S2.1 drift per row: "
          f"min={drift_s2.min():.2e}  max={drift_s2.max():.2e}  "
          f"mean={drift_s2.mean():.2e}")
    _plot_per_row_drift(drift_s1, drift_s2, out_dir / "quantitative" / "per_row_drift.png")

    # ── 5. Combined summary figure ─────────────────────────────────────
    _plot_summary(
        tf_s1, tf_s2, roll_s1, roll_s2, drift_s1, drift_s2,
        out_dir / "quantitative" / "summary.png",
    )

    # ── 6. metrics.json ────────────────────────────────────────────────
    metrics = {
        "task": args.task,
        "episode_index": int(ep_idx),
        "episode_length": int(len(frames)),
        "frame_gap": int(frame_gap),
        "stage1_ckpt": str(Path(args.stage1_ckpt).resolve()),
        "stage2_1_ckpt": str(Path(args.stage2_1_ckpt).resolve()),
        "stage2_1_split": {
            "source": source_label,
            "role_of_chosen_episode": episode_role,
            "val_fraction": float(vfrac),
            "seed": int(seed),
            "ckpt_selected_task": ckpt_task,
            "val_episodes": val_eps,
            "train_episodes": train_eps,
        },
        "examples": example_rows,
        "teacher_forcing": {
            "n_windows": int(tf_s1["n_windows"]),
            "window_starts": tf_s1["window_starts"],
            "stage1_mse_per_window": tf_s1["mse_per_window"],
            "stage2_1_mse_per_window": tf_s2["mse_per_window"],
            "stage1_mean_mse": float(np.mean(tf_s1["mse_per_window"])) if tf_s1["n_windows"] else None,
            "stage2_1_mean_mse": float(np.mean(tf_s2["mse_per_window"])) if tf_s2["n_windows"] else None,
        },
        "rollout": {
            "start_frame": int(rollout_start),
            "n_steps": int(roll_s1["n_steps"]),
            "stage1_step_mse": roll_s1["step_mse"],
            "stage2_1_step_mse": roll_s2["step_mse"],
            "stage1_mean_mse": float(np.mean(roll_s1["step_mse"])) if roll_s1["step_mse"] else None,
            "stage2_1_mean_mse": float(np.mean(roll_s2["step_mse"])) if roll_s2["step_mse"] else None,
        },
        "per_row_drift": {
            "stage1": drift_s1.tolist(),
            "stage2_1": drift_s2.tolist(),
        },
        "output_paths": {
            "examples_dir": str((out_dir / "examples").resolve()),
            "quantitative_dir": str((out_dir / "quantitative").resolve()),
            "summary_png": str((out_dir / "quantitative" / "summary.png").resolve()),
            "rollout_video": str(rollout_video_path.resolve()) if rollout_video_path else None,
        },
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    # ── Concise stdout verdict ─────────────────────────────────────────
    print("\n── Summary ─────────────────────────────────────────────────")
    if tf_s1["n_windows"]:
        s1_tf = float(np.mean(tf_s1["mse_per_window"]))
        s2_tf = float(np.mean(tf_s2["mse_per_window"]))
        imp = (s1_tf - s2_tf) / max(s1_tf, 1e-9) * 100.0
        print(f"  Teacher-forcing mean MSE:  S1={s1_tf:.5f}  "
              f"S2.1={s2_tf:.5f}  (Δ={imp:+.1f}% relative to S1)")
    if roll_s1["n_steps"]:
        s1_r = float(np.mean(roll_s1["step_mse"]))
        s2_r = float(np.mean(roll_s2["step_mse"]))
        imp = (s1_r - s2_r) / max(s1_r, 1e-9) * 100.0
        print(f"  Rollout mean MSE:          S1={s1_r:.5f}  "
              f"S2.1={s2_r:.5f}  (Δ={imp:+.1f}% relative to S1)")
    print(f"\nAll outputs under: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
