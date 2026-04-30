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
        [--episode-index 42] \\
        [--n-examples 4] [--n-rollout-steps 8] [--n-tf-windows 64]

If --episode-index is omitted, the first episode matching --task in
meta/episodes.jsonl is used.
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
def _run_single_step_sweep(
    model: ARVideoPatchTransformer,
    frames: list[torch.Tensor],
    cfg: ARPatchConfig,
    frame_gap: int,
    device: torch.device,
    *,
    stride: int = 1,
) -> dict:
    """For each input frame i (stride-spaced), independently generate one
    step ahead: ctx = GT frames[i] -> pred ~ frames[i + frame_gap].

    Unlike `_run_rollout`, every step starts from a clean GT context (no
    autoregression). Sweeps i = 0, stride, 2*stride, ..., as long as
    i + frame_gap <= len(frames) - 1, so every prediction has a GT target.
    """
    if cfg.frames_in != 1 or cfg.frames_out != 1:
        raise NotImplementedError("Single-step sweep only supports fin=fout=1.")
    last = len(frames) - 1 - frame_gap
    if last < 0:
        return {
            "n_steps": 0, "step_mse": [],
            "gt_frames": [], "pred_frames": [],
            "input_indices": [], "target_indices": [],
        }
    indices = list(range(0, last + 1, max(int(stride), 1)))
    goal = frames[-1].unsqueeze(0).to(device)
    gt_u8: list[np.ndarray] = []
    pred_u8: list[np.ndarray] = []
    step_mse: list[float] = []
    for i in indices:
        ctx = frames[i].unsqueeze(0).unsqueeze(0).to(device)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            pred = model.generate(ctx, goal=goal)
        gt = frames[i + frame_gap].unsqueeze(0).unsqueeze(0).to(device)
        step_mse.append(float(F.mse_loss(pred.float(), gt.float()).item()))
        gt_u8.append(_to_uint8(gt[0])[0])
        pred_u8.append(_to_uint8(pred[0])[0])
    return {
        "n_steps": len(indices),
        "step_mse": step_mse,
        "gt_frames": gt_u8,
        "pred_frames": pred_u8,
        "input_indices": indices,
        "target_indices": [i + frame_gap for i in indices],
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
# visualization_flow GIF (Input | GT | Stage 1 | Stage 2.1, animated)
# ---------------------------------------------------------------------------


def _save_visualization_flow_gif(
    gt_frames: list[np.ndarray],
    s1_frames: list[np.ndarray],
    s2_frames: list[np.ndarray],
    output_path: Path,
    fps: int = 4,
    image_scale: int = 2,
    step_labels: Optional[list[str]] = None,
) -> None:
    """Save an animated GIF mirroring the rollout mp4 contents:
    3 panels per frame — GT | Stage 1 pred | Stage 2.1 pred.

    Input frames are the same uint8 sequences used by
    `_save_rollout_comparison_video`, so the GIF and mp4 are time-aligned
    representations of the same AR rollout.
    """
    n = min(len(gt_frames), len(s1_frames), len(s2_frames))
    if n == 0:
        return

    def _scale(x: np.ndarray) -> np.ndarray:
        if int(image_scale) <= 1:
            return x
        pil = Image.fromarray(x).resize(
            (x.shape[1] * int(image_scale), x.shape[0] * int(image_scale)),
            resample=Image.NEAREST,
        )
        return np.asarray(pil)

    gt_s = [_scale(x) for x in gt_frames[:n]]
    s1_s = [_scale(x) for x in s1_frames[:n]]
    s2_s = [_scale(x) for x in s2_frames[:n]]
    H, W = gt_s[0].shape[:2]
    sep_w = max(4, 2 * int(image_scale))
    sep_v = np.full((H, sep_w, 3), 180, dtype=np.uint8)

    caption_h = max(18, 10 * int(image_scale))
    full_w = W * 3 + 2 * sep_w
    title_strip = Image.new("RGB", (full_w, caption_h), color=(255, 255, 255))
    tdraw = ImageDraw.Draw(title_strip)
    tdraw.text((8,                      2), "GT",        fill=(0, 0, 0))
    tdraw.text((W + sep_w + 8,          2), "Stage 1",   fill=(0x1f, 0x77, 0xb4))
    tdraw.text((2 * (W + sep_w) + 8,    2), "Stage 2.1", fill=(0x2c, 0xa0, 0x2c))
    title_np = np.asarray(title_strip)

    pil_frames: list[Image.Image] = []
    for t in range(n):
        row = np.concatenate([gt_s[t], sep_v, s1_s[t], sep_v, s2_s[t]], axis=1)
        full = np.concatenate([title_np, row], axis=0)
        img = Image.fromarray(full)
        if step_labels is not None and t < len(step_labels):
            ImageDraw.Draw(img).text(
                (4, caption_h + 4), step_labels[t],
                fill=(255, 255, 255),
            )
        pil_frames.append(img)

    duration_ms = max(1, int(1000 / max(int(fps), 1)))
    pil_frames[0].save(
        output_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
    )


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
    ap.add_argument(
        "--episode-index", type=int, default=-1,
        help=(
            "Episode index to evaluate on. If omitted (-1), the first episode "
            "matching --task in meta/episodes.jsonl is used."
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
    ap.add_argument("--gif-fps", type=int, default=4,
                    help="fps for visualization_flow.gif (rollout-mirroring).")
    ap.add_argument("--skip-flow-gif", action="store_true",
                    help="Do not save visualization_flow.gif.")
    ap.add_argument("--singlestep-stride", type=int, default=1,
                    help="Stride for single-step sweep (1 = every frame: "
                         "i -> i+gap, i+1 -> i+1+gap, ...).")
    ap.add_argument("--singlestep-fps", type=int, default=6,
                    help="fps for the single-step sweep mp4.")
    ap.add_argument("--skip-singlestep-video", action="store_true",
                    help="Do not save singlestep_comparison.mp4.")
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
    model_s2, cfg_s2, _ = _load_model(args.stage2_1_ckpt, device)

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

    # ── Pick the episode to evaluate on ────────────────────────────────
    dataset_root = Path(args.dataset)
    if args.episode_index >= 0:
        ep_idx = int(args.episode_index)
        source_label = "manual"
    else:
        ep_idx, _, _ = _find_episode_path(dataset_root, args.task)
        source_label = "first-match"

    pq = _resolve_parquet_path(dataset_root, ep_idx)
    print(f"\nTask        : {args.task}")
    print(f"Chosen ep   : {ep_idx}  ({source_label})")
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

    # ── 3c. Single-step sweep (each frame independently: i -> i+gap) ───
    singlestep_video_path: Optional[Path] = None
    ss_s1: dict = {"n_steps": 0, "step_mse": [], "input_indices": [], "target_indices": []}
    ss_s2: dict = {"n_steps": 0, "step_mse": [], "input_indices": [], "target_indices": []}
    if not args.skip_singlestep_video:
        print("\n── Single-step sweep (S1 vs S2.1, ctx = GT each step) ──")
        ss_s1 = _run_single_step_sweep(
            model_s1, frames, cfg, frame_gap, device,
            stride=int(args.singlestep_stride),
        )
        ss_s2 = _run_single_step_sweep(
            model_s2, frames, cfg, frame_gap, device,
            stride=int(args.singlestep_stride),
        )
        if ss_s1["n_steps"] > 0 and ss_s2["n_steps"] > 0:
            print(f"  S1   : n={ss_s1['n_steps']:>3d}  "
                  f"mean MSE = {np.mean(ss_s1['step_mse']):.5f}")
            print(f"  S2.1 : n={ss_s2['n_steps']:>3d}  "
                  f"mean MSE = {np.mean(ss_s2['step_mse']):.5f}")
            singlestep_video_path = (
                out_dir / "quantitative" / "singlestep_comparison.mp4"
            )
            _save_rollout_comparison_video(
                gt_frames=ss_s1["gt_frames"],
                s1_frames=ss_s1["pred_frames"],
                s2_frames=ss_s2["pred_frames"],
                output_path=singlestep_video_path,
                fps=int(args.singlestep_fps),
                image_scale=int(args.image_scale),
            )
            print(f"  saved -> {singlestep_video_path}")
        else:
            print("  episode too short for any (i, i+gap) pair; skipping.")

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

    # ── 5b. visualization_flow GIF (rollout: GT | S1 | S2.1, animated) ─
    # Mirrors the rollout mp4 contents exactly (same start frame, same
    # n_rollout_steps, same GT/pred frames). Only the container format
    # differs — handy when GIF is preferred over mp4 (slack/web embeds).
    flow_gif_path: Optional[Path] = None
    if (
        not args.skip_flow_gif
        and roll_s1["n_steps"] > 0
        and roll_s2["n_steps"] > 0
    ):
        n_gif = len(roll_s1["gt_frames"])
        # Per-frame caption: step 0 is the starting context; step k>0 lands at
        # raw frame rollout_start + k*frame_gap.
        step_labels = [
            f"start raw_frame={rollout_start}" if k == 0
            else f"step {k}  raw_frame={rollout_start + k * frame_gap}"
            for k in range(n_gif)
        ]
        gif_name = (
            "visualization_flow.gif"
            if rollout_start == 0
            else f"visualization_flow_start{rollout_start:04d}.gif"
        )
        flow_gif_path = out_dir / "quantitative" / gif_name
        print(
            f"\n── Building visualization_flow GIF "
            f"(rollout, n={n_gif}, start={rollout_start}) ──"
        )
        _save_visualization_flow_gif(
            gt_frames=roll_s1["gt_frames"],
            s1_frames=roll_s1["pred_frames"],
            s2_frames=roll_s2["pred_frames"],
            output_path=flow_gif_path,
            fps=int(args.gif_fps),
            image_scale=int(args.image_scale),
            step_labels=step_labels,
        )
        print(f"  saved → {flow_gif_path}")

    # ── 6. metrics.json ────────────────────────────────────────────────
    metrics = {
        "task": args.task,
        "episode_index": int(ep_idx),
        "episode_length": int(len(frames)),
        "frame_gap": int(frame_gap),
        "stage1_ckpt": str(Path(args.stage1_ckpt).resolve()),
        "stage2_1_ckpt": str(Path(args.stage2_1_ckpt).resolve()),
        "episode_source": source_label,
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
        "single_step": {
            "stride": int(args.singlestep_stride),
            "n_steps": int(ss_s1["n_steps"]),
            "input_indices": ss_s1["input_indices"],
            "target_indices": ss_s1["target_indices"],
            "stage1_step_mse": ss_s1["step_mse"],
            "stage2_1_step_mse": ss_s2["step_mse"],
            "stage1_mean_mse": float(np.mean(ss_s1["step_mse"])) if ss_s1["step_mse"] else None,
            "stage2_1_mean_mse": float(np.mean(ss_s2["step_mse"])) if ss_s2["step_mse"] else None,
        },
        "output_paths": {
            "examples_dir": str((out_dir / "examples").resolve()),
            "quantitative_dir": str((out_dir / "quantitative").resolve()),
            "summary_png": str((out_dir / "quantitative" / "summary.png").resolve()),
            "rollout_video": str(rollout_video_path.resolve()) if rollout_video_path else None,
            "visualization_flow_gif": str(flow_gif_path.resolve()) if flow_gif_path else None,
            "singlestep_video": str(singlestep_video_path.resolve()) if singlestep_video_path else None,
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
