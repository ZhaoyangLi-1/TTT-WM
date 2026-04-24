#!/usr/bin/env python3
"""Head-to-head: model.generate(ctx, goal) vs model(ctx, tgt, goal) — one step.

Strips out all GT/noise/ground-truth comparison. Just asks: "given the same
(ctx, goal), how close do the two prediction paths agree?" Answers with a
handful of scalars on the full frame and a scatter of per-patch diffs.

Usage:
    python test_gen_vs_forward.py \
        --checkpoint /scr2/zhaoyang/latest_stage1.pt \
        --dataset    /scr2/zhaoyang/libero_wm \
        --task "KITCHEN_SCENE10: put the butter at the back in the top drawer of the cabinet and close it" \
        [--window-start 0] [--save-png]
"""
from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

_REPO = Path(__file__).resolve().parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from cosmos_model import ARPatchConfig, ARVideoPatchTransformer, patchify


def _clean_sd(sd: dict) -> dict:
    return {k.removeprefix("_orig_mod.").removeprefix("module."): v for k, v in sd.items()}


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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--window-start", type=int, default=0,
                    help="Which window of the matched episode to test.")
    ap.add_argument("--save-png", action="store_true",
                    help="Write /tmp/gen_vs_fwd_<ep>_<start>.png side-by-side image.")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = _build_cfg(ckpt)
    model = ARVideoPatchTransformer(cfg).to(device)
    sd_ema = (ckpt.get("ema") or {}).get("shadow") or ckpt.get("ema") or {}
    sd = sd_ema if sd_ema else ckpt["model"]
    model.load_state_dict(_clean_sd(sd), strict=True)
    model.eval()
    print(f"Loaded {'EMA' if sd_ema else 'live'} weights; cfg=fin{cfg.frames_in}/"
          f"fout{cfg.frames_out}/res{cfg.resolution}/patch{cfg.patch_size}")

    # Locate first matching episode.
    root = Path(args.dataset)
    with (root / "meta" / "info.json").open() as f:
        chunks_size = int(json.load(f)["chunks_size"])
    ep_idx = None
    with (root / "meta" / "episodes.jsonl").open() as f:
        for line in f:
            rec = json.loads(line)
            if (rec.get("tasks") or [None])[0] == args.task:
                ep_idx = int(rec["episode_index"])
                break
    if ep_idx is None:
        raise SystemExit(f"No episode for task {args.task!r}")

    pq = root / "data" / f"chunk-{ep_idx // chunks_size:03d}" / f"episode_{ep_idx:06d}.parquet"
    df = pd.read_parquet(pq, columns=["image"])
    tf_tr = transforms.Compose([
        transforms.Resize((cfg.resolution, cfg.resolution)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])

    def _decode(row):
        raw = row["bytes"] if isinstance(row, dict) else row
        return tf_tr(Image.open(io.BytesIO(raw)).convert("RGB"))

    frames = [_decode(r) for r in df["image"].tolist()]

    gap = int(ckpt["cfg"]["data"].get("frame_gap", 4))
    target_offset = cfg.frames_in - 1 + gap
    start = int(args.window_start)
    if start + target_offset + cfg.frames_out > len(frames):
        raise SystemExit(
            f"window-start={start} too late for episode length {len(frames)}"
        )

    ctx  = torch.stack(frames[start:start + cfg.frames_in]).unsqueeze(0).to(device)
    tgt  = torch.stack(frames[start + target_offset:start + target_offset + cfg.frames_out]).unsqueeze(0).to(device)
    goal = frames[-1].unsqueeze(0).to(device)
    print(f"Episode {ep_idx} window_start={start}  ctx={tuple(ctx.shape)}  "
          f"tgt={tuple(tgt.shape)}  goal={tuple(goal.shape)}")

    # ── Run both paths under identical autocast ─────────────────────────
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        pred_tf,  _    = model(ctx, tgt, goal)            # (1, fout, 3, H, W)
        pred_gen       = model.generate(ctx, goal=goal)   # (1, fout, 3, H, W)

    gen = pred_gen.float()
    tf_ = pred_tf.float()
    assert gen.shape == tf_.shape, f"shape mismatch: {gen.shape} vs {tf_.shape}"

    # ── Frame-level agreement ────────────────────────────────────────────
    diff = gen - tf_
    mse  = diff.pow(2).mean().item()
    mae  = diff.abs().mean().item()
    mx   = diff.abs().max().item()
    rmse = mse ** 0.5
    print("\n── generate vs forward agreement (same ctx/goal) ──")
    print(f"  shape       : {tuple(gen.shape)}  dtype={gen.dtype}")
    print(f"  mean abs    : {mae:.6f}")
    print(f"  mean sq     : {mse:.6e}")
    print(f"  RMSE        : {rmse:.6f}")
    print(f"  max |diff|  : {mx:.6f}")

    # Relative: gen-to-tf diff divided by the scale of tf itself.
    tf_rms = tf_.float().pow(2).mean().sqrt().item()
    rel = rmse / max(tf_rms, 1e-9)
    print(f"  tf signal RMS : {tf_rms:.4f}")
    print(f"  relative RMSE : {rel * 100:.3f}%  (gen disagreement / tf magnitude)")

    # ── Patch-level agreement (row-major, row 0 = top, row 15 = bottom) ─
    P = cfg.patch_size
    gen_p = patchify(gen, P)[0]   # (N, patch_dim)
    tf_p  = patchify(tf_,  P)[0]
    per_patch_mse = (gen_p - tf_p).pow(2).mean(dim=-1)  # (N,)
    n = per_patch_mse.numel()
    print(f"\n── per-patch MSE of |gen - tf| (N={n}) ─────────")
    print(f"  min  = {per_patch_mse.min().item():.6e}  at patch {int(per_patch_mse.argmin().item())}")
    print(f"  max  = {per_patch_mse.max().item():.6e}  at patch {int(per_patch_mse.argmax().item())}")
    print(f"  mean = {per_patch_mse.mean().item():.6e}")

    # Row-by-row breakdown (16 rows of 16 patches at res=128, patch=8).
    h_patches = cfg.resolution // P
    if h_patches * h_patches == n:
        rows_mse = per_patch_mse.view(h_patches, h_patches).mean(dim=1)
        print(f"  per-row mean MSE (row 0 = top of frame):")
        for r in range(h_patches):
            bar = "#" * int(rows_mse[r].item() / max(rows_mse.max().item(), 1e-9) * 40)
            print(f"    row {r:>2d}: {rows_mse[r].item():.6e}  {bar}")

    # ── Interpretation ──────────────────────────────────────────────────
    print("\n── How to read the numbers ─────────────────────────")
    if rel * 100 < 0.5:
        verdict = "identical (sub-1% RMSE) — the two paths are numerically indistinguishable."
    elif rel * 100 < 3:
        verdict = "very close — typical healthy behavior for a decent ckpt."
    elif rel * 100 < 10:
        verdict = "noticeable drift in the bottom of the frame, but the overall image matches."
    else:
        verdict = "large drift — either the model is severely undertrained, or a path bug remains."
    print(f"  Verdict: {verdict}")

    if args.save_png:
        def _to_u8(x):
            a = ((x.clamp(-1, 1) + 1) * 127.5).cpu().numpy()
            return a.transpose(1, 2, 0).astype(np.uint8)
        sep = np.full((cfg.resolution, 4, 3), 128, dtype=np.uint8)
        panel = np.concatenate(
            [_to_u8(ctx[0, -1]), sep, _to_u8(tf_[0, 0]), sep, _to_u8(gen[0, 0])],
            axis=1,
        )
        out = Path(f"/tmp/gen_vs_fwd_ep{ep_idx:06d}_start{start:04d}.png")
        Image.fromarray(panel).save(out)
        print(f"\n  Saved side-by-side PNG: {out}   (panels: ctx | TF_pred | Gen_pred)")


if __name__ == "__main__":
    main()
