#!/usr/bin/env python3
"""Diagnose why eval_heldout_tasks.py produces NaN.

Prints:
  1. How many parameters in the loaded (live + EMA) state dicts are non-finite.
  2. Whether the model produces finite output on random in-range input (sanity).
  3. On the first held-out window: where does the first NaN appear (per-layer
     activation check via forward hooks).

Usage:
    python diagnose_ckpt.py \
        --checkpoint /scr2/zhaoyang/latest_stage1.pt \
        --dataset    /scr2/zhaoyang/libero_wm \
        --task "KITCHEN_SCENE10: put the butter at the back in the top drawer of the cabinet and close it"
"""
from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms

_REPO = Path(__file__).resolve().parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from cosmos_model import ARPatchConfig, ARVideoPatchTransformer


def _non_finite(sd: dict) -> tuple[int, int, list[str]]:
    total = bad = 0
    names: list[str] = []
    for name, tensor in sd.items():
        if not isinstance(tensor, torch.Tensor) or not tensor.is_floating_point():
            continue
        n = tensor.numel()
        nb = (~torch.isfinite(tensor)).sum().item()
        total += n
        bad += nb
        if nb > 0:
            names.append(f"{name} ({nb}/{n} non-finite)")
    return bad, total, names


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--chunks-size", type=int, default=1000)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    live_sd = ckpt.get("model", {})
    ema_sd = (ckpt.get("ema") or {}).get("shadow", {}) or ckpt.get("ema", {}) or {}

    print("\n── Step 1: checkpoint weight sanity ────────────────────────")
    for name, sd in (("live", live_sd), ("EMA shadow", ema_sd)):
        if not sd:
            print(f"  [{name}] empty")
            continue
        bad, total, culprits = _non_finite(sd)
        if bad == 0:
            print(f"  [{name}] all {total:_} params finite ✓")
        else:
            print(f"  [{name}] NON-FINITE: {bad}/{total:_} elements")
            for c in culprits[:10]:
                print(f"     - {c}")

    # Build model from saved cfg.
    saved_cfg = ckpt["cfg"]
    mcfg = saved_cfg["model"]
    cfg = ARPatchConfig(
        resolution=int(mcfg["resolution"]),
        num_channels=int(mcfg["num_channels"]),
        patch_size=int(mcfg["patch_size"]),
        d_model=int(mcfg["d_model"]),
        n_heads=int(mcfg["n_heads"]),
        n_layers=int(mcfg["n_layers"]),
        mlp_ratio=float(mcfg["mlp_ratio"]),
        dropout=float(mcfg.get("dropout", 0.0)),
        frames_in=int(mcfg["frames_in"]),
        frames_out=int(mcfg["frames_out"]),
        action_dim=int(saved_cfg["data"].get("action_dim", 7)),
        qk_norm=bool(mcfg.get("qk_norm", True)),
        parallel_attn=bool(mcfg.get("parallel_attn", False)),
    )
    model = ARVideoPatchTransformer(cfg).to(device)
    sd_to_load = ema_sd if ema_sd else live_sd
    src = "EMA" if ema_sd else "live"
    model.load_state_dict({k.removeprefix("_orig_mod.").removeprefix("module."): v
                           for k, v in sd_to_load.items()}, strict=True)
    model.eval()
    print(f"  Loaded {src} weights into model.")

    # Sanity forward on random in-range noise.
    print("\n── Step 2: forward on random in-range noise ───────────────")
    B = 2
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        ctx  = (torch.rand(B, cfg.frames_in,  3, cfg.resolution, cfg.resolution, device=device) * 2 - 1)
        tgt  = (torch.rand(B, cfg.frames_out, 3, cfg.resolution, cfg.resolution, device=device) * 2 - 1)
        goal = (torch.rand(B, 3, cfg.resolution, cfg.resolution, device=device) * 2 - 1)
        pred, loss = model(ctx, tgt, goal)
    ok_pred  = torch.isfinite(pred).all().item()
    ok_loss  = torch.isfinite(loss).item()
    print(f"  pred finite: {ok_pred}   loss finite: {ok_loss}   loss={loss.item():.4f}")
    if not (ok_pred and ok_loss):
        print("  >>> model is internally unstable even on random input. "
              "This points to Hypothesis A (corrupted EMA weights).")
        return

    # Forward on an actual held-out task window, with per-layer hooks.
    print("\n── Step 3: forward on a real held-out window with hooks ───")
    root = Path(args.dataset)
    with (root / "meta" / "info.json").open() as f:
        import json
        info = json.load(f)
    chunks_size = int(info.get("chunks_size", args.chunks_size))

    # Find the first episode whose task matches.
    ep_idx = None
    with (root / "meta" / "episodes.jsonl").open() as f:
        for line in f:
            rec = json.loads(line)
            if (rec.get("tasks") or [None])[0] == args.task:
                ep_idx = int(rec["episode_index"])
                break
    if ep_idx is None:
        raise SystemExit(f"No episode found for task {args.task!r}.")
    print(f"  Using episode {ep_idx}")

    pq = root / "data" / f"chunk-{ep_idx // chunks_size:03d}" / f"episode_{ep_idx:06d}.parquet"
    df = pd.read_parquet(pq, columns=["image"])
    tf = transforms.Compose([
        transforms.Resize((cfg.resolution, cfg.resolution)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])
    def decode(row):
        raw = row["bytes"] if isinstance(row, dict) else row
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        return tf(img)
    frames = [decode(r) for r in df["image"].tolist()]

    # span for this model: fin + gap + fout - 1; gap=4 from stage1 config
    gap = int(saved_cfg["data"].get("frame_gap", 4))
    fin, fout = cfg.frames_in, cfg.frames_out
    target_offset = fin - 1 + gap

    start = 0
    ctx = torch.stack(frames[start:start + fin]).unsqueeze(0).to(device)
    tgt = torch.stack(frames[start + target_offset:start + target_offset + fout]).unsqueeze(0).to(device)
    goal = frames[-1].unsqueeze(0).to(device)

    first_bad: list[str] = []

    def hook(name):
        def _fn(_module, _inp, out):
            t = out if isinstance(out, torch.Tensor) else (out[0] if isinstance(out, (tuple, list)) else None)
            if isinstance(t, torch.Tensor) and t.is_floating_point():
                if not torch.isfinite(t).all().item() and not first_bad:
                    n_bad = (~torch.isfinite(t)).sum().item()
                    first_bad.append(f"{name}  (shape={tuple(t.shape)}, {n_bad}/{t.numel()} non-finite)")
        return _fn

    handles = [m.register_forward_hook(hook(n)) for n, m in model.named_modules() if n]
    try:
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            pred, loss = model(ctx, tgt, goal)
    finally:
        for h in handles: h.remove()

    print(f"  ctx finite: {torch.isfinite(ctx).all().item()}   "
          f"tgt finite: {torch.isfinite(tgt).all().item()}   "
          f"goal finite: {torch.isfinite(goal).all().item()}")
    print(f"  pred finite: {torch.isfinite(pred).all().item()}   loss={loss.item()}")
    if first_bad:
        print("  FIRST module to produce non-finite activation:")
        print(f"    {first_bad[0]}")
    else:
        print("  No non-finite activation at any layer on this window — "
              "model itself is fine on in-distribution-ish input.")

    # ── Step 4: model.generate — the AR rollout path ──────────────────
    print("\n── Step 4: model.generate (AR rollout path) ───────────────")
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        gen = model.generate(ctx, goal=goal)
    print(f"  gen shape: {tuple(gen.shape)}   dtype={gen.dtype}")
    total = gen.numel()
    nan_mask = torch.isnan(gen)
    inf_mask = torch.isinf(gen)
    print(f"  NaN: {nan_mask.sum().item()}/{total}   "
          f"+Inf: {(gen == float('inf')).sum().item()}   "
          f"-Inf: {(gen == float('-inf')).sum().item()}")

    # Compare first patch of generate against teacher-forcing: under token-causal
    # attention, the first target patch is computed from [goal, ctx] only and
    # should be bit-close between the two paths.
    try:
        from cosmos_model import check_generate_matches_training
        diff = check_generate_matches_training(model, ctx, goal=goal)
        print(f"  max|train_path[0] - generate[0]| = {diff:.4e}  "
              f"(bf16 tolerance ~1e-2)")
    except Exception as e:
        print(f"  check_generate_matches_training failed: {e}")

    # Second-patch-and-onward: the AR decode path. Split genpred per-patch to
    # spot whether NaN is concentrated at specific positions.
    P = cfg.patch_size
    h = w = cfg.resolution // P
    # gen shape: (B, fout, C, H, W). Patchify to (B, fout*N_p, C*P*P).
    from cosmos_model import patchify
    gen_patches = patchify(gen.float(), P)       # (B, N, patch_dim)
    per_patch_nan = torch.isnan(gen_patches).any(dim=-1).squeeze(0)  # (N,)
    n_bad_patches = per_patch_nan.sum().item()
    if n_bad_patches > 0:
        first_nan_patch = int(torch.nonzero(per_patch_nan)[0].item())
        print(f"  {n_bad_patches}/{per_patch_nan.numel()} patches contain NaN; "
              f"first NaN patch index: {first_nan_patch}")
        if first_nan_patch == 0:
            print("  >>> Patch 0 NaN: prefill/cache-population bug. "
                  "The first AR patch comes from the prefill hidden state, "
                  "NOT from flash_attn_with_kvcache, so if it's NaN the bug is "
                  "in Phase 1 of generate().")
        else:
            print(f"  >>> Patch 0 is fine, but patch {first_nan_patch}+ drift to NaN — "
                  "the AR decode path (flash_attn_with_kvcache + KV cache) is "
                  "the culprit. Likely a dtype / cache layout mismatch.")
    else:
        print("  All patches finite — generate() is working on this input.")


if __name__ == "__main__":
    main()
