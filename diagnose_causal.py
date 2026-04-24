"""Diagnose a token-causal AR video patch transformer (flash-attn, causal=True).

This replaces the old `diagnose_cheating.py` which was designed for
block-bidirectional masks.  Under strict token-causal attention the old
metrics are misleading: later target patches *legitimately* condition on
earlier ones via AR teacher-forcing, so loss_perm / loss_real being small
is expected, not evidence of cheating.

Correct diagnostics for token-causal
--------------------------------------

Test 1 — First-patch invariance (DECISIVE)
    Position `prefix_len - 1` (last ctx patch) predicts tgt_patch[0].
    Under causal masking this position sees ONLY [goal, ctx] — zero target
    tokens.  Therefore the prediction at this position must be IDENTICAL
    regardless of what target is fed.

    We compare the first predicted patch across:
      (a) real target  vs  noise target
      (b) real target  vs  permuted target

    max_abs_diff should be < 0.02 (bf16 tolerance).
    If it's large → causal mask is broken, target is leaking.

Test 2 — Training-vs-generate consistency
    The generate() path (KV-cache AR) and the training path must agree
    on the first predicted patch, since both compute it from exactly the
    same [goal, ctx] prefix.  Uses `check_generate_matches_training`.

Test 3 — Per-position target sensitivity (informational)
    Break down MSE by patch position within the target region.
    Position 0 (first predicted patch) should be target-invariant.
    Later positions legitimately depend on target AR context, so
    loss_perm > loss_real is normal and expected there.

Test 4 — Autoregressive rollout quality
    Run model.generate() and compare output vs real future frames.
    This is the ultimate test: if the model learned something useful,
    generate() output should be closer to reality than random.

Usage
-----
    python diagnose_causal.py \\
        --checkpoint /path/to/stage1_best.pt \\
        --dataset /scr2/zhaoyang/libero_wm \\
        --n-episodes 5 \\
        --n-windows 20
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from eval import load_model_from_checkpoint
from eval_heldout_tasks import (
    DEFAULT_TASKS,
    build_transform,
    load_dataset_metadata,
    load_episode_frames,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def iter_windows(
    frames: list[torch.Tensor],
    frames_in: int,
    frames_out: int,
    frame_gap: int,
    max_windows: int,
    rng: random.Random,
):
    """Yield (context, target) tensor pairs from an episode."""
    span = frames_in + frame_gap + frames_out - 1
    target_offset = frames_in - 1 + frame_gap
    n_windows = len(frames) - span + 1
    if n_windows <= 0:
        return
    starts = list(range(n_windows))
    rng.shuffle(starts)
    for start in starts[:max_windows]:
        ctx = torch.stack(frames[start : start + frames_in])
        tgt = torch.stack(frames[start + target_offset : start + target_offset + frames_out])
        yield ctx, tgt


def _mean(xs: list[float]) -> float:
    xs = [x for x in xs if not (isinstance(x, float) and np.isnan(x))]
    return float(np.mean(xs)) if xs else float("nan")


def _std(xs: list[float]) -> float:
    xs = [x for x in xs if not (isinstance(x, float) and np.isnan(x))]
    return float(np.std(xs)) if len(xs) > 1 else float("nan")


# ---------------------------------------------------------------------------
# Core: extract per-patch predictions from teacher-forcing
# ---------------------------------------------------------------------------

@torch.no_grad()
def forward_get_patch_preds(model, ctx, target, goal, device, use_amp):
    """Run teacher-forcing forward. Returns per-patch predictions and loss.

    Returns
    -------
    pred_patches : Tensor (B, fout*N_p, patch_dim) — raw predicted patches
    loss         : float — overall MSE
    """
    cfg = model.cfg
    N_p = cfg.n_patches
    fin = cfg.frames_in
    fout = cfg.frames_out
    has_goal = goal is not None

    with torch.amp.autocast("cuda", enabled=(use_amp and device.type == "cuda")):
        if has_goal:
            all_frames = torch.cat([goal.unsqueeze(1), ctx, target], dim=1)
            prefix_len = (1 + fin) * N_p
        else:
            all_frames = torch.cat([ctx, target], dim=1)
            prefix_len = fin * N_p

        from cosmos_model import patchify
        tokens = model._embed_frames(all_frames)
        t_idx, s_idx = model._build_position_indices(fin, fout, tokens.device, has_goal)
        hidden = model._run_transformer(tokens, t_idx, s_idx)

        t_start = prefix_len - 1
        pred_patches = model._decode(hidden[:, t_start : t_start + fout * N_p])
        tgt_patches = patchify(target, cfg.patch_size)
        loss = F.mse_loss(pred_patches, tgt_patches)

    return pred_patches.float(), float(loss.detach().float().item())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", default="/scr2/zhaoyang/libero_wm")
    parser.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS)
    parser.add_argument("--n-episodes", type=int, default=5)
    parser.add_argument("--n-windows", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--frame-gap", type=int, default=None)
    parser.add_argument("--image-key", default="image")
    parser.add_argument("--use-ema", action="store_true")
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    parser.add_argument("--skip-generate", action="store_true",
                        help="Skip AR generation test (slow)")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Load model ────────────────────────────────────────────────────
    model, model_cfg, train_cfg = load_model_from_checkpoint(
        args.checkpoint, device, use_ema=args.use_ema,
    )
    model.eval()

    data_cfg = train_cfg.get("data", {}) if isinstance(train_cfg, dict) else {}
    frame_gap = args.frame_gap if args.frame_gap is not None else int(data_cfg.get("frame_gap", 4))
    use_goal = bool(data_cfg.get("use_goal", True))
    model.prebuild_mask(device, has_goal=use_goal)

    frames_in  = int(model_cfg.frames_in)
    frames_out = int(model_cfg.frames_out)
    resolution = int(model_cfg.resolution)
    N_p        = model_cfg.n_patches
    transform  = build_transform(resolution)

    print(f"Model     : frames_in={frames_in} frames_out={frames_out} "
          f"resolution={resolution} n_patches={N_p}")
    print(f"Data      : frame_gap={frame_gap} use_goal={use_goal}")
    print(f"Tasks     : {args.tasks}")
    print()

    # ── Load dataset metadata ────────────────────────────────────────
    meta = load_dataset_metadata(args.dataset)
    root        = meta["root"]
    chunks_size = meta["chunks_size"]
    task_to_eps = meta["task_to_episodes"]

    rng = random.Random(args.seed)

    # ── Accumulators ─────────────────────────────────────────────────
    # Test 1: first-patch invariance
    first_patch_diff_noise: list[float] = []
    first_patch_diff_perm:  list[float] = []

    # Test 2: training-vs-generate consistency
    train_vs_gen_diffs: list[float] = []

    # Test 3: per-position sensitivity (indexed 0..fout*N_p-1)
    n_target_patches = frames_out * N_p
    per_pos_mse_real: list[list[float]] = [[] for _ in range(n_target_patches)]
    per_pos_mse_perm: list[list[float]] = [[] for _ in range(n_target_patches)]

    # Test 4: AR generation quality
    gen_mse_vs_real:   list[float] = []
    gen_mse_vs_random: list[float] = []

    # Also keep old-style aggregate losses for reference
    losses_real:  list[float] = []
    losses_noise: list[float] = []
    losses_perm:  list[float] = []

    n_windows_total = 0

    # ── Main loop ────────────────────────────────────────────────────
    for task in args.tasks:
        if task not in task_to_eps:
            print(f"[skip] task {task!r} not in dataset")
            continue

        ep_pool = task_to_eps[task]
        ep_rng = random.Random(args.seed + hash(task) % 10000)
        sample_eps = ep_rng.sample(ep_pool, min(args.n_episodes, len(ep_pool)))

        print(f"─── Task: {task}")
        print(f"    Sampling {len(sample_eps)} episodes from {len(ep_pool)} available")

        for ep_idx in sample_eps:
            try:
                frames = load_episode_frames(
                    root, chunks_size, ep_idx, args.image_key, transform,
                )
            except Exception as e:
                print(f"    [skip ep {ep_idx}] {e}")
                continue
            if len(frames) < frames_in + frame_gap + frames_out:
                continue

            goal = frames[-1].unsqueeze(0).to(device) if use_goal else None

            windows = list(iter_windows(
                frames, frames_in, frames_out, frame_gap,
                args.n_windows, rng,
            ))
            target_pool = [w[1] for w in windows]

            for win_idx, (ctx_cpu, tgt_cpu) in enumerate(windows):
                ctx = ctx_cpu.unsqueeze(0).to(device)
                tgt = tgt_cpu.unsqueeze(0).to(device)
                n_windows_total += 1

                # ── (1) Real target ──────────────────────────────────
                pred_real, loss_real = forward_get_patch_preds(
                    model, ctx, tgt, goal, device, args.amp,
                )
                losses_real.append(loss_real)

                # ── (2) Noise target ─────────────────────────────────
                noise = torch.rand_like(tgt) * 2 - 1
                pred_noise, loss_noise = forward_get_patch_preds(
                    model, ctx, noise, goal, device, args.amp,
                )
                losses_noise.append(loss_noise)

                # ── (3) Permuted target ──────────────────────────────
                other = [t for i, t in enumerate(target_pool) if i != win_idx]
                if other:
                    perm_tgt = rng.choice(other).unsqueeze(0).to(device)
                    pred_perm, loss_perm = forward_get_patch_preds(
                        model, ctx, perm_tgt, goal, device, args.amp,
                    )
                    losses_perm.append(loss_perm)
                else:
                    pred_perm = None
                    loss_perm = float("nan")

                # ── Test 1: first-patch invariance ───────────────────
                # pred_patches[:, 0] is the prediction at position
                # prefix_len - 1, which under causal mask sees NO target.
                diff_noise = (pred_real[:, 0] - pred_noise[:, 0]).abs().max().item()
                first_patch_diff_noise.append(diff_noise)

                if pred_perm is not None:
                    diff_perm = (pred_real[:, 0] - pred_perm[:, 0]).abs().max().item()
                    first_patch_diff_perm.append(diff_perm)

                # ── Test 3: per-position MSE breakdown ───────────────
                from cosmos_model import patchify
                tgt_patches = patchify(tgt, model.cfg.patch_size).float()
                for pos in range(n_target_patches):
                    mse_r = F.mse_loss(pred_real[:, pos], tgt_patches[:, pos]).item()
                    per_pos_mse_real[pos].append(mse_r)

                    if pred_perm is not None:
                        mse_p = F.mse_loss(pred_perm[:, pos], tgt_patches[:, pos]).item()
                        per_pos_mse_perm[pos].append(mse_p)

                # ── Test 2 & 4: generate (sampled, not every window) ─
                if not args.skip_generate and win_idx < 3:
                    # Test 2: training-vs-generate first patch
                    try:
                        from cosmos_model import check_generate_matches_training
                        diff_tg = check_generate_matches_training(model, ctx, goal)
                        train_vs_gen_diffs.append(diff_tg)
                    except Exception as e:
                        print(f"    [generate consistency check failed] {e}")

                    # Test 4: AR generation quality
                    try:
                        with torch.amp.autocast(
                            "cuda",
                            enabled=(args.amp and device.type == "cuda"),
                        ):
                            gen_frames = model.generate(ctx, goal=goal)
                        gen_mse = F.mse_loss(gen_frames.float(), tgt.float()).item()
                        gen_mse_vs_real.append(gen_mse)

                        rand_frames = torch.rand_like(tgt) * 2 - 1
                        rand_mse = F.mse_loss(rand_frames.float(), tgt.float()).item()
                        gen_mse_vs_random.append(rand_mse)
                    except Exception as e:
                        print(f"    [generate rollout failed] {e}")

        # Per-task progress
        print(f"    windows so far: {n_windows_total}")

    # ==================================================================
    # RESULTS
    # ==================================================================

    print()
    print("=" * 72)
    print("TEST 1 — First-patch invariance (DECISIVE for causal mask)")
    print("=" * 72)
    print()
    print("  The first predicted patch is at position prefix_len - 1.")
    print("  Under causal masking it sees ONLY [goal, ctx] — no target.")
    print("  So pred_patch[0] MUST be identical regardless of target input.")
    print()
    mn = _mean(first_patch_diff_noise)
    mp = _mean(first_patch_diff_perm)
    print(f"  max|pred_real[0] - pred_noise[0]| = {mn:.6f}  "
          f"(mean over {len(first_patch_diff_noise)} windows)")
    print(f"  max|pred_real[0] - pred_perm[0] | = {mp:.6f}  "
          f"(mean over {len(first_patch_diff_perm)} windows)")
    print()

    CAUSAL_TOL = 0.02  # bf16 tolerance
    if mn < CAUSAL_TOL and mp < CAUSAL_TOL:
        print("  ✅ PASS — First patch is target-invariant. Causal mask works.")
    else:
        print("  ❌ FAIL — First patch changes with target input!")
        print("           Target information is leaking through the mask.")
        print("           Check that causal=True is actually being used.")
    print()

    print("=" * 72)
    print("TEST 2 — Training path vs generate() consistency")
    print("=" * 72)
    print()
    if train_vs_gen_diffs:
        mtg = _mean(train_vs_gen_diffs)
        print(f"  max|training_pred[0] - generate_pred[0]| = {mtg:.6f}  "
              f"(mean over {len(train_vs_gen_diffs)} samples)")
        if mtg < CAUSAL_TOL:
            print("  ✅ PASS — Training and generate paths agree on first patch.")
        else:
            print("  ❌ FAIL — Training and generate paths disagree.")
            print("           KV-cache or RoPE mismatch in generate().")
    else:
        print("  (skipped — use without --skip-generate to enable)")
    print()

    print("=" * 72)
    print("TEST 3 — Per-position target sensitivity (informational)")
    print("=" * 72)
    print()
    print("  Position 0 should be target-invariant (same real vs perm).")
    print("  Later positions legitimately depend on AR context.")
    print()
    print(f"  {'pos':>5}  {'frame':>5}  {'patch':>5}  {'mse_real':>10}  "
          f"{'mse_perm':>10}  {'ratio':>8}  {'note'}")
    print(f"  {'─'*5}  {'─'*5}  {'─'*5}  {'─'*10}  {'─'*10}  {'─'*8}  {'─'*20}")

    # Show first few, boundary, and last few positions
    show_positions = set()
    # first 3
    show_positions.update(range(min(3, n_target_patches)))
    # boundary: first patch of each frame
    for f in range(frames_out):
        show_positions.add(f * N_p)
    # last 2
    show_positions.update(range(max(0, n_target_patches - 2), n_target_patches))

    for pos in sorted(show_positions):
        mr = _mean(per_pos_mse_real[pos])
        mperm = _mean(per_pos_mse_perm[pos])
        ratio = mperm / mr if mr > 0 else float("nan")
        frame_i = pos // N_p
        patch_i = pos % N_p
        note = ""
        if pos == 0:
            note = "← must be ≈1.0 (no target visible)"
        elif patch_i == 0:
            note = "← frame boundary"
        print(f"  {pos:>5}  {frame_i:>5}  {patch_i:>5}  {mr:>10.6f}  "
              f"{mperm:>10.6f}  {ratio:>8.2f}  {note}")

    if n_target_patches > len(show_positions):
        print(f"  ... ({n_target_patches - len(show_positions)} positions omitted)")
    print()

    print("=" * 72)
    print("TEST 4 — Autoregressive rollout quality")
    print("=" * 72)
    print()
    if gen_mse_vs_real:
        mg = _mean(gen_mse_vs_real)
        mr = _mean(gen_mse_vs_random)
        ratio = mg / mr if mr > 0 else float("nan")
        print(f"  MSE(generate, real_future)   = {mg:.6f}  "
              f"(mean over {len(gen_mse_vs_real)} samples)")
        print(f"  MSE(random[-1,1], real)      = {mr:.6f}  (baseline)")
        print(f"  ratio generate/random        = {ratio:.4f}")
        print()
        if ratio < 0.5:
            print("  ✅ Generate output is substantially better than random.")
        elif ratio < 0.9:
            print("  ⚠️  Generate output is only slightly better than random.")
            print("     Model may need more training or have capacity issues.")
        else:
            print("  ❌ Generate output is no better than random.")
            print("     Something is wrong with generation, even if training loss")
            print("     looks good.")
    else:
        print("  (skipped — use without --skip-generate to enable)")
    print()

    # ── Reference: old-style aggregate losses ────────────────────────
    print("=" * 72)
    print("REFERENCE — Aggregate losses (for comparison with old diagnostic)")
    print("=" * 72)
    print()
    print(f"  loss_real  = {_mean(losses_real):.6f}")
    print(f"  loss_noise = {_mean(losses_noise):.6f}   "
          f"(high → model can't echo noise → good)")
    print(f"  loss_perm  = {_mean(losses_perm):.6f}   "
          f"(≈ loss_real is NORMAL for token-causal AR)")
    print()
    lr = _mean(losses_real)
    ln = _mean(losses_noise)
    if lr > 0:
        print(f"  loss_noise / loss_real = {ln/lr:.1f}x")
        print(f"  loss_perm  / loss_real = {_mean(losses_perm)/lr:.1f}x")
    print()

    # ── Final verdict ────────────────────────────────────────────────
    print("=" * 72)
    print("FINAL VERDICT")
    print("=" * 72)
    print()

    issues = []

    if mn >= CAUSAL_TOL or mp >= CAUSAL_TOL:
        issues.append("MASK_LEAK: first patch depends on target input")

    if train_vs_gen_diffs and _mean(train_vs_gen_diffs) >= CAUSAL_TOL:
        issues.append("PATH_MISMATCH: training ≠ generate on first patch")

    if gen_mse_vs_real and _mean(gen_mse_vs_real) / max(_mean(gen_mse_vs_random), 1e-8) > 0.9:
        issues.append("BAD_ROLLOUT: generate output no better than random")

    if not issues:
        print("  ✅ Model is HEALTHY under token-causal attention.")
        print()
        print("  • Causal mask verified: first patch is target-invariant.")
        if train_vs_gen_diffs:
            print("  • Training and generate paths are consistent.")
        if gen_mse_vs_real:
            ratio = _mean(gen_mse_vs_real) / max(_mean(gen_mse_vs_random), 1e-8)
            print(f"  • Rollout quality: {ratio:.2f}x random baseline.")
        print()
        print("  The loss_perm ≈ 2× loss_real you see is EXPECTED:")
        print("  in token-causal AR, later patches legitimately condition")
        print("  on earlier target patches. Swapping the target changes")
        print("  the AR context, so loss goes up — this is correct behavior.")
    else:
        print("  ❌ Issues detected:")
        for iss in issues:
            print(f"     • {iss}")
        print()
        if "MASK_LEAK" in str(issues):
            print("  The causal mask is not working correctly.")
            print("  Verify flash_attn_func is called with causal=True.")
        if "BAD_ROLLOUT" in str(issues):
            print("  The model's autoregressive generation produces garbage.")
            print("  If the mask is healthy, check:")
            print("    - KV cache population (k_norm / RoPE order)")
            print("    - Position index computation in generate()")
            print("    - Enough training (underfitting?)")


if __name__ == "__main__":
    main()