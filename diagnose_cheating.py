"""Diagnose whether a trained Stage 1 checkpoint has learned to "cheat" by
echoing target-frame embeddings through the block-bidirectional mask, instead
of actually predicting the future frame from [goal, ctx].

Rationale
---------
`cosmos_model.make_sequence_mask` is block-causal with bidirectional visibility
INSIDE each target frame block. The training loss is shift-by-1 next-patch
prediction. This combination gives the model a trivial attention path:
"copy the input embedding of tgt[j+1] into the hidden state at position
prefix_len+j, then identity-project". If the model takes this path, it will
get near-zero teacher-forcing MSE during training but rollout will fail
because there's no target to copy from.

Test
----
For each (ctx, real_target, goal) tuple:

  1. loss_real   = MSE(model(ctx, real_target, goal), real_target)
                   — baseline teacher-forcing loss.
  2. loss_noise  = MSE(model(ctx, noise_target, goal), noise_target)
                   — if model echoes the input, this also goes near 0.
  3. cross_mse   = MSE(model(ctx, noise_target, goal), real_target)
                   — the decisive number:
                     * model is genuinely predicting    →  cross_mse ≈ loss_real
                     * model is echoing whatever target →  cross_mse ≫ loss_real

The script also runs a "permuted target" variant (swap target with a
different episode's target) as a secondary check.

Usage
-----
    python diagnose_cheating.py \\
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
import torch.nn.functional as F

from eval import load_model_from_checkpoint
from eval_heldout_tasks import (
    DEFAULT_TASKS,
    build_transform,
    load_dataset_metadata,
    load_episode_frames,
)


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


@torch.no_grad()
def forward_with_target(model, ctx, target, goal, device, use_amp):
    """Run teacher-forcing forward. Returns (pred_frames, loss_in_paper).

    loss_in_paper is F.mse_loss(pred_patches, target_patches) — internal to
    the model's forward. We also get pred_frames so we can compute cross
    losses externally.
    """
    with torch.amp.autocast("cuda", enabled=(use_amp and device.type == "cuda")):
        pred_frames, loss = model(ctx, target, goal)
    return pred_frames.float(), float(loss.detach().float().item())


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, help="Path to Stage 1 checkpoint (.pt)")
    parser.add_argument("--dataset", default="/scr2/zhaoyang/libero_wm", help="Dataset root (default: /scr2/zhaoyang/libero_wm)")
    parser.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS, help="Task names to sample from")
    parser.add_argument("--n-episodes", type=int, default=5, help="Episodes to sample per task")
    parser.add_argument("--n-windows", type=int, default=20, help="Random windows per episode")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--frame-gap", type=int, default=None, help="Override frame gap (default: use checkpoint's)")
    parser.add_argument("--image-key", default="image")
    parser.add_argument("--use-ema", action="store_true", help="Use EMA weights if available")
    parser.add_argument("--amp", action="store_true", default=True, help="Use bf16 autocast")
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Load model ────────────────────────────────────────────────────────
    model, model_cfg, train_cfg = load_model_from_checkpoint(
        args.checkpoint, device, use_ema=args.use_ema
    )
    model.eval()

    data_cfg = train_cfg.get("data", {}) if isinstance(train_cfg, dict) else {}
    frame_gap = args.frame_gap if args.frame_gap is not None else int(data_cfg.get("frame_gap", 4))
    use_goal = bool(data_cfg.get("use_goal", True))
    model.prebuild_mask(device, has_goal=use_goal)

    frames_in  = int(model_cfg.frames_in)
    frames_out = int(model_cfg.frames_out)
    resolution = int(model_cfg.resolution)
    transform  = build_transform(resolution)

    print(f"Model     : frames_in={frames_in} frames_out={frames_out} resolution={resolution}")
    print(f"Data      : frame_gap={frame_gap} use_goal={use_goal}")
    print(f"Tasks     : {args.tasks}")
    print()

    # ── Load dataset metadata ────────────────────────────────────────────
    meta = load_dataset_metadata(args.dataset)
    root        = meta["root"]
    chunks_size = meta["chunks_size"]
    task_to_eps = meta["task_to_episodes"]

    rng = random.Random(args.seed)

    # Stats accumulators (per-task and overall)
    accum: dict[str, dict[str, list[float]]] = {}

    def new_bucket():
        return {"real": [], "noise": [], "perm": [],
                "cross_noise": [], "cross_perm": []}

    accum["__overall__"] = new_bucket()

    # ── Main loop ────────────────────────────────────────────────────────
    for task in args.tasks:
        if task not in task_to_eps:
            print(f"[skip] task {task!r} not in dataset")
            continue
        accum[task] = new_bucket()

        ep_pool = task_to_eps[task]
        ep_rng = random.Random(args.seed + hash(task) % 10000)
        sample_eps = ep_rng.sample(ep_pool, min(args.n_episodes, len(ep_pool)))

        print(f"─── Task: {task}")
        print(f"    Sampling {len(sample_eps)} episodes from {len(ep_pool)} available")

        for ep_idx in sample_eps:
            try:
                frames = load_episode_frames(root, chunks_size, ep_idx, args.image_key, transform)
            except Exception as e:
                print(f"    [skip ep {ep_idx}] {e}")
                continue
            if len(frames) < frames_in + frame_gap + frames_out:
                continue

            goal = frames[-1].unsqueeze(0).to(device) if use_goal else None

            # Collect a pool of real targets for the "permutation" swap.
            windows = list(iter_windows(frames, frames_in, frames_out, frame_gap,
                                        args.n_windows, rng))
            target_pool = [w[1] for w in windows]

            for win_idx, (ctx_cpu, tgt_cpu) in enumerate(windows):
                ctx = ctx_cpu.unsqueeze(0).to(device)
                tgt = tgt_cpu.unsqueeze(0).to(device)

                # (1) Real target
                _, loss_real = forward_with_target(model, ctx, tgt, goal, device, args.amp)

                # (2) Uniform noise target in [-1, 1] (matches training input range)
                noise = (torch.rand_like(tgt) * 2 - 1)
                pred_noise, loss_noise = forward_with_target(model, ctx, noise, goal, device, args.amp)
                cross_noise = F.mse_loss(pred_noise, tgt.float()).item()

                # (3) Permuted target from another window in the same episode
                other = [t for i, t in enumerate(target_pool) if i != win_idx]
                if other:
                    perm_tgt = rng.choice(other).unsqueeze(0).to(device)
                    pred_perm, loss_perm = forward_with_target(model, ctx, perm_tgt, goal, device, args.amp)
                    cross_perm = F.mse_loss(pred_perm, tgt.float()).item()
                else:
                    loss_perm = cross_perm = float("nan")

                for bucket_name in (task, "__overall__"):
                    b = accum[bucket_name]
                    b["real"].append(loss_real)
                    b["noise"].append(loss_noise)
                    b["perm"].append(loss_perm)
                    b["cross_noise"].append(cross_noise)
                    b["cross_perm"].append(cross_perm)

        # Per-task summary
        _print_summary(task, accum[task])

    # Overall summary
    print()
    _print_summary("OVERALL", accum["__overall__"])

    # ── Verdict ───────────────────────────────────────────────────────────
    b = accum["__overall__"]
    _print_verdict(b)


def _mean(xs: list[float]) -> float:
    xs = [x for x in xs if not (isinstance(x, float) and np.isnan(x))]
    return float(np.mean(xs)) if xs else float("nan")


def _print_summary(label: str, b: dict[str, list[float]]):
    n = len(b["real"])
    if n == 0:
        print(f"    [{label}] no windows collected")
        return
    print(f"    [{label}] n={n}")
    print(f"        loss_real         = {_mean(b['real']):.6f}   (baseline TF with real target)")
    print(f"        loss_noise        = {_mean(b['noise']):.6f}   (TF with noise target — ≈ loss_real  ⇒ model echoes input)")
    print(f"        loss_perm         = {_mean(b['perm']):.6f}   (TF with swapped target — ≈ loss_real ⇒ model echoes input)")
    print(f"        cross_noise→real  = {_mean(b['cross_noise']):.6f}   (pred(noise) vs real — ≈ loss_real ⇒ HEALTHY)")
    print(f"        cross_perm→real   = {_mean(b['cross_perm']):.6f}   (pred(swapped) vs real — ≈ loss_real ⇒ HEALTHY)")


def _print_verdict(b: dict[str, list[float]]):
    real = _mean(b["real"])
    perm = _mean(b["perm"])
    cross_perm = _mean(b["cross_perm"])

    print("=" * 72)
    print("VERDICT")
    print("=" * 72)

    # The decisive signal is `loss_perm / loss_real`:
    # - Permuted targets are in-distribution (real images), so if the model
    #   echoes its target input, it will reproduce them at the same fidelity
    #   as the correct target → ratio ≈ 1.
    # - `loss_noise` is misleading because pure noise is OOD for patch_embed,
    #   so even an echoing model can't fully echo noise; the ratio will look
    #   "good" even when the model is cheating.
    echo_ratio = perm / real if real > 0 else float("nan")
    cross_ratio_perm = cross_perm / real if real > 0 else float("nan")

    print(f"loss_perm / loss_real        = {echo_ratio:.2f}   (DECISIVE)")
    print(f"    ≈ 1  → model reproduces a swapped real-image target just as")
    print(f"           well as the correct target → echoing / cheating.")
    print(f"    ≫ 1  → model ignores target input → healthy prediction.")
    print()
    print(f"cross_perm  / loss_real      = {cross_ratio_perm:.2f}")
    print(f"    ≈ 1  → pred is the real future regardless of target input (HEALTHY)")
    print(f"    ≫ 1  → pred tracks the target input rather than the real future (BAD)")
    print()

    if echo_ratio < 3.0 and cross_ratio_perm > 50.0:
        print("=>  TRAINING IS CHEATING. The model is echoing target-block inputs")
        print("    through the block-bidirectional mask. Fix the mask in")
        print("    cosmos_model.py `make_sequence_mask` to strict token-causal")
        print("    (`q_idx >= kv_idx`) and retrain.")
    elif cross_ratio_perm < 3.0:
        print("=>  TRAINING LOOKS HEALTHY. Model predicts the real future even")
        print("    when the target slot is replaced with a different real frame.")
        print("    Rollout issues (if any) are in the inference code path.")
    else:
        print("=>  PARTIAL ECHOING. Model is leaking some target-block info but")
        print("    not fully. Retraining with token-causal mask recommended.")


if __name__ == "__main__":
    main()
