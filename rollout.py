"""
rollout.py — Evaluate a pretrained Stage 1 model on train or val data.

Two evaluation modes
--------------------
  1. Teacher-forcing loss across the chosen split.
  2. Open-loop autoregressive rollout on selected episodes.

Usage
-----
    # Evaluate on training data (default)
    python rollout.py --checkpoint /path/to/best.pt --dataset /scr2/zhaoyang/libero

    # Evaluate on val (held-out) tasks
    python rollout.py --checkpoint /path/to/best.pt --split val

    # Rollout specific episodes
    python rollout.py --checkpoint /path/to/best.pt --split val \
        --episodes 0 5 10 --n-steps 30

    # Loss only / rollout only
    python rollout.py --checkpoint /path/to/best.pt --loss-only
    python rollout.py --checkpoint /path/to/best.pt --rollout-only --n-episodes 5
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from omegaconf import OmegaConf
from tqdm import tqdm

from eval import (
    load_model_from_checkpoint,
    frames_to_uint8,
    save_video,
)
from train import VideoFrameDataset


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def get_split_episodes(dataset_root: str, data_cfg) -> dict[str, list[int]]:
    """Return {split_name: [episode_indices]} using the same logic as VideoFrameDataset."""
    root = Path(dataset_root)

    with open(root / "meta" / "info.json") as f:
        info = json.load(f)

    episode_meta = []
    with open(root / "meta" / "episodes.jsonl") as f:
        for line in f:
            rec = json.loads(line)
            tasks = rec.get("tasks", [])
            episode_meta.append({
                "episode_index": int(rec["episode_index"]),
                "length": int(rec["length"]),
                "task": tasks[0] if tasks else None,
            })

    # Determine held-out test tasks (same priority as VideoFrameDataset)
    task_names = []
    tasks_path = root / "meta" / "tasks.jsonl"
    if tasks_path.exists():
        with open(tasks_path) as f:
            for line in f:
                rec = json.loads(line)
                if "task" in rec:
                    task_names.append(rec["task"])
    if not task_names:
        seen = set()
        for rec in episode_meta:
            if rec["task"] and rec["task"] not in seen:
                seen.add(rec["task"])
                task_names.append(rec["task"])

    configured_test_tasks = list(data_cfg.get("test_tasks", []))
    test_tasks_meta_path = root / "meta" / "test_tasks.json"
    if not configured_test_tasks and test_tasks_meta_path.exists():
        with open(test_tasks_meta_path) as f:
            stored = json.load(f)
        if isinstance(stored, dict):
            configured_test_tasks = list(stored.get("tasks", []))
        else:
            configured_test_tasks = list(stored)

    test_task_count = int(data_cfg.get("test_task_count", 0))
    if configured_test_tasks:
        selected_test_tasks = configured_test_tasks
    elif test_task_count > 0:
        selected_test_tasks = task_names[:test_task_count]
    else:
        selected_test_tasks = []

    test_task_set = set(selected_test_tasks)
    ep_tasks = {rec["episode_index"]: rec["task"] for rec in episode_meta}

    train_eps = [rec["episode_index"] for rec in episode_meta
                 if rec["task"] not in test_task_set]
    val_eps = [rec["episode_index"] for rec in episode_meta
               if rec["task"] in test_task_set]

    # Build task → episodes mapping for display
    task_to_eps: dict[str, list[int]] = {}
    for rec in episode_meta:
        task_to_eps.setdefault(rec["task"], []).append(rec["episode_index"])

    print(f"Train episodes: {len(train_eps)} ({len(set(ep_tasks[e] for e in train_eps))} tasks)")
    print(f"Val episodes  : {len(val_eps)} ({len(set(ep_tasks[e] for e in val_eps))} tasks)")
    if selected_test_tasks:
        print(f"Val tasks     : {selected_test_tasks}")

    return {"train": train_eps, "val": val_eps}


def load_episode(dataset_root: str, episode_idx: int):
    """Load all rows from a single episode parquet."""
    root = Path(dataset_root)
    with open(root / "meta" / "info.json") as f:
        info = json.load(f)
    chunks_size = info["chunks_size"]
    chunk_idx = episode_idx // chunks_size
    path = (root / "data" / f"chunk-{chunk_idx:03d}"
            / f"episode_{episode_idx:06d}.parquet")
    if not path.exists():
        raise FileNotFoundError(f"Episode not found: {path}")
    return pd.read_parquet(path)


# ---------------------------------------------------------------------------
# 1. Teacher-forcing loss
# ---------------------------------------------------------------------------

def compute_loss(
    model,
    model_cfg,
    train_cfg: dict,
    split: str,
    device: torch.device,
    batch_size: int = 64,
    num_workers: int = 8,
    max_batches: int = 0,
) -> float:
    """Compute average teacher-forcing MSE on the given split."""
    data_cfg = OmegaConf.create(train_cfg["data"])
    mcfg = OmegaConf.create({
        "resolution":  model_cfg.resolution,
        "frames_in":   model_cfg.frames_in,
        "frames_out":  model_cfg.frames_out,
        "patch_size":  model_cfg.patch_size,
    })

    dataset = VideoFrameDataset(data_cfg, mcfg, split=split)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    use_goal = data_cfg.get("use_goal", False)
    total_loss = 0.0
    n_batches = 0

    model.eval()
    with torch.no_grad():
        for batch_idx, (context, target, _actions, goal) in enumerate(
            tqdm(loader, desc=f"TF loss [{split}]")
        ):
            if 0 < max_batches <= batch_idx:
                break

            context = context.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            goal = goal.to(device, non_blocking=True) if use_goal else None

            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                _, loss = model(context, target, goal)

            total_loss += loss.item()
            n_batches += 1

    return total_loss / max(n_batches, 1)


# ---------------------------------------------------------------------------
# 2. Open-loop autoregressive rollout
# ---------------------------------------------------------------------------

def rollout_episode(
    model,
    model_cfg,
    df: pd.DataFrame,
    image_key: str,
    frame_gap: int,
    n_steps: int,
    device: torch.device,
) -> tuple[list[np.ndarray], list[np.ndarray], list[float]]:
    """
    Open-loop AR rollout on one episode.

    Returns
    -------
    gt_uint8   : list of (H, W, 3) uint8   GT frames at rollout positions
    pred_uint8 : list of (H, W, 3) uint8   model predictions
    step_mse   : list of float              per-step MSE
    """
    res = model_cfg.resolution
    tfm = transforms.Compose([
        transforms.Resize((res, res)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])

    def read_frame(idx: int) -> torch.Tensor:
        raw = df.iloc[idx][image_key]
        img = Image.open(io.BytesIO(raw["bytes"])).convert("RGB")
        return tfm(img)

    ep_len = len(df)
    max_steps = (ep_len - 1) // frame_gap if frame_gap > 0 else ep_len - 1
    n_steps = min(n_steps, max_steps)
    if n_steps <= 0:
        print(f"  Episode too short ({ep_len} frames) for gap={frame_gap}")
        return [], [], []

    # GT frames at positions 0, gap, 2*gap, ...
    gt_tensors = [read_frame(k * frame_gap) for k in range(n_steps + 1)]

    # Initial context (shared between GT and prediction)
    ctx = gt_tensors[0].unsqueeze(0).unsqueeze(0).to(device)  # (1,1,C,H,W)
    init_uint8 = frames_to_uint8(gt_tensors[0].unsqueeze(0))[0]

    gt_uint8 = [init_uint8]
    pred_uint8 = [init_uint8]
    step_mse: list[float] = []

    model.eval()
    with torch.no_grad():
        for k in range(n_steps):
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                pred, _ = model(ctx)  # (1,1,C,H,W)

            gt_t = gt_tensors[k + 1].unsqueeze(0).unsqueeze(0).to(device)
            step_mse.append(F.mse_loss(pred, gt_t).item())

            pred_uint8.append(frames_to_uint8(pred[0])[0])
            gt_uint8.append(frames_to_uint8(gt_t[0])[0])

            ctx = pred  # open-loop: feed prediction back

    return gt_uint8, pred_uint8, step_mse


def save_rollout_video(
    gt_frames: list[np.ndarray],
    pred_frames: list[np.ndarray],
    output_path: str,
    fps: int = 4,
):
    """Save side-by-side [GT | pred] rollout video."""
    gt_arr = np.stack(gt_frames)
    pred_arr = np.stack(pred_frames)
    H = gt_arr.shape[1]
    separator = np.ones((len(gt_arr), H, 4, 3), dtype=np.uint8) * 128
    combined = np.concatenate([gt_arr, separator, pred_arr], axis=2)
    save_video(combined, output_path, fps=fps)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Stage 1 model via teacher-forcing loss and open-loop rollout"
    )
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="/scr2/zhaoyang/libero")
    parser.add_argument("--image-key", type=str, default="image")
    parser.add_argument("--split", choices=["train", "val"], default="train",
                        help="Data split to evaluate on")
    parser.add_argument("--use-ema", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")

    # Teacher-forcing loss
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--max-batches", type=int, default=0,
                        help="Limit batches for loss (0 = all)")

    # Rollout
    parser.add_argument("--episodes", type=int, nargs="+", default=None,
                        help="Specific episode indices (default: auto-pick from split)")
    parser.add_argument("--n-episodes", type=int, default=3,
                        help="Number of episodes to auto-pick when --episodes is not given")
    parser.add_argument("--n-steps", type=int, default=30,
                        help="Max AR rollout steps per episode")
    parser.add_argument("--fps", type=int, default=4)
    parser.add_argument("--output-dir", type=str, default=None)

    # Mode
    parser.add_argument("--loss-only", action="store_true")
    parser.add_argument("--rollout-only", action="store_true")
    args = parser.parse_args()

    device = torch.device(
        args.device if torch.cuda.is_available() or args.device != "cuda" else "cpu"
    )
    print(f"Device: {device}")

    # ── Load model ────────────────────────────────────────────────────────
    model, model_cfg, train_cfg = load_model_from_checkpoint(
        args.checkpoint, device, args.use_ema,
    )
    frame_gap = int(train_cfg.get("data", {}).get("frame_gap", 1))
    print(f"Frame gap: {frame_gap}")

    if args.dataset:
        train_cfg.setdefault("data", {})["root"] = args.dataset

    data_cfg = OmegaConf.create(train_cfg["data"])

    # ── Resolve episodes per split ────────────────────────────────────────
    split_eps = get_split_episodes(args.dataset, data_cfg)

    if args.episodes is not None:
        rollout_eps = args.episodes
    else:
        pool = split_eps[args.split]
        if not pool:
            print(f"No episodes in split '{args.split}'")
            rollout_eps = []
        else:
            # Evenly space across the split
            n = min(args.n_episodes, len(pool))
            indices = np.linspace(0, len(pool) - 1, n, dtype=int)
            rollout_eps = [pool[i] for i in indices]

    if args.output_dir is None:
        ckpt_stem = Path(args.checkpoint).stem
        args.output_dir = f"rollout_results/{ckpt_stem}_{args.split}"

    # ── 1) Teacher-forcing loss ───────────────────────────────────────────
    if not args.rollout_only:
        print(f"\n{'=' * 60}")
        print(f"Teacher-forcing loss on [{args.split}] ...")
        print(f"{'=' * 60}")
        avg_loss = compute_loss(
            model, model_cfg, train_cfg, args.split, device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            max_batches=args.max_batches,
        )
        print(f"\n>>> [{args.split}] teacher-forcing MSE: {avg_loss:.6f}")

    # ── 2) Open-loop rollout ──────────────────────────────────────────────
    if not args.loss_only and rollout_eps:
        print(f"\n{'=' * 60}")
        print(f"Open-loop rollout on [{args.split}] episodes: {rollout_eps}")
        print(f"{'=' * 60}")

        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        all_mses: list[float] = []

        for ep_idx in rollout_eps:
            print(f"\n--- Episode {ep_idx} ---")
            try:
                df = load_episode(args.dataset, ep_idx)
            except FileNotFoundError as e:
                print(f"  Skipping: {e}")
                continue

            print(f"  Length: {len(df)} frames")
            gt_frames, pred_frames, step_mse = rollout_episode(
                model, model_cfg, df, args.image_key,
                frame_gap=frame_gap,
                n_steps=args.n_steps,
                device=device,
            )
            if not step_mse:
                continue

            all_mses.extend(step_mse)
            print(f"  Steps     : {len(step_mse)}")
            print(f"  Avg MSE   : {np.mean(step_mse):.6f}")
            print(f"  Step1 MSE : {step_mse[0]:.6f}")
            print(f"  Last  MSE : {step_mse[-1]:.6f}")

            # Save video
            video_path = str(out_dir / f"ep{ep_idx:04d}.mp4")
            save_rollout_video(gt_frames, pred_frames, video_path, fps=args.fps)

            # Save per-step MSE
            mse_path = out_dir / f"ep{ep_idx:04d}_mse.txt"
            with open(mse_path, "w") as f:
                f.write("step\tmse\n")
                for i, mse in enumerate(step_mse):
                    f.write(f"{i + 1}\t{mse:.6f}\n")

        if all_mses:
            print(f"\n>>> [{args.split}] rollout MSE (all episodes): {np.mean(all_mses):.6f}")

    print("\nDone!")


if __name__ == "__main__":
    main()
