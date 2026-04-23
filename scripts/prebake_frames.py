"""One-shot preprocessing: decode every parquet's PNG frames to uint8 tensors
and save alongside as .pt files.

Training's `VideoFrameDataset` currently opens each episode's parquet on
every random `__getitem__`, deserializes the whole DataFrame, and PIL-decodes
the 3 needed PNGs. With ~6k episodes and a 64-entry LRU cache, hit rate is
~1% under shuffling, so ~99% of samples pay the full parquet + decode cost
(~400ms/sample → 35s/batch).

This script does that decode + resize once up front and writes:

    episode_XXXXXX.pt  # dict with:
        'frames':  uint8 tensor (T, resolution, resolution, 3)
        'actions': float32 tensor (T, action_dim)

Each .pt is small (~3-6 MB for a 200-frame episode at 128x128 uint8).
Total size: ~20-40 GB for the LIBERO held-out dataset — fits comfortably
on /linting-fast-vol. The OS page cache warms up on first epoch, making
subsequent reads essentially free.

Usage:
    python scripts/prebake_frames.py \\
        --dataset /linting-fast-vol/libero_wm \\
        --resolution 128 \\
        --num-workers 12
"""

from __future__ import annotations

import argparse
import io
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm


def process_one(args: tuple[str, int, bool]) -> tuple[str, str]:
    parquet_path, resolution, force = args
    pq = Path(parquet_path)
    out = pq.with_suffix(".pt")
    if out.exists() and not force:
        return str(pq), "skip"

    try:
        df = pd.read_parquet(pq)
        frames = np.empty((len(df), resolution, resolution, 3), dtype=np.uint8)
        for i, value in enumerate(df["image"]):
            raw = value["bytes"] if isinstance(value, dict) else value
            img = Image.open(io.BytesIO(raw)).convert("RGB")
            if img.size != (resolution, resolution):
                img = img.resize((resolution, resolution), Image.BILINEAR)
            frames[i] = np.asarray(img)

        actions = np.asarray(df["actions"].tolist(), dtype=np.float32)
        payload = {
            "frames":  torch.from_numpy(frames),
            "actions": torch.from_numpy(actions),
        }
        tmp = out.with_suffix(".pt.tmp")
        torch.save(payload, tmp)
        tmp.rename(out)
        return str(pq), "ok"
    except Exception as e:
        return str(pq), f"ERR: {e}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, help="Dataset root")
    p.add_argument("--resolution", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--force", action="store_true", help="Re-bake even if .pt exists")
    args = p.parse_args()

    root = Path(args.dataset) / "data"
    parquets = sorted(root.rglob("*.parquet"))
    if not parquets:
        raise SystemExit(f"No parquet files under {root}")

    print(f"Found {len(parquets)} parquets under {root}")
    print(f"Resolution: {args.resolution}x{args.resolution}, workers: {args.num_workers}")

    jobs = [(str(p), args.resolution, args.force) for p in parquets]
    n_ok = n_skip = n_err = 0
    with ProcessPoolExecutor(max_workers=args.num_workers) as ex:
        futures = [ex.submit(process_one, job) for job in jobs]
        for fut in tqdm(as_completed(futures), total=len(futures)):
            pq, status = fut.result()
            if status == "ok":
                n_ok += 1
            elif status == "skip":
                n_skip += 1
            else:
                n_err += 1
                print(f"\n[err] {pq}: {status}")

    print(f"\nDone. ok={n_ok}, skipped={n_skip}, errors={n_err}")


if __name__ == "__main__":
    main()
