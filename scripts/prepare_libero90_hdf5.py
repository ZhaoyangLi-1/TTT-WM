#!/usr/bin/env python
"""Convert raw LIBERO-90 HDF5 demos into the parquet layout used by train.py.

Example:
    python scripts/prepare_libero90_hdf5.py \
        --input-root /scr2/zhaoyang/LIBERO-data/libero_90 \
        --output-root /scr2/zhaoyang/libero_90_parquet
"""

from __future__ import annotations

import argparse
import io
import json
import shutil
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True, help="Directory containing LIBERO-90 *.hdf5 files.")
    parser.add_argument("--output-root", type=Path, required=True, help="Output parquet dataset root.")
    parser.add_argument("--chunks-size", type=int, default=1000, help="Episodes per parquet chunk directory.")
    parser.add_argument("--fps", type=int, default=20, help="Frame rate written into meta/info.json.")
    parser.add_argument("--camera-key", type=str, default="agentview_rgb", help="Observation key used as the main image column.")
    parser.add_argument("--wrist-camera-key", type=str, default="eye_in_hand_rgb", help="Observation key used as the wrist_image column.")
    parser.add_argument("--max-tasks", type=int, default=None, help="Optional limit for smoke tests.")
    parser.add_argument("--max-demos-per-task", type=int, default=None, help="Optional limit for smoke tests.")
    parser.add_argument("--overwrite", action="store_true", help="Remove output_root before writing.")
    return parser.parse_args()


def encode_png(image: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(image).save(buf, format="PNG")
    return buf.getvalue()


def parse_task_name(h5_file: h5py.File, fallback_name: str) -> str:
    data_group = h5_file["data"]
    problem_info = data_group.attrs.get("problem_info")
    if problem_info:
        info = json.loads(problem_info)
        language_instruction = info.get("language_instruction")
        if language_instruction:
            return language_instruction

    stem = fallback_name.removesuffix("_demo.hdf5")
    parts = stem.split("_")
    scene_pos = next((i for i, part in enumerate(parts) if part.startswith("SCENE")), None)
    if scene_pos is None:
        return stem.replace("_", " ")
    return " ".join(parts[scene_pos + 1 :]).replace("_", " ")


def sorted_demo_keys(data_group: h5py.Group) -> list[str]:
    return sorted(data_group.keys(), key=lambda key: int(key.split("_")[-1]))


def ensure_output_root(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"{path} already exists. Pass --overwrite to replace it.")
        shutil.rmtree(path)

    (path / "data").mkdir(parents=True, exist_ok=True)
    (path / "meta").mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()
    ensure_output_root(args.output_root, overwrite=args.overwrite)

    hdf5_files = sorted(args.input_root.glob("*.hdf5"))
    if args.max_tasks is not None:
        hdf5_files = hdf5_files[: args.max_tasks]

    if not hdf5_files:
        raise FileNotFoundError(f"No .hdf5 files found under {args.input_root}")

    task_records: list[dict] = []
    episode_records: list[dict] = []
    total_frames = 0
    episode_index = 0
    global_frame_index = 0
    image_shape = None

    for task_index, hdf5_path in enumerate(tqdm(hdf5_files, desc="tasks")):
        with h5py.File(hdf5_path, "r") as h5_file:
            data_group = h5_file["data"]
            task_name = parse_task_name(h5_file, hdf5_path.name)
            task_records.append({"task_index": task_index, "task": task_name})

            demo_keys = sorted_demo_keys(data_group)
            if args.max_demos_per_task is not None:
                demo_keys = demo_keys[: args.max_demos_per_task]

            for demo_key in demo_keys:
                demo = data_group[demo_key]
                obs = demo["obs"]
                images = obs[args.camera_key][:]
                wrist_images = obs[args.wrist_camera_key][:]
                actions = demo["actions"][:].astype(np.float32)
                state = np.concatenate(
                    [
                        obs["ee_states"][:].astype(np.float32),
                        obs["gripper_states"][:].astype(np.float32),
                    ],
                    axis=-1,
                )

                if images.shape[0] != actions.shape[0]:
                    raise ValueError(
                        f"{hdf5_path.name}:{demo_key} has {images.shape[0]} images but {actions.shape[0]} actions."
                    )

                if image_shape is None:
                    image_shape = list(images.shape[1:])

                rows = []
                for frame_idx in range(images.shape[0]):
                    rows.append(
                        {
                            "image": {
                                "bytes": encode_png(images[frame_idx]),
                                "path": f"{hdf5_path.stem}/{demo_key}/image/{frame_idx:06d}.png",
                            },
                            "wrist_image": {
                                "bytes": encode_png(wrist_images[frame_idx]),
                                "path": f"{hdf5_path.stem}/{demo_key}/wrist_image/{frame_idx:06d}.png",
                            },
                            "state": state[frame_idx],
                            "actions": actions[frame_idx],
                            "timestamp": np.float32(frame_idx / args.fps),
                            "frame_index": np.int64(frame_idx),
                            "episode_index": np.int64(episode_index),
                            "index": np.int64(global_frame_index + frame_idx),
                            "task_index": np.int64(task_index),
                        }
                    )

                chunk_dir = args.output_root / "data" / f"chunk-{episode_index // args.chunks_size:03d}"
                chunk_dir.mkdir(parents=True, exist_ok=True)
                parquet_path = chunk_dir / f"episode_{episode_index:06d}.parquet"
                pd.DataFrame(rows).to_parquet(parquet_path, index=False)

                episode_records.append(
                    {
                        "episode_index": episode_index,
                        "tasks": [task_name],
                        "length": len(rows),
                    }
                )
                total_frames += len(rows)
                global_frame_index += len(rows)
                episode_index += 1

    if image_shape is None:
        raise RuntimeError("Failed to infer image shape from the input demos.")

    with open(args.output_root / "meta" / "tasks.jsonl", "w") as f:
        for rec in task_records:
            f.write(json.dumps(rec) + "\n")

    with open(args.output_root / "meta" / "episodes.jsonl", "w") as f:
        for rec in episode_records:
            f.write(json.dumps(rec) + "\n")

    info = {
        "codebase_version": "custom-libero90",
        "robot_type": "panda",
        "total_episodes": episode_index,
        "total_frames": total_frames,
        "total_tasks": len(task_records),
        "total_videos": 0,
        "total_chunks": int(np.ceil(episode_index / args.chunks_size)),
        "chunks_size": args.chunks_size,
        "fps": args.fps,
        "splits": {"train": f"0:{episode_index}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {
            "image": {"dtype": "image", "shape": image_shape, "names": ["height", "width", "channel"]},
            "wrist_image": {"dtype": "image", "shape": image_shape, "names": ["height", "width", "channel"]},
            "state": {"dtype": "float32", "shape": [8], "names": ["state"]},
            "actions": {"dtype": "float32", "shape": [7], "names": ["actions"]},
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None},
        },
    }
    with open(args.output_root / "meta" / "info.json", "w") as f:
        json.dump(info, f, indent=4)

    stats = {
        "total_episodes": episode_index,
        "total_frames": total_frames,
        "total_tasks": len(task_records),
    }
    with open(args.output_root / "meta" / "stats.json", "w") as f:
        json.dump(stats, f, indent=4)

    print(
        f"Wrote {episode_index} episodes across {len(task_records)} tasks "
        f"to {args.output_root}"
    )


if __name__ == "__main__":
    main()
