#!/usr/bin/env python
"""Build a parquet dataset for train.py from raw LIBERO-90 HDF5 files.

This script can optionally merge an existing parquet dataset root first, then
append converted LIBERO-90 episodes after it. It also randomly samples a small
set of LIBERO-90 tasks as held-out tasks and writes them into
`meta/test_tasks.json`, which train.py can read automatically.

Example:
    python scripts/prepare_libero90_hdf5.py \
        --base-root /scr2/zhaoyang/libero \
        --input-root /scr2/zhaoyang/LIBERO-data/libero_90 \
        --output-root /scr2/zhaoyang/libero_combined \
        --num-test-tasks 3 \
        --seed 42 \
        --overwrite
"""

from __future__ import annotations

import argparse
import io
import json
import random
import shutil
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True, help="Directory containing LIBERO-90 *.hdf5 files.")
    parser.add_argument("--output-root", type=Path, required=True, help="Output parquet dataset root.")
    parser.add_argument("--base-root", type=Path, default=None, help="Optional existing parquet dataset root to merge first, e.g. /scr2/zhaoyang/libero.")
    parser.add_argument("--chunks-size", type=int, default=1000, help="Episodes per parquet chunk directory.")
    parser.add_argument("--fps", type=int, default=20, help="Frame rate written into meta/info.json.")
    parser.add_argument("--camera-key", type=str, default="agentview_rgb", help="Observation key used as the main image column.")
    parser.add_argument("--wrist-camera-key", type=str, default="eye_in_hand_rgb", help="Observation key used as the wrist_image column.")
    parser.add_argument("--num-test-tasks", type=int, default=3, help="Number of LIBERO-90 tasks to hold out into meta/test_tasks.json.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for held-out task sampling.")
    parser.add_argument("--max-tasks", type=int, default=None, help="Optional limit for smoke tests.")
    parser.add_argument("--max-demos-per-task", type=int, default=None, help="Optional limit for smoke tests.")
    parser.add_argument("--max-base-episodes", type=int, default=None, help="Optional limit when copying the base parquet dataset.")
    parser.add_argument("--overwrite", action="store_true", help="Remove output_root before writing.")
    return parser.parse_args()


def ensure_output_root(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"{path} already exists. Pass --overwrite to replace it.")
        shutil.rmtree(path)

    (path / "data").mkdir(parents=True, exist_ok=True)
    (path / "meta").mkdir(parents=True, exist_ok=True)


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


def register_task(task_name: str, task_to_idx: dict[str, int], task_records: list[dict[str, Any]]) -> int:
    if task_name not in task_to_idx:
        task_to_idx[task_name] = len(task_records)
        task_records.append({"task_index": task_to_idx[task_name], "task": task_name})
    return task_to_idx[task_name]


def write_episode_dataframe(
    df: pd.DataFrame,
    task_name: str,
    output_root: Path,
    chunks_size: int,
    task_to_idx: dict[str, int],
    task_records: list[dict[str, Any]],
    episode_records: list[dict[str, Any]],
    state: dict[str, Any],
) -> None:
    task_index = register_task(task_name, task_to_idx, task_records)
    length = len(df)

    df = df.copy()
    df["episode_index"] = np.int64(state["episode_index"])
    df["index"] = np.arange(
        state["global_frame_index"],
        state["global_frame_index"] + length,
        dtype=np.int64,
    )
    df["task_index"] = np.int64(task_index)

    if "frame_index" not in df.columns:
        df["frame_index"] = np.arange(length, dtype=np.int64)
    if "timestamp" not in df.columns:
        df["timestamp"] = np.arange(length, dtype=np.float32)

    chunk_dir = output_root / "data" / f"chunk-{state['episode_index'] // chunks_size:03d}"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = chunk_dir / f"episode_{state['episode_index']:06d}.parquet"
    df.to_parquet(parquet_path, index=False)

    episode_records.append(
        {
            "episode_index": state["episode_index"],
            "tasks": [task_name],
            "length": length,
        }
    )

    state["episode_index"] += 1
    state["global_frame_index"] += length
    state["total_frames"] += length


def merge_base_dataset(
    base_root: Path,
    output_root: Path,
    chunks_size: int,
    task_to_idx: dict[str, int],
    task_records: list[dict[str, Any]],
    episode_records: list[dict[str, Any]],
    state: dict[str, Any],
    max_base_episodes: int | None,
) -> None:
    with open(base_root / "meta" / "info.json") as f:
        base_info = json.load(f)

    if state["image_shape"] is None:
        state["image_shape"] = list(base_info["features"]["image"]["shape"])

    with open(base_root / "meta" / "episodes.jsonl") as f:
        base_episode_recs = [json.loads(line) for line in f]

    if max_base_episodes is not None:
        base_episode_recs = base_episode_recs[:max_base_episodes]

    base_chunks_size = int(base_info["chunks_size"])
    for rec in tqdm(base_episode_recs, desc="base_episodes"):
        episode_index = int(rec["episode_index"])
        task_name = rec["tasks"][0]
        parquet_path = (
            base_root
            / "data"
            / f"chunk-{episode_index // base_chunks_size:03d}"
            / f"episode_{episode_index:06d}.parquet"
        )
        df = pd.read_parquet(parquet_path)
        write_episode_dataframe(
            df=df,
            task_name=task_name,
            output_root=output_root,
            chunks_size=chunks_size,
            task_to_idx=task_to_idx,
            task_records=task_records,
            episode_records=episode_records,
            state=state,
        )


def collect_hdf5_task_names(hdf5_files: list[Path]) -> list[str]:
    task_names = []
    for hdf5_path in hdf5_files:
        with h5py.File(hdf5_path, "r") as h5_file:
            task_names.append(parse_task_name(h5_file, hdf5_path.name))
    return task_names


def convert_hdf5_dataset(
    hdf5_files: list[Path],
    output_root: Path,
    chunks_size: int,
    fps: int,
    camera_key: str,
    wrist_camera_key: str,
    task_to_idx: dict[str, int],
    task_records: list[dict[str, Any]],
    episode_records: list[dict[str, Any]],
    state: dict[str, Any],
    max_demos_per_task: int | None,
) -> None:
    for hdf5_path in tqdm(hdf5_files, desc="libero90_tasks"):
        with h5py.File(hdf5_path, "r") as h5_file:
            data_group = h5_file["data"]
            task_name = parse_task_name(h5_file, hdf5_path.name)

            demo_keys = sorted_demo_keys(data_group)
            if max_demos_per_task is not None:
                demo_keys = demo_keys[:max_demos_per_task]

            for demo_key in demo_keys:
                demo = data_group[demo_key]
                obs = demo["obs"]
                images = obs[camera_key][:]
                wrist_images = obs[wrist_camera_key][:]
                actions = demo["actions"][:].astype(np.float32)
                state_vec = np.concatenate(
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

                if state["image_shape"] is None:
                    state["image_shape"] = list(images.shape[1:])

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
                            "state": state_vec[frame_idx],
                            "actions": actions[frame_idx],
                            "timestamp": np.float32(frame_idx / fps),
                            "frame_index": np.int64(frame_idx),
                        }
                    )

                df = pd.DataFrame(rows)
                write_episode_dataframe(
                    df=df,
                    task_name=task_name,
                    output_root=output_root,
                    chunks_size=chunks_size,
                    task_to_idx=task_to_idx,
                    task_records=task_records,
                    episode_records=episode_records,
                    state=state,
                )


def write_metadata(
    output_root: Path,
    chunks_size: int,
    fps: int,
    task_records: list[dict[str, Any]],
    episode_records: list[dict[str, Any]],
    state: dict[str, Any],
    sampled_test_tasks: list[str],
    seed: int,
) -> None:
    if state["image_shape"] is None:
        raise RuntimeError("Failed to infer image shape from the source datasets.")

    with open(output_root / "meta" / "tasks.jsonl", "w") as f:
        for rec in task_records:
            f.write(json.dumps(rec) + "\n")

    with open(output_root / "meta" / "episodes.jsonl", "w") as f:
        for rec in episode_records:
            f.write(json.dumps(rec) + "\n")

    with open(output_root / "meta" / "test_tasks.json", "w") as f:
        json.dump(
            {
                "seed": seed,
                "num_test_tasks": len(sampled_test_tasks),
                "tasks": sampled_test_tasks,
            },
            f,
            indent=4,
        )

    info = {
        "codebase_version": "custom-libero90-merged",
        "robot_type": "panda",
        "total_episodes": state["episode_index"],
        "total_frames": state["total_frames"],
        "total_tasks": len(task_records),
        "total_videos": 0,
        "total_chunks": int(np.ceil(state["episode_index"] / chunks_size)),
        "chunks_size": chunks_size,
        "fps": fps,
        "splits": {"train": f"0:{state['episode_index']}"},
        "test_tasks_path": "meta/test_tasks.json",
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {
            "image": {"dtype": "image", "shape": state["image_shape"], "names": ["height", "width", "channel"]},
            "wrist_image": {"dtype": "image", "shape": state["image_shape"], "names": ["height", "width", "channel"]},
            "state": {"dtype": "float32", "shape": [8], "names": ["state"]},
            "actions": {"dtype": "float32", "shape": [7], "names": ["actions"]},
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None},
        },
    }
    with open(output_root / "meta" / "info.json", "w") as f:
        json.dump(info, f, indent=4)

    stats = {
        "total_episodes": state["episode_index"],
        "total_frames": state["total_frames"],
        "total_tasks": len(task_records),
        "test_tasks": sampled_test_tasks,
    }
    with open(output_root / "meta" / "stats.json", "w") as f:
        json.dump(stats, f, indent=4)


def main() -> None:
    args = parse_args()
    ensure_output_root(args.output_root, overwrite=args.overwrite)

    hdf5_files = sorted(args.input_root.glob("*.hdf5"))
    if args.max_tasks is not None:
        hdf5_files = hdf5_files[: args.max_tasks]
    if not hdf5_files:
        raise FileNotFoundError(f"No .hdf5 files found under {args.input_root}")

    hdf5_task_names = collect_hdf5_task_names(hdf5_files)
    if args.num_test_tasks < 0:
        raise ValueError("--num-test-tasks must be >= 0")
    if args.num_test_tasks > len(hdf5_task_names):
        raise ValueError(
            f"--num-test-tasks={args.num_test_tasks} is larger than the number of available "
            f"HDF5 tasks ({len(hdf5_task_names)})."
        )

    sampled_test_tasks = (
        sorted(random.Random(args.seed).sample(hdf5_task_names, args.num_test_tasks))
        if args.num_test_tasks > 0
        else []
    )

    task_to_idx: dict[str, int] = {}
    task_records: list[dict[str, Any]] = []
    episode_records: list[dict[str, Any]] = []
    state = {
        "episode_index": 0,
        "global_frame_index": 0,
        "total_frames": 0,
        "image_shape": None,
    }

    if args.base_root is not None:
        merge_base_dataset(
            base_root=args.base_root,
            output_root=args.output_root,
            chunks_size=args.chunks_size,
            task_to_idx=task_to_idx,
            task_records=task_records,
            episode_records=episode_records,
            state=state,
            max_base_episodes=args.max_base_episodes,
        )

    convert_hdf5_dataset(
        hdf5_files=hdf5_files,
        output_root=args.output_root,
        chunks_size=args.chunks_size,
        fps=args.fps,
        camera_key=args.camera_key,
        wrist_camera_key=args.wrist_camera_key,
        task_to_idx=task_to_idx,
        task_records=task_records,
        episode_records=episode_records,
        state=state,
        max_demos_per_task=args.max_demos_per_task,
    )

    write_metadata(
        output_root=args.output_root,
        chunks_size=args.chunks_size,
        fps=args.fps,
        task_records=task_records,
        episode_records=episode_records,
        state=state,
        sampled_test_tasks=sampled_test_tasks,
        seed=args.seed,
    )

    print(
        f"Wrote {state['episode_index']} episodes across {len(task_records)} tasks to {args.output_root}\n"
        f"Held-out tasks ({len(sampled_test_tasks)}): {sampled_test_tasks}"
    )


if __name__ == "__main__":
    main()
