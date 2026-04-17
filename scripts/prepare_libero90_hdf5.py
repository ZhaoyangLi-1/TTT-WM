#!/usr/bin/env python
"""Build a parquet dataset for train.py from raw LIBERO-90 HDF5 files.

This script can optionally merge an existing parquet dataset root first, then
append converted LIBERO-90 episodes after it. It also randomly samples a small
set of LIBERO-90 source tasks as held-out tasks, excludes them from the merged
parquet output, and writes their metadata into `meta/test_tasks.json`.

Example:
    python scripts/prepare_libero90_hdf5.py \
        --base-root /scr2/zhaoyang/libero \
        --input-root /scr2/zhaoyang/LIBERO-data/libero_90 \
        --output-root /scr2/zhaoyang/libero_wm\
        --num-test-tasks 3 \
        --seed 42 \
        --num-workers 48 \
        --overwrite
"""

from __future__ import annotations

import argparse
import io
import json
import random
import shutil
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
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
    parser.add_argument("--fps", type=int, default=10, help="Frame rate written into meta/info.json.")
    parser.add_argument("--camera-key", type=str, default="agentview_rgb", help="Observation key used as the main image column.")
    parser.add_argument("--wrist-camera-key", type=str, default="eye_in_hand_rgb", help="Observation key used as the wrist_image column.")
    parser.add_argument("--num-test-tasks", type=int, default=3, help="Number of LIBERO-90 tasks to hold out into meta/test_tasks.json.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for held-out task sampling.")
    parser.add_argument("--max-tasks", type=int, default=None, help="Optional limit for smoke tests.")
    parser.add_argument("--max-demos-per-task", type=int, default=None, help="Optional limit for smoke tests.")
    parser.add_argument("--max-base-episodes", type=int, default=None, help="Optional limit when copying the base parquet dataset.")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Parallel worker processes for LIBERO-90 demo conversion. 1 disables parallelism.",
    )
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


def _decode_h5_attr(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


def _fallback_instruction(task_stem: str) -> str:
    parts = task_stem.split("_")
    scene_pos = next((i for i, part in enumerate(parts) if part.startswith("SCENE")), None)
    if scene_pos is None:
        return task_stem.replace("_", " ")
    return " ".join(parts[scene_pos + 1 :]).replace("_", " ")


def _extract_scene_name(task_stem: str) -> str:
    parts = task_stem.split("_")
    scene_pos = next((i for i, part in enumerate(parts) if part.startswith("SCENE")), None)
    if scene_pos is None:
        return ""
    scene_start = max(scene_pos - 1, 0)
    return "_".join(parts[scene_start : scene_pos + 1])


def parse_hdf5_task_record(h5_file: h5py.File, fallback_name: str) -> dict[str, str]:
    data_group = h5_file["data"]
    problem_info = data_group.attrs.get("problem_info")
    parsed_problem_info: dict[str, Any] = {}
    if problem_info:
        parsed_problem_info = json.loads(_decode_h5_attr(problem_info))

    bddl_file_name = _decode_h5_attr(data_group.attrs.get("bddl_file_name", ""))
    source_task = Path(bddl_file_name).stem if bddl_file_name else fallback_name.removesuffix("_demo.hdf5")
    instruction = parsed_problem_info.get("language_instruction") or _fallback_instruction(source_task)
    scene = _extract_scene_name(source_task)

    # Keep per-scene task identity so repeated instructions across scenes stay distinct.
    task_name = f"{scene}: {instruction}" if scene else instruction
    return {
        "task": task_name,
        "instruction": instruction,
        "source_task": source_task,
        "scene": scene,
        "source_file": fallback_name,
    }


def sorted_demo_keys(data_group: h5py.Group) -> list[str]:
    return sorted(data_group.keys(), key=lambda key: int(key.split("_")[-1]))


def register_task(
    task_name: str,
    task_to_idx: dict[str, int],
    task_records: list[dict[str, Any]],
    task_record: dict[str, Any] | None = None,
) -> int:
    if task_name not in task_to_idx:
        task_index = len(task_records)
        task_to_idx[task_name] = task_index
        record = dict(task_record or {})
        record["task"] = task_name
        record["task_index"] = task_index
        task_records.append(record)
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
    task_record: dict[str, Any] | None = None,
) -> None:
    task_index = register_task(task_name, task_to_idx, task_records, task_record=task_record)
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


def collect_hdf5_task_records(hdf5_files: list[Path]) -> list[dict[str, str]]:
    task_records = []
    for hdf5_path in hdf5_files:
        with h5py.File(hdf5_path, "r") as h5_file:
            task_records.append(parse_hdf5_task_record(h5_file, hdf5_path.name))
    return task_records


def _convert_hdf5_demo_job(job: dict[str, Any]) -> dict[str, Any]:
    hdf5_path = Path(job["hdf5_path"])
    output_path = Path(job["output_path"])

    with h5py.File(hdf5_path, "r") as h5_file:
        demo = h5_file["data"][job["demo_key"]]
        obs = demo["obs"]
        images = obs[job["camera_key"]][:]
        wrist_images = obs[job["wrist_camera_key"]][:]
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
                f"{hdf5_path.name}:{job['demo_key']} has {images.shape[0]} images but {actions.shape[0]} actions."
            )
        if wrist_images.shape[0] != images.shape[0]:
            raise ValueError(
                f"{hdf5_path.name}:{job['demo_key']} has {wrist_images.shape[0]} wrist frames but {images.shape[0]} images."
            )

        rows = []
        for frame_idx in range(images.shape[0]):
            rows.append(
                {
                    "image": {
                        "bytes": encode_png(images[frame_idx]),
                        "path": f"{hdf5_path.stem}/{job['demo_key']}/image/{frame_idx:06d}.png",
                    },
                    "wrist_image": {
                        "bytes": encode_png(wrist_images[frame_idx]),
                        "path": f"{hdf5_path.stem}/{job['demo_key']}/wrist_image/{frame_idx:06d}.png",
                    },
                    "state": state_vec[frame_idx],
                    "actions": actions[frame_idx],
                    "timestamp": np.float32(frame_idx / job["fps"]),
                    "frame_index": np.int64(frame_idx),
                    "episode_index": np.int64(job["episode_index"]),
                    "index": np.int64(job["global_frame_index"] + frame_idx),
                    "task_index": np.int64(job["task_index"]),
                }
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(output_path, index=False)
    return {
        "episode_index": int(job["episode_index"]),
        "length": int(job["length"]),
        "output_path": str(output_path),
    }


def build_hdf5_demo_jobs(
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
    heldout_tasks: set[str],
) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    for hdf5_path in tqdm(hdf5_files, desc="scan_libero90", leave=False):
        with h5py.File(hdf5_path, "r") as h5_file:
            data_group = h5_file["data"]
            task_record = parse_hdf5_task_record(h5_file, hdf5_path.name)
            task_name = task_record["task"]
            if task_name in heldout_tasks:
                continue

            task_index = register_task(
                task_name, task_to_idx, task_records, task_record=task_record
            )

            demo_keys = sorted_demo_keys(data_group)
            if max_demos_per_task is not None:
                demo_keys = demo_keys[:max_demos_per_task]

            for demo_key in demo_keys:
                demo = data_group[demo_key]
                obs = demo["obs"]
                frame_count = int(obs[camera_key].shape[0])
                action_count = int(demo["actions"].shape[0])

                if frame_count != action_count:
                    raise ValueError(
                        f"{hdf5_path.name}:{demo_key} has {frame_count} images but {action_count} actions."
                    )

                if state["image_shape"] is None:
                    state["image_shape"] = list(obs[camera_key].shape[1:])

                episode_index = int(state["episode_index"])
                global_frame_index = int(state["global_frame_index"])
                output_path = (
                    output_root
                    / "data"
                    / f"chunk-{episode_index // chunks_size:03d}"
                    / f"episode_{episode_index:06d}.parquet"
                )

                jobs.append(
                    {
                        "hdf5_path": str(hdf5_path),
                        "demo_key": demo_key,
                        "camera_key": camera_key,
                        "wrist_camera_key": wrist_camera_key,
                        "fps": fps,
                        "task_index": task_index,
                        "episode_index": episode_index,
                        "global_frame_index": global_frame_index,
                        "length": frame_count,
                        "output_path": str(output_path),
                    }
                )
                episode_records.append(
                    {
                        "episode_index": episode_index,
                        "tasks": [task_name],
                        "length": frame_count,
                    }
                )
                state["episode_index"] += 1
                state["global_frame_index"] += frame_count
                state["total_frames"] += frame_count

    return jobs


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
    heldout_tasks: set[str],
    num_workers: int,
) -> None:
    jobs = build_hdf5_demo_jobs(
        hdf5_files=hdf5_files,
        output_root=output_root,
        chunks_size=chunks_size,
        fps=fps,
        camera_key=camera_key,
        wrist_camera_key=wrist_camera_key,
        task_to_idx=task_to_idx,
        task_records=task_records,
        episode_records=episode_records,
        state=state,
        max_demos_per_task=max_demos_per_task,
        heldout_tasks=heldout_tasks,
    )

    if not jobs:
        return

    if num_workers <= 1:
        iterator = map(_convert_hdf5_demo_job, jobs)
        for _ in tqdm(iterator, total=len(jobs), desc="libero90_demos"):
            pass
        return

    chunksize = max(1, len(jobs) // (num_workers * 8))
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        iterator = executor.map(_convert_hdf5_demo_job, jobs, chunksize=chunksize)
        for _ in tqdm(iterator, total=len(jobs), desc=f"libero90_demos[{num_workers}p]"):
            pass


def write_metadata(
    output_root: Path,
    chunks_size: int,
    fps: int,
    task_records: list[dict[str, Any]],
    episode_records: list[dict[str, Any]],
    state: dict[str, Any],
    sampled_test_tasks: list[str],
    heldout_task_records: list[dict[str, Any]],
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
                "records": heldout_task_records,
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
        "num_heldout_tasks": len(sampled_test_tasks),
    }
    with open(output_root / "meta" / "stats.json", "w") as f:
        json.dump(stats, f, indent=4)


def main() -> None:
    args = parse_args()
    if args.num_workers < 1:
        raise ValueError("--num-workers must be >= 1")
    ensure_output_root(args.output_root, overwrite=args.overwrite)

    hdf5_files = sorted(args.input_root.glob("*.hdf5"))
    if args.max_tasks is not None:
        hdf5_files = hdf5_files[: args.max_tasks]
    if not hdf5_files:
        raise FileNotFoundError(f"No .hdf5 files found under {args.input_root}")

    hdf5_task_records = collect_hdf5_task_records(hdf5_files)
    hdf5_task_names = [rec["task"] for rec in hdf5_task_records]
    duplicate_task_names = sorted(
        task_name for task_name, count in Counter(hdf5_task_names).items() if count > 1
    )
    if duplicate_task_names:
        raise ValueError(
            "Expected unique LIBERO-90 task ids after scene-aware parsing, but found duplicates: "
            + ", ".join(duplicate_task_names)
        )

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
    heldout_task_set = set(sampled_test_tasks)
    heldout_task_records = [
        rec for rec in hdf5_task_records if rec["task"] in heldout_task_set
    ]

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
        heldout_tasks=heldout_task_set,
        num_workers=args.num_workers,
    )

    write_metadata(
        output_root=args.output_root,
        chunks_size=args.chunks_size,
        fps=args.fps,
        task_records=task_records,
        episode_records=episode_records,
        state=state,
        sampled_test_tasks=sampled_test_tasks,
        heldout_task_records=heldout_task_records,
        seed=args.seed,
    )

    print(
        f"Wrote {state['episode_index']} episodes across {len(task_records)} tasks to {args.output_root}\n"
        f"Held-out tasks ({len(sampled_test_tasks)}): {sampled_test_tasks}"
    )


if __name__ == "__main__":
    main()
