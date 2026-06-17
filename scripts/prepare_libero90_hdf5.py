#!/usr/bin/env python
"""Build a parquet dataset for train.py from raw LIBERO-90 HDF5 files.

This script can optionally merge an existing parquet dataset root first, then
append converted LIBERO-90 episodes after it. It also randomly samples a small
set of LIBERO-90 source tasks as held-out tasks and writes their metadata into
`meta/test_tasks.json`. Held-out task episodes are still converted into the
parquet output so downstream evaluation can read them; train.py filters them
out at load time via `meta/test_tasks.json`.

No-op filtering (NEW):
    By default, frames whose action is a "no-op" (near-zero translation +
    rotation AND an unchanged gripper command) are dropped before writing,
    mirroring OpenVLA's LIBERO preprocessing. This prevents expressive policies
    (e.g. diffusion policy) from learning to idle/freeze. Disable with
    `--no-filter-noops`; tune sensitivity with `--noop-eps`.

Example:
    python scripts/prepare_libero90_hdf5.py \
        --base-root /scr2/zhaoyang/libero \
        --input-root /scr2/zhaoyang/LIBERO-data/libero_90 \
        --output-root /scr2/zhaoyang/libero_wm\
        --num-test-tasks 3 \
        --seed 42 \
        --num-workers 48 \
        --filter-noops \
        --overwrite

Rerunning without `--overwrite` resumes an existing output root. Tasks already
recorded in `meta/tasks.jsonl` are skipped instead of being converted again.
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

import sys

import h5py
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

# Make the repo root importable so we can reuse the canonical rotation helper
# even when this script is launched as `python scripts/prepare_libero90_hdf5.py`
# (in which case only scripts/ — not the repo root — is on sys.path).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dp.action_utils import axis_angle_to_quat  # noqa: E402

# Dimensionality of derived proprio columns added on top of the raw LIBERO-90
# obs. joint_pos = 7-DoF Panda joints; ee_quat = scalar-last [x,y,z,w] end
# effector orientation converted from the stored axis-angle. Both are used to
# backfill schema-consistent NaN columns for base episodes that predate them.
JOINT_POS_DIM = 7
EE_QUAT_DIM = 4


def compute_keep_mask(actions: np.ndarray, eps: float) -> np.ndarray:
    """Boolean mask (True = KEEP) marking the non-no-op frames of one demo.

    Mirrors OpenVLA's LIBERO no-op definition: a frame is treated as a no-op
    when its 6-DoF translation+rotation action component is within ``eps`` of
    zero AND its gripper command is unchanged from the previous frame. Such
    frames teach expressive single-step / chunked policies to idle and freeze
    during rollout, so they are dropped from training. The first frame is always
    kept so every episode keeps a well-defined start.

    ``actions`` is expected to be ``(T, 7)`` = [dx, dy, dz, d_roll, d_pitch,
    d_yaw, gripper].
    """
    actions = np.asarray(actions, dtype=np.float32)
    n = actions.shape[0]
    if n == 0:
        return np.ones(0, dtype=bool)
    # Per-frame max magnitude over the 6 motion dims (translation + rotation).
    motion = np.max(np.abs(actions[:, :6]), axis=-1)
    gripper = actions[:, -1]
    prev_gripper = np.concatenate(([gripper[0]], gripper[:-1]))
    gripper_unchanged = np.isclose(gripper, prev_gripper)
    is_noop = (motion < eps) & gripper_unchanged
    is_noop[0] = False  # never drop the first frame
    return ~is_noop


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
    parser.add_argument(
        "--filter-noops",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Drop no-op frames (near-zero translation+rotation AND unchanged "
            "gripper) before writing, mirroring OpenVLA's LIBERO filtering. This "
            "is on by default; pass --no-filter-noops to keep every frame."
        ),
    )
    parser.add_argument(
        "--noop-eps",
        type=float,
        default=1e-3,
        help=(
            "Max |action| over the 6 translation/rotation dims that still counts "
            "as 'no motion'. Increase if too few frames are dropped, decrease if "
            "real motion is being filtered. Check the reported drop fraction."
        ),
    )
    parser.add_argument(
        "--on-missing-joint-pos",
        choices=["nan", "error"],
        default="nan",
        help=(
            "What to do when a --base-root parquet lacks a derived proprio column "
            "(joint_pos or ee_quat). 'nan' backfills a schema-consistent all-NaN "
            "column (and warns); 'error' aborts so the schema mismatch is fixed "
            "upstream."
        ),
    )
    return parser.parse_args()


def ensure_output_root(path: Path, overwrite: bool) -> None:
    if path.exists():
        if overwrite:
            shutil.rmtree(path)

    (path / "data").mkdir(parents=True, exist_ok=True)
    (path / "meta").mkdir(parents=True, exist_ok=True)


def encode_png(image: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(image).save(buf, format="PNG")
    return buf.getvalue()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def load_existing_output_dataset(output_root: Path) -> dict[str, Any] | None:
    info_path = output_root / "meta" / "info.json"
    tasks_path = output_root / "meta" / "tasks.jsonl"
    episodes_path = output_root / "meta" / "episodes.jsonl"

    if not any(path.exists() for path in (info_path, tasks_path, episodes_path)):
        data_dir = output_root / "data"
        meta_dir = output_root / "meta"
        has_untracked_outputs = any(data_dir.iterdir()) or any(meta_dir.iterdir())
        if has_untracked_outputs:
            raise FileNotFoundError(
                f"Cannot resume from {output_root}; found existing files under data/ or meta/ but missing "
                "required metadata (meta/info.json, meta/tasks.jsonl, meta/episodes.jsonl). "
                "Pass --overwrite to rebuild from scratch."
            )
        return None

    missing = [str(path.relative_to(output_root)) for path in (info_path, tasks_path, episodes_path) if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"Cannot resume from {output_root}; missing required metadata files: {', '.join(missing)}. "
            "Pass --overwrite to rebuild from scratch."
        )

    with open(info_path) as f:
        info = json.load(f)
    task_records = _read_jsonl(tasks_path)
    episode_records = _read_jsonl(episodes_path)

    task_to_idx: dict[str, int] = {}
    normalized_task_records: list[dict[str, Any]] = []
    for idx, rec in enumerate(task_records):
        task_name = rec.get("task")
        if not task_name:
            raise ValueError(f"{tasks_path} contains a task record without `task`.")
        if task_name in task_to_idx:
            raise ValueError(f"{tasks_path} contains duplicate task entries for {task_name!r}.")
        normalized_rec = dict(rec)
        normalized_rec["task_index"] = idx
        task_to_idx[task_name] = idx
        normalized_task_records.append(normalized_rec)

    next_episode_index = max((int(rec["episode_index"]) for rec in episode_records), default=-1) + 1
    total_frames = sum(int(rec["length"]) for rec in episode_records)
    image_shape = info.get("features", {}).get("image", {}).get("shape")

    return {
        "task_to_idx": task_to_idx,
        "task_records": normalized_task_records,
        "episode_records": episode_records,
        "state": {
            "episode_index": next_episode_index,
            "global_frame_index": total_frames,
            "total_frames": total_frames,
            "image_shape": list(image_shape) if image_shape is not None else None,
        },
    }


def load_existing_test_tasks(output_root: Path) -> list[str]:
    test_tasks_path = output_root / "meta" / "test_tasks.json"
    if not test_tasks_path.exists():
        return []

    with open(test_tasks_path) as f:
        payload = json.load(f)
    if isinstance(payload, dict):
        tasks = payload.get("tasks", [])
    else:
        tasks = payload
    return [str(task) for task in tasks]


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


# Derived proprio columns that freshly converted LIBERO-90 episodes carry but
# older base datasets may not: (column name, dimensionality).
DERIVED_COLUMNS: tuple[tuple[str, int], ...] = (
    ("joint_pos", JOINT_POS_DIM),
    ("ee_quat", EE_QUAT_DIM),
)


def _ensure_derived_columns(df: pd.DataFrame, on_missing: str, parquet_path: Path) -> list[str]:
    """Guarantee the dataframe carries every derived proprio column so merged
    base episodes share the schema of freshly converted LIBERO-90 episodes.

    Returns the list of columns that had to be backfilled. With
    ``on_missing="error"`` a missing column aborts instead, surfacing the schema
    mismatch upstream rather than letting NaN proprio silently flow into training.
    """
    backfilled: list[str] = []
    for col, dim in DERIVED_COLUMNS:
        if col in df.columns:
            continue
        if on_missing == "error":
            raise ValueError(
                f"Base parquet {parquet_path} has no `{col}` column but the output "
                f"schema requires it. Re-generate the base dataset with {col}, or "
                f"pass --on-missing-joint-pos nan to backfill an all-NaN [{dim}] column."
            )
        nan_row = np.full(dim, np.nan, dtype=np.float32)
        df[col] = [nan_row.copy() for _ in range(len(df))]
        backfilled.append(col)
    return backfilled


def _filter_base_df_noops(df: pd.DataFrame, eps: float) -> tuple[pd.DataFrame, int]:
    """Drop no-op rows from a merged base-dataset dataframe.

    Returns the filtered dataframe and the number of dropped rows. Stale
    per-frame index columns are removed so write_episode_dataframe regenerates
    them contiguously.
    """
    if "actions" not in df.columns or len(df) == 0:
        return df, 0
    actions_arr = np.stack([np.asarray(a, dtype=np.float32) for a in df["actions"].to_numpy()])
    keep_mask = compute_keep_mask(actions_arr, eps)
    dropped = int((~keep_mask).sum())
    if dropped == 0:
        return df, 0
    df = df.loc[keep_mask].reset_index(drop=True)
    df = df.drop(columns=[c for c in ("index", "frame_index", "timestamp") if c in df.columns])
    return df, dropped


def merge_base_dataset(
    base_root: Path,
    output_root: Path,
    chunks_size: int,
    task_to_idx: dict[str, int],
    task_records: list[dict[str, Any]],
    episode_records: list[dict[str, Any]],
    state: dict[str, Any],
    max_base_episodes: int | None,
    on_missing_joint_pos: str,
    filter_noops: bool,
    noop_eps: float,
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
    backfill_counts: Counter[str] = Counter()
    base_dropped_frames = 0
    base_raw_frames = 0
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
        backfill_counts.update(_ensure_derived_columns(df, on_missing_joint_pos, parquet_path))
        base_raw_frames += len(df)
        if filter_noops:
            df, dropped = _filter_base_df_noops(df, noop_eps)
            base_dropped_frames += dropped
            if len(df) == 0:
                # Entirely no-op base episode; skip so we never write an empty parquet.
                continue
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

    for col, count in backfill_counts.items():
        print(
            f"WARNING: {count} base episode(s) lacked a `{col}` column; backfilled an "
            f"all-NaN column so the merged dataset stays schema-consistent. Any policy "
            f"that conditions on `{col}` must mask/skip these episodes (re-generate the "
            f"base dataset with `{col}` to avoid NaNs)."
        )
    if filter_noops and base_raw_frames > 0:
        print(
            f"[no-op filter] base dataset: dropped {base_dropped_frames}/{base_raw_frames} "
            f"frames ({100.0 * base_dropped_frames / base_raw_frames:.1f}%)."
        )


def collect_hdf5_task_records(hdf5_files: list[Path]) -> list[dict[str, str]]:
    task_records = []
    for hdf5_path in hdf5_files:
        with h5py.File(hdf5_path, "r") as h5_file:
            record = parse_hdf5_task_record(h5_file, hdf5_path.name)
            record["hdf5_path"] = str(hdf5_path)
            task_records.append(record)
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
        # Raw 7-DoF Panda joint positions, stored as a separate column so
        # downstream policies can condition on joint space without re-reading
        # the source HDF5.
        joint_pos = obs["joint_states"][:].astype(np.float32)
        if joint_pos.shape[0] != images.shape[0]:
            raise ValueError(
                f"{hdf5_path.name}:{job['demo_key']} has {joint_pos.shape[0]} joint_states "
                f"but {images.shape[0]} images."
            )
        # End-effector orientation as a scalar-last [x,y,z,w] quaternion,
        # converted from the stored axis-angle (ee_states[:, 3:6]) and
        # canonicalized to w>=0. Lets a policy condition on a 4D quat (matching
        # the lpb diffusion-policy obs) instead of the 3D axis-angle in `state`.
        ee_quat = axis_angle_to_quat(
            obs["ee_states"][:, 3:6].astype(np.float32)
        ).astype(np.float32)

        if images.shape[0] != actions.shape[0]:
            raise ValueError(
                f"{hdf5_path.name}:{job['demo_key']} has {images.shape[0]} images but {actions.shape[0]} actions."
            )
        if wrist_images.shape[0] != images.shape[0]:
            raise ValueError(
                f"{hdf5_path.name}:{job['demo_key']} has {wrist_images.shape[0]} wrist frames but {images.shape[0]} images."
            )

        # Which source frames survive no-op filtering. Computed at scan time and
        # passed in via the job so the pre-allocated global indices match exactly.
        keep_indices = job.get("keep_indices")
        if keep_indices is None:
            keep_indices = list(range(images.shape[0]))

        rows = []
        # `new_idx` is the contiguous index in the written episode (post-filter);
        # `frame_idx` is the original source frame it came from.
        for new_idx, frame_idx in enumerate(keep_indices):
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
                    "joint_pos": joint_pos[frame_idx],
                    "ee_quat": ee_quat[frame_idx],
                    "actions": actions[frame_idx],
                    "timestamp": np.float32(new_idx / job["fps"]),
                    "frame_index": np.int64(new_idx),
                    "episode_index": np.int64(job["episode_index"]),
                    "index": np.int64(job["global_frame_index"] + new_idx),
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
    filter_noops: bool,
    noop_eps: float,
) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    total_raw_frames = 0
    total_dropped_frames = 0
    skipped_empty_demos = 0
    for hdf5_path in tqdm(hdf5_files, desc="scan_libero90", leave=False):
        with h5py.File(hdf5_path, "r") as h5_file:
            data_group = h5_file["data"]
            task_record = parse_hdf5_task_record(h5_file, hdf5_path.name)
            task_name = task_record["task"]

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

                # Compute which frames survive no-op filtering *here*, at scan
                # time, so the global frame indices and episode lengths we
                # allocate below reflect the post-filter counts (no gaps).
                if filter_noops:
                    actions_arr = demo["actions"][:].astype(np.float32)
                    keep_mask = compute_keep_mask(actions_arr, noop_eps)
                else:
                    keep_mask = np.ones(frame_count, dtype=bool)
                keep_indices = np.nonzero(keep_mask)[0].tolist()
                kept_count = len(keep_indices)

                total_raw_frames += frame_count
                total_dropped_frames += frame_count - kept_count

                if kept_count == 0:
                    # Degenerate demo (entirely no-op) — skip it entirely so we
                    # never allocate an index range or write an empty parquet.
                    skipped_empty_demos += 1
                    continue

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
                        "length": kept_count,
                        "keep_indices": keep_indices,
                        "output_path": str(output_path),
                    }
                )
                episode_records.append(
                    {
                        "episode_index": episode_index,
                        "tasks": [task_name],
                        "length": kept_count,
                    }
                )
                state["episode_index"] += 1
                state["global_frame_index"] += kept_count
                state["total_frames"] += kept_count

    if filter_noops and total_raw_frames > 0:
        print(
            f"[no-op filter] LIBERO-90: dropped {total_dropped_frames}/{total_raw_frames} "
            f"frames ({100.0 * total_dropped_frames / total_raw_frames:.1f}%); "
            f"skipped {skipped_empty_demos} fully-no-op demo(s)."
        )

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
    num_workers: int,
    filter_noops: bool,
    noop_eps: float,
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
        filter_noops=filter_noops,
        noop_eps=noop_eps,
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
            "joint_pos": {"dtype": "float32", "shape": [7], "names": ["joint_pos"]},
            "ee_quat": {"dtype": "float32", "shape": [4], "names": ["ee_quat"]},
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
    if args.noop_eps < 0:
        raise ValueError("--noop-eps must be >= 0")
    ensure_output_root(args.output_root, overwrite=args.overwrite)
    existing_output = load_existing_output_dataset(args.output_root)

    hdf5_files = sorted(args.input_root.glob("*.hdf5"))
    if args.max_tasks is not None:
        hdf5_files = hdf5_files[: args.max_tasks]
    if not hdf5_files:
        raise FileNotFoundError(f"No .hdf5 files found under {args.input_root}")

    hdf5_task_records = collect_hdf5_task_records(hdf5_files)

    if existing_output is not None:
        task_to_idx = existing_output["task_to_idx"]
        task_records = existing_output["task_records"]
        episode_records = existing_output["episode_records"]
        state = existing_output["state"]
        existing_task_names = {rec["task"] for rec in task_records}
        existing_source_files = {rec["source_file"] for rec in task_records if rec.get("source_file")}
        pending_hdf5_task_records = [
            rec
            for rec in hdf5_task_records
            if rec["task"] not in existing_task_names and rec["source_file"] not in existing_source_files
        ]
        skipped_task_names = sorted(
            rec["task"]
            for rec in hdf5_task_records
            if rec["task"] in existing_task_names or rec["source_file"] in existing_source_files
        )
        if skipped_task_names:
            print(
                f"Skipping {len(skipped_task_names)} already-saved LIBERO-90 tasks in {args.output_root}."
            )
        if args.base_root is not None:
            print(
                f"Resuming existing dataset at {args.output_root}; skipping --base-root merge from {args.base_root}."
            )
    else:
        task_to_idx = {}
        task_records = []
        episode_records = []
        state = {
            "episode_index": 0,
            "global_frame_index": 0,
            "total_frames": 0,
            "image_shape": None,
        }
        pending_hdf5_task_records = hdf5_task_records

    existing_hdf5_task_records = [rec for rec in task_records if rec.get("source_file")]
    all_hdf5_task_records = existing_hdf5_task_records + pending_hdf5_task_records
    hdf5_task_names = [rec["task"] for rec in all_hdf5_task_records]
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

    existing_test_tasks = load_existing_test_tasks(args.output_root)
    if existing_test_tasks:
        known_hdf5_tasks = set(hdf5_task_names)
        preserved_test_tasks = []
        seen_test_tasks = set()
        for task in existing_test_tasks:
            if task in known_hdf5_tasks and task not in seen_test_tasks:
                preserved_test_tasks.append(task)
                seen_test_tasks.add(task)
        additional_needed = max(0, args.num_test_tasks - len(preserved_test_tasks))
        additional_candidates = sorted(
            rec["task"]
            for rec in pending_hdf5_task_records
            if rec["task"] not in preserved_test_tasks
        )
        if additional_needed > len(additional_candidates):
            raise ValueError(
                f"Need {additional_needed} additional held-out tasks to reach --num-test-tasks={args.num_test_tasks}, "
                f"but only found {len(additional_candidates)} new HDF5 tasks to sample from."
            )
        sampled_test_tasks = sorted(
            preserved_test_tasks
            + random.Random(args.seed).sample(additional_candidates, additional_needed)
        )
    else:
        sampled_test_tasks = (
            sorted(random.Random(args.seed).sample(hdf5_task_names, args.num_test_tasks))
            if args.num_test_tasks > 0
            else []
        )
    heldout_task_set = set(sampled_test_tasks)
    heldout_task_records = [
        {key: value for key, value in rec.items() if key != "hdf5_path"}
        for rec in all_hdf5_task_records
        if rec["task"] in heldout_task_set
    ]

    if args.base_root is not None and existing_output is None:
        merge_base_dataset(
            base_root=args.base_root,
            output_root=args.output_root,
            chunks_size=args.chunks_size,
            task_to_idx=task_to_idx,
            task_records=task_records,
            episode_records=episode_records,
            state=state,
            max_base_episodes=args.max_base_episodes,
            on_missing_joint_pos=args.on_missing_joint_pos,
            filter_noops=args.filter_noops,
            noop_eps=args.noop_eps,
        )

    convert_hdf5_dataset(
        hdf5_files=[Path(rec["hdf5_path"]) for rec in pending_hdf5_task_records],
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
        num_workers=args.num_workers,
        filter_noops=args.filter_noops,
        noop_eps=args.noop_eps,
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
        f"Held-out tasks (stored in parquet, skipped by train.py via meta/test_tasks.json, "
        f"count={len(sampled_test_tasks)}): {sampled_test_tasks}"
    )


if __name__ == "__main__":
    main()