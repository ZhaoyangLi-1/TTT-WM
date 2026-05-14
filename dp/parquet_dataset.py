from __future__ import annotations

import copy
import io
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from PIL import Image

from diffusion_policy.common.normalize_util import (
    get_image_range_normalizer,
    robomimic_abs_action_only_normalizer_from_stat,
    array_to_stats,
)
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.model.common.normalizer import (
    LinearNormalizer,
    SingleFieldLinearNormalizer,
)

from dp.action_utils import state_to_abs_action
from dp.common import dict_apply


def _to_container(value: Any) -> Any:
    if OmegaConf.is_config(value):
        return OmegaConf.to_container(value, resolve=True)
    return copy.deepcopy(value)


class TTTWMParquetImageDataset(BaseImageDataset):
    def __init__(
        self,
        shape_meta: dict,
        dataset_root: str,
        horizon: int,
        pad_before: int = 0,
        pad_after: int = 0,
        n_obs_steps: int | None = None,
        sample_mode: str = "horizon_pad",
        frame_gap: int | None = None,
        frames_out: int = 1,
        action_key: str = "actions",
        obs_key_mapping: dict[str, str] | None = None,
        split: str = "train",
        split_mode: str = "auto",
        val_ratio: float = 0.1,
        test_task_count: int = 0,
        test_tasks: list[str] | None = None,
        task_filter: str | None = None,
        seed: int = 42,
        max_train_episodes: int | None = None,
        cache_size: int = 32,
        verbose: bool = True,
        abs_action: bool = False,
        rotation_rep: str = "rotation_6d",
        state_key: str = "state",
        state_pos_slice: tuple[int, int] = (0, 3),
        state_rot_slice: tuple[int, int] = (3, 6),
    ):
        super().__init__()

        if split not in {"train", "val"}:
            raise ValueError(f"Unsupported split: {split}")

        shape_meta = _to_container(shape_meta)
        obs_key_mapping = _to_container(obs_key_mapping) or {}
        test_tasks = list(_to_container(test_tasks) or [])

        self.shape_meta = shape_meta
        self.dataset_root = str(Path(dataset_root).expanduser())
        self.root = Path(self.dataset_root)
        self.horizon = int(horizon)
        self.pad_before = int(pad_before)
        self.pad_after = int(pad_after)
        self.n_obs_steps = int(n_obs_steps) if n_obs_steps is not None else self.horizon
        self.sample_mode = str(sample_mode).lower()
        self.frame_gap = int(frame_gap) if frame_gap is not None else self.horizon
        self.frames_out = int(frames_out)
        self.action_key = str(action_key)
        self.obs_key_mapping = dict(obs_key_mapping)
        self.split = split
        self.split_mode = str(split_mode).lower()
        self.val_ratio = float(val_ratio)
        self.test_task_count = int(test_task_count)
        self.test_tasks = list(test_tasks)
        self.task_filter = (
            str(task_filter) if task_filter not in (None, "", "None") else None
        )
        self.seed = int(seed)
        self.max_train_episodes = max_train_episodes
        self.cache_size = max(int(cache_size), 1)
        self.verbose = bool(verbose)
        self.abs_action = bool(abs_action)
        self.rotation_rep = str(rotation_rep)
        self.state_key = str(state_key)
        self.state_pos_slice = tuple(int(v) for v in state_pos_slice)
        self.state_rot_slice = tuple(int(v) for v in state_rot_slice)

        if self.abs_action:
            if self.rotation_rep != "rotation_6d":
                raise ValueError(
                    f"Only rotation_rep='rotation_6d' is implemented for abs_action; got {self.rotation_rep!r}"
                )
            if len(self.state_pos_slice) != 2 or len(self.state_rot_slice) != 2:
                raise ValueError("state_pos_slice and state_rot_slice must be (start, end) pairs")
            pos_dim = self.state_pos_slice[1] - self.state_pos_slice[0]
            rot_dim = self.state_rot_slice[1] - self.state_rot_slice[0]
            if pos_dim != 3 or rot_dim != 3:
                raise ValueError(
                    f"abs_action expects 3D position and 3D axis-angle slices of state, got pos={pos_dim}, rot={rot_dim}"
                )

        if self.horizon < 1:
            raise ValueError("horizon must be >= 1")
        if self.n_obs_steps < 1:
            raise ValueError("n_obs_steps must be >= 1")
        if self.sample_mode not in {"horizon_pad", "video_frame"}:
            raise ValueError(
                f"Unsupported sample_mode={self.sample_mode}. Use horizon_pad or video_frame."
            )
        if self.sample_mode == "video_frame" and self.horizon != self.frame_gap:
            raise ValueError(
                "video_frame sample_mode requires horizon == frame_gap so the "
                "returned action sequence matches the policy horizon."
            )

        action_shape = tuple(shape_meta["action"]["shape"])
        if len(action_shape) != 1:
            raise ValueError(f"Only 1D actions are supported, got {action_shape}")
        self.action_shape = action_shape
        if self.abs_action and action_shape[0] != 10:
            raise ValueError(
                f"abs_action + rotation_6d requires shape_meta.action.shape=[10], got {list(action_shape)}"
            )

        obs_meta = shape_meta["obs"]
        self.rgb_keys: list[str] = []
        self.lowdim_keys: list[str] = []
        self.obs_shapes: dict[str, tuple[int, ...]] = {}
        self.obs_columns: dict[str, str] = {}
        for key, attr in obs_meta.items():
            obs_type = attr.get("type", "low_dim")
            self.obs_shapes[key] = tuple(attr["shape"])
            self.obs_columns[key] = self.obs_key_mapping.get(key, key)
            if obs_type == "rgb":
                self.rgb_keys.append(key)
            elif obs_type == "low_dim":
                self.lowdim_keys.append(key)
            else:
                raise ValueError(f"Unsupported obs type for {key}: {obs_type}")

        required_columns = [self.action_key] + [
            self.obs_columns[key] for key in self.rgb_keys + self.lowdim_keys
        ]
        if self.abs_action:
            required_columns.append(self.state_key)
        self._required_columns = list(dict.fromkeys(required_columns))

        self._episode_paths, episode_meta = self._load_episode_manifest()
        self._episode_meta = episode_meta
        (
            self._train_episode_indices,
            self._val_episode_indices,
            self._split_reason,
        ) = self._split_episode_indices(episode_meta)
        self._train_episode_indices = self._downsample_episode_indices(
            self._train_episode_indices, self.max_train_episodes
        )

        self._episode_indices = (
            self._train_episode_indices if split == "train" else self._val_episode_indices
        )

        self.samples: list[tuple[int, int]] = []
        if self.sample_mode == "video_frame":
            span = self.n_obs_steps + self.frame_gap + self.frames_out - 1
            for episode_idx in self._episode_indices:
                ep_length = self._episode_meta[episode_idx]["length"]
                if ep_length < span:
                    continue
                for start in range(ep_length - span + 1):
                    self.samples.append((episode_idx, start))
        else:
            for episode_idx in self._episode_indices:
                ep_length = self._episode_meta[episode_idx]["length"]
                min_start = -self.pad_before
                max_start = ep_length - self.horizon + self.pad_after
                for start in range(min_start, max_start + 1):
                    self.samples.append((episode_idx, start))

        if self.verbose:
            split_tasks = sorted(
                {
                    self._episode_meta[idx]["task"]
                    for idx in self._episode_indices
                    if self._episode_meta[idx]["task"] is not None
                }
            )
            print(
                f"[TTTWMParquetImageDataset:{self.split}] "
                f"{len(self.samples)} windows / {len(self._episode_indices)} episodes "
                f"/ {len(split_tasks)} tasks ({self._split_reason})"
            )

        self._parquet_cache: dict[str, pd.DataFrame] = {}
        self._abs_action_cache: dict[int, np.ndarray] = {}
        self._all_actions_cache: torch.Tensor | None = None

        self._init_kwargs = dict(
            shape_meta=shape_meta,
            dataset_root=self.dataset_root,
            horizon=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            n_obs_steps=self.n_obs_steps,
            sample_mode=self.sample_mode,
            frame_gap=self.frame_gap,
            frames_out=self.frames_out,
            action_key=self.action_key,
            obs_key_mapping=self.obs_key_mapping,
            split_mode=self.split_mode,
            val_ratio=self.val_ratio,
            test_task_count=self.test_task_count,
            test_tasks=self.test_tasks,
            task_filter=self.task_filter,
            seed=self.seed,
            max_train_episodes=self.max_train_episodes,
            cache_size=self.cache_size,
            verbose=self.verbose,
            abs_action=self.abs_action,
            rotation_rep=self.rotation_rep,
            state_key=self.state_key,
            state_pos_slice=self.state_pos_slice,
            state_rot_slice=self.state_rot_slice,
        )

    def _load_episode_manifest(self) -> tuple[dict[int, Path], dict[int, dict[str, Any]]]:
        info_path = self.root / "meta" / "info.json"
        episodes_path = self.root / "meta" / "episodes.jsonl"
        if not info_path.is_file() or not episodes_path.is_file():
            raise FileNotFoundError(
                f"{self.root} is missing `meta/info.json` or `meta/episodes.jsonl`."
            )

        with info_path.open() as f:
            info = json.load(f)

        total_episodes = int(info["total_episodes"])
        chunks_size = int(info["chunks_size"])
        episode_paths = {
            episode_idx: self.root
            / "data"
            / f"chunk-{episode_idx // chunks_size:03d}"
            / f"episode_{episode_idx:06d}.parquet"
            for episode_idx in range(total_episodes)
        }

        episode_meta: dict[int, dict[str, Any]] = {}
        with episodes_path.open() as f:
            for line in f:
                record = json.loads(line)
                tasks = record.get("tasks", [])
                episode_idx = int(record["episode_index"])
                episode_meta[episode_idx] = {
                    "episode_index": episode_idx,
                    "length": int(record["length"]),
                    "task": tasks[0] if tasks else None,
                }

        return episode_paths, episode_meta

    def _load_task_names(self, episode_meta: dict[int, dict[str, Any]]) -> list[str]:
        tasks_path = self.root / "meta" / "tasks.jsonl"
        task_names: list[str] = []
        if tasks_path.is_file():
            with tasks_path.open() as f:
                for line in f:
                    record = json.loads(line)
                    task_name = record.get("task")
                    if task_name:
                        task_names.append(task_name)

        if task_names:
            return task_names

        seen: set[str] = set()
        for episode_idx in sorted(episode_meta.keys()):
            task_name = episode_meta[episode_idx]["task"]
            if task_name and task_name not in seen:
                seen.add(task_name)
                task_names.append(task_name)
        return task_names

    def _resolve_heldout_tasks(self, task_names: list[str]) -> list[str]:
        selected = list(self.test_tasks)
        meta_path = self.root / "meta" / "test_tasks.json"
        if not selected and meta_path.is_file():
            with meta_path.open() as f:
                stored = json.load(f)
            selected = list(stored.get("tasks", []) if isinstance(stored, dict) else stored)

        if not selected and self.test_task_count > 0 and task_names:
            if self.test_task_count >= len(task_names):
                raise ValueError(
                    f"test_task_count={self.test_task_count} must be < total tasks ({len(task_names)})"
                )
            selected = task_names[: self.test_task_count]

        if selected:
            known_task_names = set(task_names)
            unknown = [task for task in selected if task not in known_task_names]
            if unknown and self.verbose:
                print(
                    "[TTTWMParquetImageDataset] ignoring held-out tasks absent from the "
                    f"current dataset root: {', '.join(unknown)}"
                )
            selected = [task for task in selected if task in known_task_names]
        return selected

    def _split_episode_indices(
        self, episode_meta: dict[int, dict[str, Any]]
    ) -> tuple[list[int], list[int], str]:
        task_names = self._load_task_names(episode_meta)
        all_episode_indices = sorted(episode_meta.keys())
        split_reason_parts: list[str] = []

        if self.split_mode not in {"auto", "heldout_tasks", "episode"}:
            raise ValueError(
                f"Unsupported split_mode={self.split_mode}. Use auto, heldout_tasks, or episode."
            )

        if self.task_filter is not None:
            matching_episode_indices = [
                idx
                for idx in all_episode_indices
                if episode_meta[idx]["task"] == self.task_filter
            ]
            if not matching_episode_indices:
                raise ValueError(
                    f"No episodes found for task_filter={self.task_filter!r} "
                    f"under dataset_root={self.dataset_root}."
                )
            all_episode_indices = matching_episode_indices
            split_reason_parts.append(f"task_filter={self.task_filter}")

        if self.task_filter is None and self.split_mode in {"auto", "heldout_tasks"} and task_names:
            heldout_tasks = self._resolve_heldout_tasks(task_names)
            if heldout_tasks:
                heldout_set = set(heldout_tasks)
                train_eps = [
                    idx for idx in all_episode_indices if episode_meta[idx]["task"] not in heldout_set
                ]
                val_eps = [
                    idx for idx in all_episode_indices if episode_meta[idx]["task"] in heldout_set
                ]
                return train_eps, val_eps, f"heldout_tasks={','.join(heldout_tasks)}"

        if self.val_ratio > 0.0 and len(all_episode_indices) > 1:
            n_val = min(
                max(1, round(len(all_episode_indices) * self.val_ratio)),
                len(all_episode_indices) - 1,
            )
            rng = np.random.default_rng(self.seed)
            shuffled = np.array(all_episode_indices)
            rng.shuffle(shuffled)
            val_eps = sorted(int(x) for x in shuffled[:n_val])
            val_set = set(val_eps)
            train_eps = [idx for idx in all_episode_indices if idx not in val_set]
            split_reason_parts.append(f"episode_val_ratio={self.val_ratio:.3f}")
            return train_eps, val_eps, ",".join(split_reason_parts)

        split_reason_parts.append("single_split")
        return all_episode_indices, [], ",".join(split_reason_parts)

    def _downsample_episode_indices(
        self, episode_indices: list[int], max_n: int | None
    ) -> list[int]:
        if max_n is None or len(episode_indices) <= int(max_n):
            return list(episode_indices)
        rng = np.random.default_rng(self.seed)
        selected = rng.choice(episode_indices, size=int(max_n), replace=False)
        return sorted(int(x) for x in selected)

    def _read_parquet(self, episode_idx: int) -> pd.DataFrame:
        path = self._episode_paths[episode_idx]
        cache_key = str(path)
        if cache_key in self._parquet_cache:
            df = self._parquet_cache.pop(cache_key)
            self._parquet_cache[cache_key] = df
            return df

        df = pd.read_parquet(path, columns=self._required_columns)
        self._parquet_cache[cache_key] = df
        while len(self._parquet_cache) > self.cache_size:
            oldest = next(iter(self._parquet_cache))
            del self._parquet_cache[oldest]
        return df

    def _compute_abs_actions(self, df: pd.DataFrame) -> np.ndarray:
        raw_actions = np.stack(
            [np.asarray(value, dtype=np.float32) for value in df[self.action_key].tolist()],
            axis=0,
        )
        states = np.stack(
            [np.asarray(value, dtype=np.float32) for value in df[self.state_key].tolist()],
            axis=0,
        )
        return state_to_abs_action(
            states,
            raw_actions,
            pos_slice=self.state_pos_slice,
            rot_slice=self.state_rot_slice,
        )

    def _get_abs_actions(self, episode_idx: int) -> np.ndarray:
        cached = self._abs_action_cache.get(episode_idx)
        if cached is not None:
            # move-to-end for LRU-ish behaviour
            self._abs_action_cache.pop(episode_idx)
            self._abs_action_cache[episode_idx] = cached
            return cached

        df = self._read_parquet(episode_idx)
        abs_actions = self._compute_abs_actions(df)
        self._abs_action_cache[episode_idx] = abs_actions
        while len(self._abs_action_cache) > self.cache_size:
            oldest = next(iter(self._abs_action_cache))
            del self._abs_action_cache[oldest]
        return abs_actions

    @staticmethod
    def _extract_image_bytes(value: Any) -> bytes | Any:
        if isinstance(value, dict) and "bytes" in value:
            return value["bytes"]
        return value

    def _decode_rgb_value(self, value: Any, obs_key: str) -> np.ndarray:
        value = self._extract_image_bytes(value)
        if isinstance(value, (bytes, bytearray)):
            image = Image.open(io.BytesIO(value)).convert("RGB")
        elif isinstance(value, Image.Image):
            image = value.convert("RGB")
        else:
            array = np.asarray(value)
            if array.ndim != 3:
                raise ValueError(
                    f"RGB observation `{obs_key}` must decode to HWC, got shape {array.shape}."
                )
            if array.dtype != np.uint8:
                array = np.clip(array, 0, 255).astype(np.uint8)
            image = Image.fromarray(array).convert("RGB")

        channels, height, width = self.obs_shapes[obs_key]
        if channels != 3:
            raise ValueError(f"RGB observation `{obs_key}` must have 3 channels, got {channels}.")
        if image.size != (width, height):
            image = image.resize((width, height), Image.BILINEAR)

        array = np.asarray(image, dtype=np.float32) / 255.0
        return np.moveaxis(array, -1, 0)

    def _decode_lowdim_value(self, value: Any, obs_key: str) -> np.ndarray:
        array = np.asarray(value, dtype=np.float32)
        expected_shape = self.obs_shapes[obs_key]
        return array.reshape(expected_shape)

    def _decode_action_value(self, value: Any) -> np.ndarray:
        array = np.asarray(value, dtype=np.float32)
        return array.reshape(self.action_shape)

    def _build_row_indices(self, episode_idx: int, start: int) -> list[int]:
        ep_length = self._episode_meta[episode_idx]["length"]
        return [
            min(max(start + offset, 0), ep_length - 1)
            for offset in range(self.horizon)
        ]

    def get_validation_dataset(self) -> "TTTWMParquetImageDataset":
        return self.__class__(**self._init_kwargs, split="val")

    def get_all_actions(self) -> torch.Tensor:
        if self._all_actions_cache is None:
            actions: list[np.ndarray] = []
            indices = self._train_episode_indices
            try:
                from tqdm import tqdm
                iterator = tqdm(indices, desc="Reading actions", unit="ep")
            except ImportError:
                iterator = indices
            for episode_idx in iterator:
                if self.abs_action:
                    actions.append(self._get_abs_actions(episode_idx))
                else:
                    df = self._read_parquet(episode_idx)
                    actions.append(
                        np.stack(
                            [
                                self._decode_action_value(value)
                                for value in df[self.action_key].tolist()
                            ],
                            axis=0,
                        )
                    )
            if actions:
                self._all_actions_cache = torch.from_numpy(np.concatenate(actions, axis=0))
            else:
                self._all_actions_cache = torch.empty((0,) + self.action_shape)
        return self._all_actions_cache

    def get_normalizer(self, **kwargs: Any) -> LinearNormalizer:
        normalizer = LinearNormalizer()
        all_actions = self.get_all_actions().cpu().numpy()
        if self.abs_action:
            # Range-normalize position to [-1, 1] and leave rotation_6d /
            # gripper untouched, matching lpb's libero recipe.
            action_stat = array_to_stats(all_actions)
            normalizer["action"] = robomimic_abs_action_only_normalizer_from_stat(
                action_stat
            )
        else:
            normalizer["action"] = SingleFieldLinearNormalizer.create_fit(all_actions)

        for obs_key in self.lowdim_keys:
            values: list[np.ndarray] = []
            column_name = self.obs_columns[obs_key]
            for episode_idx in self._train_episode_indices:
                df = self._read_parquet(episode_idx)
                values.append(
                    np.stack(
                        [
                            self._decode_lowdim_value(value, obs_key)
                            for value in df[column_name].tolist()
                        ],
                        axis=0,
                    )
                )
            if values:
                normalizer[obs_key] = SingleFieldLinearNormalizer.create_fit(
                    np.concatenate(values, axis=0)
                )

        for obs_key in self.rgb_keys:
            normalizer[obs_key] = get_image_range_normalizer()
        return normalizer

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        episode_idx, start = self.samples[idx]
        df = self._read_parquet(episode_idx)
        if self.sample_mode == "video_frame":
            obs_row_indices = [start + offset for offset in range(self.n_obs_steps)]
            action_start = start + self.n_obs_steps - 1
            action_row_indices = [
                action_start + offset for offset in range(self.frame_gap)
            ]
        else:
            action_row_indices = self._build_row_indices(episode_idx, start)
            obs_row_indices = action_row_indices[: self.n_obs_steps]

        obs_dict: dict[str, np.ndarray] = {}
        for obs_key in self.rgb_keys:
            column_name = self.obs_columns[obs_key]
            obs_dict[obs_key] = np.stack(
                [
                    self._decode_rgb_value(df.iloc[row_idx][column_name], obs_key)
                    for row_idx in obs_row_indices
                ],
                axis=0,
            )

        for obs_key in self.lowdim_keys:
            column_name = self.obs_columns[obs_key]
            obs_dict[obs_key] = np.stack(
                [
                    self._decode_lowdim_value(df.iloc[row_idx][column_name], obs_key)
                    for row_idx in obs_row_indices
                ],
                axis=0,
            )

        if self.abs_action:
            abs_actions = self._get_abs_actions(episode_idx)
            ep_len = abs_actions.shape[0]
            clipped = [min(max(idx, 0), ep_len - 1) for idx in action_row_indices]
            action = abs_actions[clipped]
        else:
            action = np.stack(
                [
                    self._decode_action_value(df.iloc[row_idx][self.action_key])
                    for row_idx in action_row_indices
                ],
                axis=0,
            )

        data = {"obs": obs_dict, "action": action.astype(np.float32)}
        return dict_apply(data, torch.from_numpy)
