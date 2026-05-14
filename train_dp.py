from __future__ import annotations

import builtins
import json
import logging
import os
import sys
import traceback
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf
from torch.distributed.elastic.multiprocessing.errors import record

from dp.runtime import configure_diffusion_policy_path, register_omegaconf_resolvers


def _is_main_rank() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


def _silence_non_main_rank_logging() -> None:
    if _is_main_rank():
        return

    def _silent_print(*args, **kwargs) -> None:
        return None

    builtins.print = _silent_print
    logging.disable(logging.WARNING)


_silence_non_main_rank_logging()
register_omegaconf_resolvers()


def _resolve_dataset_root(cfg: DictConfig) -> Path:
    dataset_root = OmegaConf.select(cfg, "task.dataset.dataset_root", default=None)
    if dataset_root in (None, "", "None"):
        dataset_root = OmegaConf.select(cfg, "dataset_root", default=None)
    if dataset_root in (None, "", "None"):
        raise ValueError(
            "Dataset root is missing. Set `dataset_root` or `task.dataset.dataset_root`."
        )
    return Path(str(dataset_root)).expanduser()


def _load_heldout_tasks(dataset_root: Path) -> tuple[Path, list[str]]:
    meta_path = dataset_root / "meta" / "test_tasks.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"Held-out task metadata not found: {meta_path}")

    with meta_path.open() as f:
        payload = json.load(f)

    records = payload.get("records") if isinstance(payload, dict) else None
    if not isinstance(records, list) or not records:
        raise ValueError(
            f"`{meta_path}` must contain a non-empty `records` list."
        )

    heldout_tasks = [
        str(record["task"])
        for record in records
        if isinstance(record, dict) and record.get("task") not in (None, "")
    ]
    if not heldout_tasks:
        raise ValueError(f"No held-out tasks were found in {meta_path}.")

    return meta_path, heldout_tasks


def _resolve_selected_heldout_task(
    selected_task: object,
    heldout_tasks: list[str],
) -> str:
    selector = str(selected_task).strip()
    if selector in ("", "None"):
        raise ValueError("`data.selected_task` must not be empty.")

    if selector in heldout_tasks:
        return selector

    raise ValueError(
        "Configured `data.selected_task` does not match any held-out task in "
        f"meta/test_tasks.json: {selector!r}. Available held-out tasks: {heldout_tasks}"
    )


def _apply_selected_task_overrides(cfg: DictConfig) -> None:
    selected_task = OmegaConf.select(cfg, "data.selected_task", default=None)
    if selected_task in (None, "", "None"):
        return

    dataset_root = _resolve_dataset_root(cfg)
    meta_path, heldout_tasks = _load_heldout_tasks(dataset_root)
    resolved_task = _resolve_selected_heldout_task(selected_task, heldout_tasks)

    cfg.data.selected_task = resolved_task
    if _is_main_rank():
        print(
            "Using held-out task-filtered episode split for diffusion policy training: "
            f"meta={meta_path}, "
            f"data.selected_task={resolved_task!r}, "
            f"data.val_ratio={float(cfg.data.val_ratio):.3f}"
        )


def _log_episode_split(cfg: DictConfig) -> None:
    """Print the episode-level train/val split chosen by the dataset.

    Same convention as Stage 2.2's HeldoutTaskSplitDataset: episodes
    matching `data.selected_task` are deterministically shuffled with
    `seed`, the first `round(N * val_ratio)` go to val, the rest to train.
    Datasets are NOT split at the sample-window level — every window
    belongs whole to one split, eliminating intra-episode leakage.
    """
    if not _is_main_rank():
        return
    try:
        # Force verbose=False so the dataset's own banner stays out of the way;
        # split=train so both _train/_val episode lists are populated identically.
        preview = hydra.utils.instantiate(
            cfg.task.dataset, verbose=False, split="train"
        )
    except Exception as exc:
        print(f"[train_dp] Could not preview episode split: {exc}")
        return

    train_eps = list(preview._train_episode_indices)
    val_eps = list(preview._val_episode_indices)

    print("=" * 78)
    print("[train_dp] Episode-level train/val split (same scheme as Stage 2.2):")
    print(f"  task_filter   : {preview.task_filter!r}")
    print(f"  seed          : {preview.seed}")
    print(f"  val_ratio     : {preview.val_ratio}")
    print(f"  split_reason  : {preview._split_reason}")
    print(f"  train episodes: {len(train_eps)} -> {train_eps}")
    print(f"  val   episodes: {len(val_eps)} -> {val_eps}")
    if len(val_eps) <= 1:
        print(
            "  WARNING: val_ratio is low; only "
            f"{len(val_eps)} episode in the val split — val_loss will be noisy. "
            "Consider raising data.val_ratio."
        )
    print("=" * 78)


@hydra.main(version_base=None, config_path="configs", config_name="dp_config")
def _hydra_main(cfg: DictConfig) -> None:
    diffusion_policy_src = OmegaConf.select(
        cfg, "runtime.diffusion_policy_src", default=None
    )
    configured_paths = configure_diffusion_policy_path(diffusion_policy_src)
    if configured_paths and _is_main_rank():
        print("Configured diffusion_policy search path:")
        for path in configured_paths:
            print(f"  - {path}")

    _apply_selected_task_overrides(cfg)
    OmegaConf.resolve(cfg)

    _log_episode_split(cfg)

    from dp.common import cleanup_distributed
    from dp.train_workspace import TrainDiffusionWorkspace

    try:
        workspace = TrainDiffusionWorkspace(cfg)
        workspace.run()
    except BaseException as exc:
        rank = os.environ.get("RANK", "0")
        sys.stderr.write(
            f"[rank{rank}] Unhandled exception in train_dp.py: "
            f"{type(exc).__name__}: {exc}\n"
        )
        traceback.print_exc(file=sys.stderr)
        sys.stderr.flush()
        raise
    finally:
        cleanup_distributed()


@record
def main() -> None:
    _hydra_main()


if __name__ == "__main__":
    main()
