"""Helpers for route (b): drive the online LIBERO env from a libero_wm demo's
OWN initial sim state (read from the original LIBERO-90 HDF5), so the rollout
starts exactly where the demo started and that demo's terminal frame is a
perfectly consistent goal.

Mapping (verified): libero_wm stores each task's 50 demos contiguously in
sorted ``demo_<n>`` order (scripts/prepare_libero90_hdf5.py:207). So for a task
whose libero_wm episodes are ``[base .. base+49]``, libero_wm ``episode (base+j)``
== HDF5 ``data/demo_j``. The full 77-dim init sim state is ``demo_j/states[0]``
and ``env.set_init_state(states[0])`` reproduces the demo's t=0 scene exactly.
"""
import functools
import pathlib
from typing import List

import h5py
import numpy as np


@functools.lru_cache(maxsize=128)
def hdf5_path_for_task(hdf5_root_str: str, source_task: str) -> pathlib.Path:
    """Locate ``<source_task>_demo.hdf5`` anywhere under ``hdf5_root``.

    ``source_task`` is the bddl stem (e.g. KITCHEN_SCENE10_put_the_butter...),
    which is exactly how prepare_libero90_hdf5.py named the inputs. We search
    common layouts first (``libero_90/``, ``libero_100/``, root) then fall back
    to a recursive glob, so it doesn't matter whether the HF/original download
    nested the file under libero_90/ or libero_100/.
    """
    hdf5_root = pathlib.Path(hdf5_root_str)
    fname = f"{source_task}_demo.hdf5"
    for sub in ("libero_90", "libero_100", "."):
        cand = hdf5_root / sub / fname
        if cand.is_file():
            return cand
    matches = list(hdf5_root.rglob(fname))
    if matches:
        return matches[0]
    raise FileNotFoundError(
        f"'{fname}' not found anywhere under {hdf5_root}. Download it, e.g.:\n"
        "  cd third_party/libero && python benchmark_scripts/download_libero_datasets.py "
        "--datasets libero_100 --use-huggingface --download-dir " + str(hdf5_root)
    )


def _sorted_demo_keys(f: h5py.File) -> List[str]:
    # identical ordering to prepare_libero90_hdf5.py:sorted_demo_keys
    return sorted(f["data"].keys(), key=lambda k: int(k.split("_")[-1]))


@functools.lru_cache(maxsize=8)
def _open_demo_keys(hdf5_file: str) -> tuple:
    with h5py.File(hdf5_file, "r") as f:
        return tuple(_sorted_demo_keys(f))


def demo_init_state(hdf5_root: pathlib.Path, source_task: str, demo_idx: int) -> np.ndarray:
    """Return the 77-dim t=0 sim state of the ``demo_idx``-th (sorted) demo."""
    hp = hdf5_path_for_task(str(hdf5_root), source_task)
    keys = _open_demo_keys(str(hp))
    if demo_idx < 0 or demo_idx >= len(keys):
        raise IndexError(
            f"demo_idx={demo_idx} out of range; {hp.name} has {len(keys)} demos."
        )
    with h5py.File(str(hp), "r") as f:
        states = f["data"][keys[demo_idx]]["states"]
        return np.asarray(states[0])


def num_demos(hdf5_root: pathlib.Path, source_task: str) -> int:
    return len(_open_demo_keys(str(hdf5_path_for_task(str(hdf5_root), source_task))))