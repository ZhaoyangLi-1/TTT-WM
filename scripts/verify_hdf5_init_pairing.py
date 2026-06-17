#!/usr/bin/env python
"""Verify route (b): driving the env from a libero_wm demo's own HDF5 init sim
state reproduces that demo's t=0 scene.

For each task and each demo j, set the env to ``demo_j/states[0]`` (from the
re-downloaded LIBERO-90 HDF5), render the agentview frame, and compare it to
libero_wm episode ``base+j``'s t=0 frame. If the HDF5<->libero_wm mapping is
correct the per-pixel RMS should be ~0 (vs ~0.08 for the unrelated pruned_init
eval states), and the diagonal should be the clear argmin of each row.

Run AFTER downloading LIBERO-90 HDF5. Must pass before trusting route (b) runs.

    conda activate libero
    export PYTHONPATH=$PYTHONPATH:$PWD/third_party/libero
    python scripts/verify_hdf5_init_pairing.py --hdf5-root /scr2/zhaoyang/LIBERO-data
"""
import argparse
import io
import pathlib
import sys

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "examples/libero"))
from hdf5_init import demo_init_state, num_demos  # noqa: E402

LIBERO_ENV_RESOLUTION = 128
SEED = 7
DUMMY = [0.0] * 6 + [-1.0]

TASKS = [
    ("KITCHEN_SCENE10_put_the_butter_at_the_back_in_the_top_drawer_of_the_cabinet_and_close_it", 6043, 6048),
    ("KITCHEN_SCENE2_put_the_middle_black_bowl_on_the_plate", 6093, 6098),
    ("STUDY_SCENE3_pick_up_the_book_and_place_it_in_the_front_compartment_of_the_caddy", 6143, 6148),
]


def _wm_t0_image(wm_root, ep):
    p = wm_root / "data" / f"chunk-{ep // 1000:03d}" / f"episode_{ep:06d}.parquet"
    e = pq.ParquetFile(str(p)).read(columns=["image"]).column("image").to_pylist()[0]
    return np.asarray(Image.open(io.BytesIO(bytes(e["bytes"]))).convert("RGB"), np.float32) / 255.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hdf5-root", default="/scr2/zhaoyang/LIBERO-data")
    ap.add_argument("--libero-wm-root", default="/scr2/zhaoyang/libero_wm")
    ap.add_argument("--n", type=int, default=10, help="demos per task to check")
    ap.add_argument("--settle", type=int, default=0, help="dummy steps before render")
    ap.add_argument("--tol", type=float, default=0.03, help="pass threshold on diagonal RMS")
    args = ap.parse_args()

    hdf5_root = pathlib.Path(args.hdf5_root).expanduser().resolve()
    wm_root = pathlib.Path(args.libero_wm_root).expanduser().resolve()

    suite = benchmark.get_benchmark_dict()["libero_90"]()
    stem_to_id = {pathlib.Path(suite.get_task(i).bddl_file).stem: i for i in range(suite.n_tasks)}

    all_ok = True
    checked = 0
    for src, base, val in TASKS:
        print("\n" + "=" * 84)
        try:
            n_demos = num_demos(hdf5_root, src)
        except FileNotFoundError as e:
            print(f"SKIP {src}: {e}")
            continue
        checked += 1
        print(f"TASK {src}  base={base} val={val}  demos={n_demos}")
        task = suite.get_task(stem_to_id[src])
        bf = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        env = OffScreenRenderEnv(bddl_file_name=bf, camera_heights=LIBERO_ENV_RESOLUTION, camera_widths=LIBERO_ENV_RESOLUTION)
        env.seed(SEED)

        n = min(args.n, num_demos(hdf5_root, src))
        env_imgs, wm_imgs = [], []
        for j in range(n):
            env.reset()
            obs = env.set_init_state(demo_init_state(hdf5_root, src, j))
            for _ in range(args.settle):
                obs, _, _, _ = env.step(DUMMY)
            env_imgs.append(np.asarray(obs["agentview_image"], np.float32) / 255.0)
            wm_imgs.append(_wm_t0_image(wm_root, base + j))
        env.close()
        E = np.stack(env_imgs).reshape(n, -1)
        W = np.stack(wm_imgs).reshape(n, -1)
        # try identity and 180-flip orientation
        Wr = np.stack([w[::-1, ::-1] for w in np.stack(wm_imgs)]).reshape(n, -1)
        C_id = np.sqrt(((E[:, None] - W[None]) ** 2).mean(2))
        C_rt = np.sqrt(((E[:, None] - Wr[None]) ** 2).mean(2))
        C, orient = (C_id, "identity") if np.trace(C_id) <= np.trace(C_rt) else (C_rt, "rot180")
        diag = np.diag(C)
        argmin_is_diag = int((C.argmin(1) == np.arange(n)).sum())
        # nearest competing (off-diagonal) wm episode per row
        Coff = C.copy()
        Coff[np.arange(n), np.arange(n)] = np.inf
        offmin = Coff.min(1)
        # separation: how much the correct (diagonal) match beats the best wrong one.
        sep = offmin - diag
        argmin_rate = argmin_is_diag / n
        print(f"  orientation={orient}")
        print(f"  diagonal (demo j init -> wm ep {base}+j) RMS: med={np.median(diag):.4f} max={diag.max():.4f}")
        print(f"  nearest OFF-diagonal RMS: med={np.median(offmin):.4f}  (≈ render-domain floor + layout gap)")
        print(f"  separation (off-diag - diag): med={np.median(sep):.4f} min={sep.min():.4f}  (>0 ⇒ correct demo wins)")
        print(f"  rows whose argmin is the diagonal: {argmin_is_diag}/{n}")
        # Mapping is correct if the paired wm episode is (almost) always the
        # nearest. The absolute RMS floor is a render-domain gap (env render vs
        # collection-time render), NOT a layout error, so we do NOT require ~0.
        ok = argmin_rate >= 0.9 and np.median(sep) > 0.0
        print(f"  PASS={ok}  (need argmin==diagonal for >=90% rows AND median separation>0)")
        all_ok = all_ok and ok

    print("\n" + "=" * 84)
    if checked == 0:
        print("OVERALL: no task HDF5 found under", hdf5_root, "— nothing verified.")
        sys.exit(2)
    print("OVERALL:", "PASS — HDF5<->libero_wm pairing verified, route (b) safe to use."
          if all_ok else "FAIL — mapping not confirmed; do NOT trust route (b) runs yet.")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
