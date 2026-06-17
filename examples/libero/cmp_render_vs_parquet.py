"""Compare the LIVE env render against the stored parquet frames along a demo.

If the live agentview render diverges from the parquet image the policy was
trained on (beyond t=0), that image domain gap explains why the policy fails
online despite matching everything statically.

Run in the `libero` conda env from repo root.
"""
from __future__ import annotations
import argparse, io, pathlib
import h5py, numpy as np, pandas as pd
from PIL import Image
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

LIBERO_DUMMY = [0.0]*6 + [-1.0]

def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--source-task", default="KITCHEN_SCENE2_put_the_middle_black_bowl_on_the_plate")
    p.add_argument("--hdf5", default="/scr2/zhaoyang/libero_90/KITCHEN_SCENE2_put_the_middle_black_bowl_on_the_plate_demo.hdf5")
    p.add_argument("--parquet", default="/scr2/zhaoyang/libero_wm/data/chunk-000/episode_000700.parquet")
    p.add_argument("--suite", default="libero_90")
    p.add_argument("--resolution", type=int, default=128)
    p.add_argument("--steps", default="0,10,30,60")
    p.add_argument("--out", default="/scr2/zhaoyang/TTT-WM/examples/libero/render_cmp")
    return p.parse_args()

def pq_frame(cell):
    v = cell["bytes"] if isinstance(cell, dict) and "bytes" in cell else cell
    return np.asarray(Image.open(io.BytesIO(v)).convert("RGB"))

def main():
    a = parse()
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    suite = benchmark.get_benchmark_dict()[a.suite]()
    task = next(suite.get_task(i) for i in range(suite.n_tasks)
                if pathlib.Path(suite.get_task(i).bddl_file).stem == a.source_task)
    bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=a.resolution, camera_widths=a.resolution)
    env.seed(0)

    df = pd.read_parquet(a.parquet)
    f = h5py.File(a.hdf5, "r")
    d0 = f["data"][sorted(f["data"].keys(), key=lambda k:int(k.split("_")[1]))[0]]
    actions = np.asarray(d0["actions"]); init = np.asarray(d0["states"][0])

    steps = [int(s) for s in a.steps.split(",")]
    env.reset(); obs = env.set_init_state(init)
    cur = 0
    for tgt in steps:
        while cur < tgt:
            obs, *_ = env.step(actions[cur].tolist()); cur += 1
        live = np.ascontiguousarray(obs["agentview_image"])
        pq = pq_frame(df.iloc[min(tgt, len(df)-1)]["image"])
        # try live as-is and live flipped 180, report whichever matches parquet better
        mse_raw = float(((live.astype(np.float32)-pq)**2).mean())
        mse_rot = float(((np.ascontiguousarray(live[::-1,::-1]).astype(np.float32)-pq)**2).mean())
        print(f"step {tgt:3d}: pixel MSE  live-vs-parquet = {mse_raw:8.1f} | live(rot180)-vs-parquet = {mse_rot:8.1f}")
        # save side by side: [live | parquet | live_rot180]
        strip = np.concatenate([live, pq, np.ascontiguousarray(live[::-1,::-1])], axis=1)
        Image.fromarray(strip).save(out / f"cmp_step{tgt:03d}.png")
    print(f"\nsaved side-by-side strips (live | parquet | live_rot180) to {out}")
    env.close()

if __name__ == "__main__":
    main()
