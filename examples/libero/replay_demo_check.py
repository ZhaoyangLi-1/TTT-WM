"""Open-loop GT-action replay sanity check.

Replays the recorded HDF5 demo actions for a libero_90 task in the SAME
OffScreenRenderEnv the online rollout uses, starting from the demo's own t=0
sim state. If the task is NOT solved by replaying the ground-truth actions, the
problem is in the env/controller execution (not the policy). If it IS solved,
the policy/closed-loop is at fault.

Run in the `libero` conda env, from the repo root:

    python examples/libero/replay_demo_check.py \
        --task-suite-name libero_90 \
        --source-task KITCHEN_SCENE2_put_the_middle_black_bowl_on_the_plate \
        --hdf5 /scr2/zhaoyang/libero_90/KITCHEN_SCENE2_put_the_middle_black_bowl_on_the_plate_demo.hdf5 \
        --num-demos 5 --num-steps-wait 10
"""
from __future__ import annotations

import argparse
import pathlib

import h5py
import numpy as np
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--task-suite-name", default="libero_90")
    p.add_argument("--source-task", required=True, help="bddl stem")
    p.add_argument("--hdf5", required=True)
    p.add_argument("--num-demos", type=int, default=5)
    p.add_argument("--num-steps-wait", type=int, default=10)
    p.add_argument("--resolution", type=int, default=128)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def get_env(task, resolution, seed):
    bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=resolution, camera_widths=resolution)
    env.seed(seed)
    return env


def main():
    args = parse_args()
    suite = benchmark.get_benchmark_dict()[args.task_suite_name]()
    # find task whose bddl stem matches
    task = None
    for i in range(suite.n_tasks):
        t = suite.get_task(i)
        if pathlib.Path(t.bddl_file).stem == args.source_task:
            task, task_id = t, i
            break
    if task is None:
        raise SystemExit(f"source_task {args.source_task} not found in {args.task_suite_name}")
    print(f"task_id={task_id}  desc={task.language}")

    env = get_env(task, args.resolution, args.seed)
    f = h5py.File(args.hdf5, "r")
    keys = sorted(f["data"].keys(), key=lambda k: int(k.split("_")[1]))

    n_ok = 0
    n = min(args.num_demos, len(keys))
    for j in range(n):
        d = f["data"][keys[j]]
        init_state = np.asarray(d["states"][0])
        actions = np.asarray(d["actions"])
        env.reset()
        env.set_init_state(init_state)
        for _ in range(args.num_steps_wait):
            env.step(LIBERO_DUMMY_ACTION)
        done = False
        for a in actions:
            obs, reward, done, info = env.step(a.tolist())
            if done:
                break
        n_ok += int(bool(done))
        print(f"  demo_{j}: replay {len(actions)} GT actions -> success={bool(done)}")
    print(f"\nGT-replay success: {n_ok}/{n}")
    env.close()


if __name__ == "__main__":
    main()
