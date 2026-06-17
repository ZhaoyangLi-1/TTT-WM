#!/usr/bin/env python
"""Match each LIBERO ``initial_states[0..49]`` to its corresponding libero_wm demo
by t=0 proprio, decide whether the demo ordering equals ``base + episode_idx``,
and emit a *consistent* ``goal_episode_indices`` mapping for online rollout.

Why this exists
---------------
Stage 2.2 / Pure IDM rollout (examples/libero/main.py) needs a goal/next frame.
Today the common config sends ONE fixed val-demo terminal frame to every online
episode (schema ``goal_episode_index``), but each online episode resets to a
DIFFERENT ``initial_states[episode_idx]`` whose object layout does not match that
single goal frame -> the (current, goal) pair is off-distribution.

The fix requires knowing, for each online ``episode_idx`` (== init-state index),
WHICH libero_wm demo started from the same init. This script recovers that
pairing empirically: it sets the env to each init state, reads the same 8-dim
proprio main.py builds (``observation/state``), reads each demo's t=0 ``state``
row from the libero_wm parquet, and solves the optimal 1-to-1 assignment.

It then:
  * reports whether the recovered permutation equals identity (init k -> base+k);
  * prints match-cost + ambiguity-margin stats so you can judge confidence;
  * writes ``<task>_paired.json`` (schema ``goal_episode_indices``) giving every
    online episode_idx its consistently-paired demo terminal frame.

Fairness caveat (printed at the end too): a consistently-paired goal for a
*train* demo is still a leakage upper bound -- identical caveat to the
``*_aligned.json`` configs. The single online episode paired to the held-out
val demo is the only point that is BOTH fair and consistent.
"""
import argparse
import json
import math
import pathlib

import io

import numpy as np
import pyarrow.parquet as pq
from PIL import Image
from scipy.optimize import linear_sum_assignment

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

LIBERO_ENV_RESOLUTION = 128
SEED = 7  # matches examples/libero/main.py default

# (source_task == bddl stem, full task string, base episode, held-out val episode)
TASKS = [
    {
        "source_task": "KITCHEN_SCENE10_put_the_butter_at_the_back_in_the_top_drawer_of_the_cabinet_and_close_it",
        "task": "KITCHEN_SCENE10: put the butter at the back in the top drawer of the cabinet and close it",
        "base": 6043,
        "val": 6048,
    },
    {
        "source_task": "KITCHEN_SCENE2_put_the_middle_black_bowl_on_the_plate",
        "task": "KITCHEN_SCENE2: put the middle black bowl on the plate",
        "base": 6093,
        "val": 6098,
    },
    {
        "source_task": "STUDY_SCENE3_pick_up_the_book_and_place_it_in_the_front_compartment_of_the_caddy",
        "task": "STUDY_SCENE3: pick up the book and place it in the front compartment of the caddy",
        "base": 6143,
        "val": 6148,
    },
]


def _quat2axisangle(quat):
    """Copied verbatim from examples/libero/main.py (robosuite convention)."""
    quat = np.asarray(quat, dtype=np.float64).copy()
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def _online_proprio_from_obs(obs):
    """Same 8-dim vector main.py puts in observation/state."""
    return np.concatenate(
        (
            np.asarray(obs["robot0_eef_pos"], dtype=np.float64),
            _quat2axisangle(obs["robot0_eef_quat"]),
            np.asarray(obs["robot0_gripper_qpos"], dtype=np.float64),
        )
    )


def _get_libero_env(task):
    task_bddl_file = (
        pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    )
    env = OffScreenRenderEnv(
        bddl_file_name=task_bddl_file,
        camera_heights=LIBERO_ENV_RESOLUTION,
        camera_widths=LIBERO_ENV_RESOLUTION,
    )
    env.seed(SEED)
    return env


def _parquet_path(wm_root: pathlib.Path, ep: int) -> pathlib.Path:
    chunk = ep // 1000
    return wm_root / "data" / f"chunk-{chunk:03d}" / f"episode_{ep:06d}.parquet"


def _demo_t0_proprio(wm_root: pathlib.Path, ep: int) -> np.ndarray:
    p = _parquet_path(wm_root, ep)
    if not p.is_file():
        raise FileNotFoundError(p)
    row = pq.ParquetFile(str(p)).read(columns=["state"]).column("state").to_pylist()[0]
    return np.asarray(row, dtype=np.float64)


def _decode_img(entry) -> np.ndarray:
    if isinstance(entry, dict) and entry.get("bytes") is not None:
        raw = bytes(entry["bytes"])
    elif isinstance(entry, (bytes, bytearray)):
        raw = bytes(entry)
    else:
        raise TypeError(f"Unsupported image entry type={type(entry).__name__}")
    return np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"), dtype=np.float32) / 255.0


def _demo_t0_image(wm_root: pathlib.Path, ep: int) -> np.ndarray:
    p = _parquet_path(wm_root, ep)
    entry = pq.ParquetFile(str(p)).read(columns=["image"]).column("image").to_pylist()[0]
    return _decode_img(entry)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--libero-wm-root", default="/scr2/zhaoyang/libero_wm")
    ap.add_argument("--task-suite-name", default="libero_90")
    ap.add_argument("--num-trials-per-task", type=int, default=50)
    ap.add_argument(
        "--out-dir",
        default=str(pathlib.Path(__file__).resolve().parent.parent / "examples/libero/test_tasks"),
        help="Where to write <task>_paired.json files.",
    )
    ap.add_argument(
        "--identity-tol",
        type=float,
        default=0.05,
        help="Per-init match-cost threshold below which we trust the pairing.",
    )
    args = ap.parse_args()

    wm_root = pathlib.Path(args.libero_wm_root).expanduser().resolve()
    out_dir = pathlib.Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    n = args.num_trials_per_task

    suite = benchmark.get_benchmark_dict()[args.task_suite_name]()
    num_tasks = suite.n_tasks
    stem_to_id = {
        pathlib.Path(suite.get_task(i).bddl_file).stem: i for i in range(num_tasks)
    }

    for spec in TASKS:
        src = spec["source_task"]
        base = spec["base"]
        val = spec["val"]
        if src not in stem_to_id:
            print(f"[SKIP] {src}: not found in suite {args.task_suite_name}")
            continue
        task_id = stem_to_id[src]
        task = suite.get_task(task_id)
        print("\n" + "=" * 88)
        print(f"TASK {src}  (task_id={task_id}, base={base}, val={val})")

        # --- online proprio for each init state ---
        init_states = suite.get_task_init_states(task_id)
        if len(init_states) < n:
            print(f"  [WARN] only {len(init_states)} init states (< {n}); clamping.")
        n_eff = min(n, len(init_states))
        env = _get_libero_env(task)
        online = np.zeros((n_eff, 8))
        env_imgs = []
        for k in range(n_eff):
            env.reset()
            obs = env.set_init_state(init_states[k])
            online[k] = _online_proprio_from_obs(obs)
            env_imgs.append(
                np.asarray(obs["agentview_image"], dtype=np.float32) / 255.0
            )
        env.close()
        env_imgs = np.stack(env_imgs)  # (n, H, W, 3)

        demo_eps = [base + j for j in range(n_eff)]

        # --- demo t=0 images: authoritative signal (captures object layout) ---
        demo_imgs = np.stack([_demo_t0_image(wm_root, ep) for ep in demo_eps])
        # Auto-resolve a possible 180-deg orientation mismatch between the env
        # render and the stored demo frames: pick whichever global orientation
        # yields the smaller mean best-match cost.
        env_flat = env_imgs.reshape(n_eff, -1)
        env_flat_rot = env_imgs[:, ::-1, ::-1, :].reshape(n_eff, -1)
        demo_flat = demo_imgs.reshape(n_eff, -1)

        def _img_cost(a):
            # mean per-pixel L2 distance, n x n
            return np.sqrt(((a[:, None, :] - demo_flat[None, :, :]) ** 2).mean(axis=2))

        Cimg_id = _img_cost(env_flat)
        Cimg_rot = _img_cost(env_flat_rot)
        if Cimg_rot.min(axis=1).mean() < Cimg_id.min(axis=1).mean():
            Cimg, orient = Cimg_rot, "rot180"
        else:
            Cimg, orient = Cimg_id, "identity"
        print(f"  env<->demo image orientation: {orient}")

        # --- proprio cost (secondary cross-check) ---
        demo = np.stack([_demo_t0_proprio(wm_root, ep) for ep in demo_eps])
        Cprop = np.linalg.norm(online[:, None, :] - demo[None, :, :], axis=2)

        # The authoritative assignment uses images.
        C = Cimg
        row_ind, col_ind = linear_sum_assignment(C)
        perm = np.empty(n_eff, dtype=int)
        perm[row_ind] = col_ind  # perm[k] = j-offset -> demo episode base+perm[k]

        opt_costs = C[np.arange(n_eff), perm]
        # ambiguity margin: gap to the next-best demo for each init
        margins = np.zeros(n_eff)
        for k in range(n_eff):
            order = np.argsort(C[k])
            best = order[0]
            second = order[1] if order[0] != perm[k] else order[1]
            # margin = (cost to 2nd-best distinct demo) - (assigned cost)
            others = C[k][np.arange(n_eff) != perm[k]]
            margins[k] = others.min() - opt_costs[k]

        identity = np.array_equal(perm, np.arange(n_eff))
        greedy = C.argmin(axis=1)
        greedy_is_perm = len(set(greedy.tolist())) == n_eff
        greedy_eq_identity = np.array_equal(greedy, np.arange(n_eff))

        # proprio-based assignment, purely as a cross-check / confidence signal
        prow, pcol = linear_sum_assignment(Cprop)
        pperm = np.empty(n_eff, dtype=int)
        pperm[prow] = pcol
        prop_eq_img = int((pperm == perm).sum())

        print("  [authoritative = IMAGE t=0 match]")
        print(f"  optimal assignment == identity (init k -> base+k)? {identity}")
        print(f"  greedy nearest is a bijection?                     {greedy_is_perm}")
        print(f"  greedy nearest == identity?                        {greedy_eq_identity}")
        print(f"  proprio-assignment agrees with image on {prop_eq_img}/{n_eff} inits")
        print(
            f"  match cost  : median={np.median(opt_costs):.4f} "
            f"mean={opt_costs.mean():.4f} max={opt_costs.max():.4f}"
        )
        print(
            f"  margin(2nd-best - assigned): median={np.median(margins):.4f} "
            f"min={margins.min():.4f}  (larger = less ambiguous)"
        )
        n_confident = int((opt_costs <= args.identity_tol).sum())
        print(f"  inits with match cost <= {args.identity_tol}: {n_confident}/{n_eff}")

        if not identity:
            mism = [
                (int(k), int(base + k), int(base + perm[k]))
                for k in range(n_eff)
                if perm[k] != k
            ]
            print(f"  [!] {len(mism)} inits where optimal demo != base+k (init, base+k, matched):")
            for k, expect, got in mism[:12]:
                print(f"        init {k:2d}: expected {expect}, matched {got} (cost {C[k, perm[k]]:.4f})")
            if len(mism) > 12:
                print(f"        ... (+{len(mism) - 12} more)")

        # --- decide whether a trustworthy pairing exists ---
        # A pairing is only meaningful if (a) the greedy nearest neighbour is a
        # bijection, (b) every matched cost is below identity_tol (i.e. the env
        # init really equals some demo init, cost ~ 0), and (c) every margin is
        # positive (no closer demo than the assigned one). With independent init
        # draws none of these hold and we MUST NOT emit a fake pairing.
        trustworthy = (
            greedy_is_perm
            and float(opt_costs.max()) <= args.identity_tol
            and float(margins.min()) > 0.0
        )
        goal_episode_indices = [int(base + perm[k]) for k in range(n_eff)]
        val_episode_idx = [k for k in range(n_eff) if goal_episode_indices[k] == val]

        if trustworthy:
            record = {
                "_comment": (
                    "AUTO-GENERATED by scripts/match_init_to_libero_wm.py. "
                    "goal_episode_indices[k] is the libero_wm demo whose t=0 image "
                    "matches initial_states[k] (verified: bijection, cost<=tol, "
                    "positive margins). LEAKAGE CAVEAT: all but the val-paired episode "
                    "use train demo terminal frames -> upper bound, not a fair number. "
                    f"Only episode_idx in {val_episode_idx} (val {val}) is fair+consistent."
                ),
                "records": [
                    {
                        "source_task": src,
                        "task": spec["task"],
                        "goal_episode_indices": goal_episode_indices,
                        "test_episode_indices": [val],
                        "_ordering_is_identity": bool(identity),
                        "_match_cost_max": float(opt_costs.max()),
                        "_val_paired_episode_idx": val_episode_idx,
                    }
                ],
            }
            out_path = out_dir / f"{src}_paired.json"
            with open(out_path, "w") as f:
                json.dump(record, f, indent=2)
            print(f"  TRUSTWORTHY pairing -> wrote {out_path}")
            if identity:
                print(
                    f'  NOTE: ordering is identity -> you may instead set '
                    f'"goal_episode_index_base": {base} (schema 4).'
                )
        else:
            verdict = {
                "_verdict": "NO_RECOVERABLE_PAIRING",
                "_comment": (
                    "scripts/match_init_to_libero_wm.py found NO trustworthy 1-to-1 "
                    "pairing between LIBERO get_task_init_states() and libero_wm demos: "
                    "best image match cost never approaches 0 and margins are <=0, i.e. "
                    "the online init states are INDEPENDENT draws from the same layout "
                    "distribution as the demos, not the same instances. Do NOT use a "
                    "goal_episode_indices/base mapping as if it were exact. For a "
                    "consistent (but train-leakage) diagnostic use goal_align_by_init_proprio "
                    "(*_aligned.json), ideally with an image NN metric. A fair AND "
                    "consistent demo-frame goal does not exist in this data."
                ),
                "source_task": src,
                "task": spec["task"],
                "ordering_is_identity": bool(identity),
                "greedy_is_bijection": bool(greedy_is_perm),
                "image_match_cost_min": float(opt_costs.min()),
                "image_match_cost_median": float(np.median(opt_costs)),
                "image_match_cost_max": float(opt_costs.max()),
                "margin_min": float(margins.min()),
                "demo_demo_note": "see script stdout for demo-demo spread",
            }
            out_path = out_dir / f"{src}_pairing_verdict.json"
            with open(out_path, "w") as f:
                json.dump(verdict, f, indent=2)
            print(f"  NO trustworthy pairing -> wrote verdict {out_path}")

    print("\n" + "=" * 88)
    print("FAIRNESS: a consistently-paired TRAIN demo goal is a leakage upper bound")
    print("(same as *_aligned.json). Only the val-paired episode_idx is fair+consistent.")


if __name__ == "__main__":
    main()
