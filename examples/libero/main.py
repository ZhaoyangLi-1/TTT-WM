import collections
import dataclasses
import json
import logging
import math
import pathlib
from datetime import datetime
from typing import Optional

import imageio
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import numpy as np
from PIL import Image
from client import image_tools
from client import websocket_client_policy as _websocket_client_policy
import tqdm
import tyro

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 128  # resolution used to render training data


@dataclasses.dataclass
class Args:
    #################################################################################################################
    # Model server parameters
    #################################################################################################################
    host: str = "0.0.0.0"
    port: int = 8000
    resize_size: Optional[int] = None
    replan_steps: int = 5
    rotate_images_180: bool = False

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = (
        "libero_spatial"  # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    )
    num_steps_wait: int = 10  # Number of steps to wait for objects to stabilize i n sim
    num_trials_per_task: int = 50  # Number of rollouts per task

    #################################################################################################################
    # Utils
    #################################################################################################################
    video_out_path: str = "data/libero/videos"  # Path to save videos
    save_videos: bool = True
    save_preprocessed_video: bool = False
    summary_out_path: Optional[str] = "data/libero/results"
    run_name: Optional[str] = None

    # Path to a JSON file mapping each task to a goal frame image path.
    # Used by Stage 2.2 (goal-conditioned IDM) and Pure IDM (goal acts as next_image).
    # Supported schemas:
    #   {"<task_description>": "<path>", "<task_id>": "<path>", ...}
    #   [{"task_id": int, "task_description": str, "image_path": str}, ...]
    # Relative paths are resolved against the JSON file's parent directory.
    goal_frames_json: Optional[str] = None

    # Path to libero_wm's meta/test_tasks.json. When provided, the rollout loop
    # only evaluates the tasks listed in this file (matched against the LIBERO
    # benchmark by `source_task` == bddl filename without .bddl). Goal frames
    # for those tasks are extracted from libero_wm parquet data (see
    # libero_wm_root). When set, --goal-frames-json is ignored.
    test_tasks_json: Optional[str] = None

    # Root of the libero_wm dataset (containing `meta/` and `data/chunk-XXX/`).
    # Required when test_tasks_json is set so we can read episodes.jsonl and
    # the corresponding parquet file to extract the goal frame.
    libero_wm_root: Optional[str] = None

    # Directory under which extracted goal frame PNGs are cached. Defaults to
    # `<libero_wm_root>/derived_goal_frames/` when test_tasks_json is in use.
    goal_frames_cache_dir: Optional[str] = None

    seed: int = 7  # Random Seed (for reproducibility)


def eval_libero(args: Args) -> None:
    # Set random seed
    np.random.seed(args.seed)

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    logging.info(f"Task suite: {args.task_suite_name}")

    run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    video_root = pathlib.Path(args.video_out_path) / run_name
    if args.save_videos:
        video_root.mkdir(parents=True, exist_ok=True)

    summary_root = pathlib.Path(args.summary_out_path) / run_name if args.summary_out_path is not None else None
    if summary_root is not None:
        summary_root.mkdir(parents=True, exist_ok=True)

    if args.task_suite_name == "libero_spatial":
        max_steps = 220  # longest training demo has 193 steps
    elif args.task_suite_name == "libero_object":
        max_steps = 280  # longest training demo has 254 steps
    elif args.task_suite_name == "libero_goal":
        max_steps = 300  # longest training demo has 270 steps
    elif args.task_suite_name == "libero_10":
        max_steps = 520  # longest training demo has 505 steps
    elif args.task_suite_name == "libero_90":
        max_steps = 400  # longest training demo has 373 steps
    else:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}")

    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    server_metadata = client.get_server_metadata()
    logging.info(f"Connected to policy server metadata: {server_metadata}")

    model_resize_size = server_metadata.get("input_resolution", None)
    resize_size = int(model_resize_size) if args.resize_size is None and model_resize_size is not None else (
        int(args.resize_size) if args.resize_size is not None else 128
    )

    action_horizon = server_metadata.get("action_horizon", None)
    if action_horizon is not None and int(action_horizon) < args.replan_steps:
        raise ValueError(
            f"replan_steps={args.replan_steps} exceeds the server action horizon={action_horizon}."
        )

    requires_next_image = bool(server_metadata.get("requires_next_image", False))
    goal_conditioned = bool(server_metadata.get("goal_conditioned", False))

    # Decide which observation key (if any) to use for shipping the per-task goal frame.
    if requires_next_image:
        goal_send_key = "observation/next_image"
    elif goal_conditioned:
        goal_send_key = "observation/goal_image"
    else:
        goal_send_key = None

    # Resolve the test_tasks filter (preferred) and/or fallback goal_frames_json.
    # `--libero-wm-root` is only required when the connected policy actually
    # consumes a goal frame (e.g., Stage 2.2 / Pure IDM). For DP policies served
    # via train_dp.py, goal_send_key is None and we just use the JSON to filter
    # which tasks to roll out.
    test_task_filter: Optional[dict] = None  # source_task -> record (with full "task" string)
    wm_root: Optional[pathlib.Path] = None
    cache_dir: Optional[pathlib.Path] = None
    if args.test_tasks_json:
        test_task_filter = _load_test_tasks_filter(args.test_tasks_json)
        if goal_send_key is not None:
            if not args.libero_wm_root:
                raise ValueError(
                    "The connected policy expects a goal frame, so --libero-wm-root is "
                    "required to extract goal frames from the libero_wm parquet data."
                )
            wm_root = pathlib.Path(args.libero_wm_root).expanduser().resolve()
            if not (wm_root / "meta" / "episodes.jsonl").is_file():
                raise FileNotFoundError(f"libero_wm episodes.jsonl not found under {wm_root}")
            cache_dir = (
                pathlib.Path(args.goal_frames_cache_dir).expanduser().resolve()
                if args.goal_frames_cache_dir
                else wm_root / "derived_goal_frames"
            )
        else:
            logging.info(
                "Connected policy does not require a goal frame; "
                "--test-tasks-json is being used purely as a task filter."
            )
            if args.libero_wm_root:
                logging.info(
                    "--libero-wm-root is ignored because the policy does not consume goal frames."
                )
        if args.goal_frames_json:
            logging.warning("--goal-frames-json is ignored because --test-tasks-json is set.")

    goal_index = (
        _load_goal_frames_index(args.goal_frames_json)
        if args.goal_frames_json and test_task_filter is None
        else {}
    )
    if goal_send_key is not None and test_task_filter is None and not goal_index:
        if requires_next_image:
            raise ValueError(
                "The connected policy is non-causal and requires a `next_image`. "
                "Provide --goal-frames-json (or --test-tasks-json + --libero-wm-root) so the "
                "goal frame can be used as next_image."
            )
        logging.warning(
            "Server reports goal_conditioned=True but no goal source was provided; "
            "inference will run with goal=None."
        )

    if server_metadata.get("causal") is False and goal_send_key != "observation/next_image":
        # Non-causal model with no goal-as-next-image route — refuse, the loop cannot proceed.
        raise ValueError(
            "The connected policy server is non-causal and cannot drive the online LIBERO loop "
            "without a goal frame to use as next_image."
        )

    # Start evaluation
    total_episodes, total_successes = 0, 0
    episode_records = []
    task_summaries = []
    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
        # Get task
        task = task_suite.get_task(task_id)

        # When --test-tasks-json is set, restrict rollouts to listed tasks only.
        # source_task in the JSON matches the bddl filename without the extension.
        bddl_stem = pathlib.Path(task.bddl_file).stem
        test_record: Optional[dict] = None
        if test_task_filter is not None:
            test_record = test_task_filter.get(bddl_stem)
            if test_record is None:
                continue

        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)

        # Initialize LIBERO environment and task description
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

        # Resolve and pre-process this task's goal frame.
        task_goal_image: Optional[np.ndarray] = None
        if test_record is not None and goal_send_key is not None:
            assert wm_root is not None and cache_dir is not None
            goal_path = _ensure_goal_frame_from_libero_wm(
                wm_root=wm_root,
                cache_dir=cache_dir,
                source_task=test_record["source_task"],
                full_task=test_record["task"],
            )
            task_goal_image = _load_and_preprocess_image(goal_path, resize_size, args.rotate_images_180)
            logging.info(
                f"Loaded goal frame for task_id={task_id} ({bddl_stem}) from {goal_path}"
            )
        elif goal_send_key is not None and goal_index:
            goal_path = _lookup_goal_path(goal_index, task_id, task_description)
            if goal_path is None:
                msg = (
                    f"No goal frame found for task_id={task_id} / "
                    f"description={task_description!r} in {args.goal_frames_json}."
                )
                if requires_next_image:
                    raise FileNotFoundError(msg)
                logging.warning(msg + " Falling back to goal=None for this task.")
            else:
                task_goal_image = _load_and_preprocess_image(goal_path, resize_size, args.rotate_images_180)
                logging.info(f"Loaded goal frame for task {task_id} from {goal_path}")

        # Start episodes
        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm.tqdm(range(args.num_trials_per_task)):
            logging.info(f"\nTask: {task_description}")

            # Reset environment
            env.reset()
            action_plan = collections.deque()

            # Set initial states
            obs = env.set_init_state(initial_states[episode_idx])

            # Setup
            t = 0
            done = False
            reward = 0.0
            replay_images = []
            replay_input_images = []
            episode_policy_started = False

            logging.info(f"Starting episode {task_episodes+1}...")
            while t < max_steps + args.num_steps_wait:
                try:
                    replay_images.append(np.ascontiguousarray(obs["agentview_image"]))

                    # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                    # and we need to wait for them to fall
                    if t < args.num_steps_wait:
                        obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                        t += 1
                        continue

                    # Get preprocessed image
                    img = np.ascontiguousarray(obs["agentview_image"])
                    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"])
                    if args.rotate_images_180:
                        img = np.ascontiguousarray(img[::-1, ::-1])
                        wrist_img = np.ascontiguousarray(wrist_img[::-1, ::-1])
                    img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(img, resize_size, resize_size)
                    )
                    wrist_img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(wrist_img, resize_size, resize_size)
                    )

                    # Save preprocessed image for optional debug replay video
                    replay_input_images.append(img)

                    if not action_plan:
                        # Finished executing previous action chunk -- compute new chunk
                        # Prepare observations dict
                        element = {
                            "observation/image": img,
                            "observation/wrist_image": wrist_img,
                            "observation/state": np.concatenate(
                                (
                                    obs["robot0_eef_pos"],
                                    _quat2axisangle(obs["robot0_eef_quat"]),
                                    obs["robot0_gripper_qpos"],
                                )
                            ),
                            "prompt": str(task_description),
                        }
                        if not episode_policy_started:
                            element["reset"] = True
                            episode_policy_started = True

                        # Inject per-task goal frame if applicable. For Stage 2.2 this is the
                        # `observation/goal_image`; for Pure IDM it is sent as `observation/next_image`
                        # so the IDM head treats it as the target frame.
                        if goal_send_key is not None and task_goal_image is not None:
                            element[goal_send_key] = task_goal_image

                        # Query model to get action
                        action_chunk = client.infer(element)["actions"]
                        assert (
                            len(action_chunk) >= args.replan_steps
                        ), f"We want to replan every {args.replan_steps} steps, but policy only predicts {len(action_chunk)} steps."
                        action_plan.extend(action_chunk[: args.replan_steps])

                    action = action_plan.popleft()

                    # Execute action in environment
                    obs, reward, done, info = env.step(action.tolist())
                    if done:
                        task_successes += 1
                        total_successes += 1
                        break
                    t += 1

                except Exception as e:
                    logging.error(f"Caught exception: {e}")
                    break

            task_episodes += 1
            total_episodes += 1

            # Save replay videos for the episode
            suffix = "success" if done else "failure"
            task_segment = task_description.replace(" ", "_").replace("/", "_")
            episode_record = {
                "task_id": int(task_id),
                "task_description": str(task_description),
                "episode_idx": int(episode_idx),
                "success": bool(done),
                "reward": float(reward),
                "steps_executed": int(t),
            }

            if args.save_videos and replay_images:
                video_path = video_root / f"task_{task_id:03d}_episode_{episode_idx:03d}_{suffix}.mp4"
                imageio.mimwrite(
                    video_path,
                    [np.asarray(x) for x in replay_images],
                    fps=10,
                )
                episode_record["video_path"] = str(video_path)

            if args.save_videos and args.save_preprocessed_video and replay_input_images:
                input_video_path = (
                    video_root / f"task_{task_id:03d}_episode_{episode_idx:03d}_{suffix}_policy_input.mp4"
                )
                imageio.mimwrite(
                    input_video_path,
                    [np.asarray(x) for x in replay_input_images],
                    fps=10,
                )
                episode_record["policy_input_video_path"] = str(input_video_path)

            episode_records.append(episode_record)

            # Log current results
            logging.info(f"Success: {done}")
            logging.info(f"# episodes completed so far: {total_episodes}")
            logging.info(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")

        # Log final results
        task_success_rate = float(task_successes) / float(task_episodes)
        task_summary = {
            "task_id": int(task_id),
            "task_description": str(task_description),
            "episodes": int(task_episodes),
            "successes": int(task_successes),
            "success_rate": float(task_success_rate),
        }
        task_summaries.append(task_summary)
        logging.info(f"Current task success rate: {task_success_rate}")
        logging.info(f"Current total success rate: {float(total_successes) / float(total_episodes)}")

    if total_episodes == 0:
        if test_task_filter is not None:
            raise RuntimeError(
                "No LIBERO tasks matched any source_task in --test-tasks-json. "
                "Verify that the task suite contains the expected scenes."
            )
        raise RuntimeError("No episodes were executed; cannot compute success rate.")

    total_success_rate = float(total_successes) / float(total_episodes)
    logging.info(f"Total success rate: {total_success_rate}")
    logging.info(f"Total episodes: {total_episodes}")

    if summary_root is not None:
        summary = {
            "run_name": run_name,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "task_suite_name": args.task_suite_name,
            "num_tasks": int(num_tasks_in_suite),
            "num_trials_per_task": int(args.num_trials_per_task),
            "total_episodes": int(total_episodes),
            "total_successes": int(total_successes),
            "total_success_rate": float(total_success_rate),
            "server_metadata": server_metadata,
            "videos_dir": str(video_root) if args.save_videos else None,
            "tasks": task_summaries,
            "episodes": episode_records,
        }
        summary_path = summary_root / "summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        logging.info(f"Saved rollout summary to {summary_path}")


def _get_libero_env(task, resolution, seed):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


def _load_test_tasks_filter(json_path: str) -> dict:
    """Parse libero_wm/meta/test_tasks.json into a {source_task: record} dict.

    Each record carries the full SCENE-prefixed task string used to look up
    matching episodes in libero_wm's episodes.jsonl.
    """
    p = pathlib.Path(json_path).expanduser().resolve()
    if not p.is_file():
        raise FileNotFoundError(f"test_tasks_json not found: {json_path}")
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    records = data.get("records") if isinstance(data, dict) else None
    if not isinstance(records, list) or not records:
        raise ValueError(f"test_tasks_json is missing a non-empty 'records' list: {json_path}")
    filt: dict = {}
    for rec in records:
        for k in ("task", "source_task"):
            if k not in rec:
                raise ValueError(f"test_tasks record missing '{k}': {rec}")
        filt[str(rec["source_task"])] = rec
    return filt


def _ensure_goal_frame_from_libero_wm(
    wm_root: pathlib.Path,
    cache_dir: pathlib.Path,
    source_task: str,
    full_task: str,
) -> pathlib.Path:
    """Return a path to the goal-frame PNG for `source_task`, extracting from
    libero_wm parquet data if not already cached."""
    cache_path = cache_dir / f"{source_task}.png"
    if cache_path.is_file():
        return cache_path

    eps_path = wm_root / "meta" / "episodes.jsonl"
    episode_index = None
    with open(eps_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if full_task in (rec.get("tasks") or []):
                episode_index = int(rec["episode_index"])
                break
    if episode_index is None:
        raise ValueError(f"No episode in {eps_path} matches task: {full_task!r}")

    try:
        import pyarrow.parquet as pq
    except ImportError as e:
        raise RuntimeError(
            "pyarrow is required to extract goal frames from libero_wm parquet."
        ) from e

    chunk = episode_index // 1000
    parquet_path = (
        wm_root / "data" / f"chunk-{chunk:03d}" / f"episode_{episode_index:06d}.parquet"
    )
    if not parquet_path.is_file():
        raise FileNotFoundError(f"libero_wm parquet not found: {parquet_path}")

    images = pq.ParquetFile(parquet_path).read(columns=["image"]).column("image").to_pylist()
    if not images:
        raise ValueError(f"libero_wm parquet has no rows: {parquet_path}")
    last = images[-1]
    if isinstance(last, dict) and "bytes" in last and last["bytes"] is not None:
        img_bytes = bytes(last["bytes"])
    elif isinstance(last, (bytes, bytearray)):
        img_bytes = bytes(last)
    else:
        raise TypeError(
            f"Unsupported image entry in {parquet_path} (type={type(last).__name__})"
        )

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as f:
        f.write(img_bytes)
    logging.info(
        f"Cached goal frame for source_task={source_task} "
        f"(episode_index={episode_index}) -> {cache_path}"
    )
    return cache_path


def _load_goal_frames_index(json_path: str) -> dict:
    """Load a goal-frames JSON file into a dict.

    Returns a dict whose keys may be task descriptions and/or stringified task ids,
    and whose values are absolute pathlib.Path objects pointing at the goal image.
    """
    json_file = pathlib.Path(json_path).expanduser().resolve()
    if not json_file.is_file():
        raise FileNotFoundError(f"goal_frames_json not found: {json_path}")
    base_dir = json_file.parent
    with open(json_file, "r", encoding="utf-8") as f:
        raw = json.load(f)

    def _abs(p: str) -> pathlib.Path:
        path = pathlib.Path(p).expanduser()
        return path if path.is_absolute() else (base_dir / path).resolve()

    index: dict = {}
    if isinstance(raw, dict):
        # Either flat {key: path} or wrapped {"tasks": {...}}
        body = raw.get("tasks", raw) if "tasks" in raw and isinstance(raw["tasks"], dict) else raw
        for key, value in body.items():
            if isinstance(value, str):
                index[str(key)] = _abs(value)
            elif isinstance(value, dict) and "image_path" in value:
                index[str(key)] = _abs(value["image_path"])
    elif isinstance(raw, list):
        for record in raw:
            if not isinstance(record, dict):
                continue
            path = record.get("image_path") or record.get("path")
            if path is None:
                continue
            if "task_description" in record:
                index[str(record["task_description"])] = _abs(path)
            if "task_id" in record:
                index[str(record["task_id"])] = _abs(path)
    else:
        raise ValueError(f"Unsupported goal_frames_json structure in {json_path}.")

    if not index:
        raise ValueError(f"goal_frames_json is empty after parsing: {json_path}")
    return index


def _lookup_goal_path(index: dict, task_id: int, task_description: str) -> Optional[pathlib.Path]:
    """Resolve a goal image path for a task, trying description first then task_id."""
    if task_description in index:
        return index[task_description]
    desc_norm = str(task_description).strip()
    if desc_norm in index:
        return index[desc_norm]
    if str(task_id) in index:
        return index[str(task_id)]
    return None


def _load_and_preprocess_image(path: pathlib.Path, resize_size: int, rotate_180: bool) -> np.ndarray:
    """Load an image from disk and apply the same preprocessing pipeline used for agentview frames."""
    with Image.open(path) as pil_img:
        rgb = pil_img.convert("RGB")
        arr = np.asarray(rgb, dtype=np.uint8)
    if rotate_180:
        arr = np.ascontiguousarray(arr[::-1, ::-1])
    arr = np.asarray(image_tools.resize_with_pad(arr, resize_size, resize_size))
    return image_tools.convert_to_uint8(arr)


def _quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(eval_libero)
