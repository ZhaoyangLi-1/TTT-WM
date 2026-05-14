import collections
import dataclasses
import json
import logging
import math
import pathlib
from datetime import datetime
from typing import Dict, List, Optional

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
    max_steps: Optional[int] = None  # Override the per-suite max episode length

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
    if args.max_steps is not None:
        logging.info(
            f"Overriding max_steps for {args.task_suite_name}: {max_steps} -> {args.max_steps}"
        )
        max_steps = int(args.max_steps)

    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    server_metadata = client.get_server_metadata()
    logging.info(f"Connected to policy server metadata: {server_metadata}")
    # [DIAG] Pretty-print abs_action wiring so a bad rollout can be triaged
    # without hunting through the verbose metadata blob.
    logging.info(
        "[DIAG] server abs_action=%s, action_dim=%s, action_horizon=%s",
        server_metadata.get("abs_action"),
        server_metadata.get("action_dim"),
        server_metadata.get("action_horizon"),
    )

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
    # When the policy was trained with abs_action=True it outputs absolute
    # 7D ``[xyz, axis_angle, gripper]`` (server already decoded from 10D),
    # so the env must run its OSC controller in abs mode.
    abs_action = bool(server_metadata.get("abs_action", False))
    if abs_action:
        logging.info("Server reports abs_action=True; switching OSC controller to use_delta=False.")

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
        env, task_description = _get_libero_env(
            task, LIBERO_ENV_RESOLUTION, args.seed, abs_action=abs_action,
        )
        # [DIAG] Verify OSC was actually flipped to abs mode (use_delta=False).
        # If this prints True, abs_action plumbing is broken end-to-end and the
        # policy's absolute targets are being interpreted as deltas → robot drifts.
        try:
            _diag_ctrl = env.env.robots[0].controller
            logging.info(
                "[DIAG] task_id=%s controller=%s use_delta=%s",
                task_id, type(_diag_ctrl).__name__, getattr(_diag_ctrl, "use_delta", None),
            )
        except Exception as _diag_exc:
            logging.warning("[DIAG] could not read controller.use_delta: %s", _diag_exc)

        # Resolve and pre-process this task's goal frame(s). Each online episode
        # gets its own goal image keyed by episode_idx so callers can align the
        # goal scene to the LIBERO init state actually used for that rollout.
        task_goal_images: Dict[int, np.ndarray] = {}
        # Per-episode source path of the goal PNG (preserved at original orientation)
        # so we can dump a 228x228 copy alongside the rollout video without re-running
        # the orientation flip applied for the policy server.
        task_goal_source_paths: Dict[int, pathlib.Path] = {}
        # Which libero_wm episode_index was used for each online episode_idx (when
        # the goal came from libero_wm). Powers split bookkeeping in summary.json.
        task_goal_episode_indices: Dict[int, int] = {}
        # Set of libero_wm episode_indices that should be labelled as the held-out
        # test/val split in summary.json.
        test_episode_indices: set = set(
            int(x) for x in (test_record.get("test_episode_indices") if test_record else None) or []
        )
        # When goal_align_by_init_proprio is True, defer goal selection to the
        # episode loop (after env.set_init_state) so we can pick the libero_wm
        # demo whose t=0 proprio is closest to the online rollout's t=0 proprio
        # — this matches the training distribution where (context, goal) come
        # from the same trajectory's start and end.
        align_by_proprio = bool(
            test_record.get("goal_align_by_init_proprio", False) if test_record else False
        )
        candidate_goal_eps: List[int] = []
        candidate_goal_proprios: Dict[int, np.ndarray] = {}
        # Cache for loaded goal images / source paths during runtime selection.
        runtime_goal_image_cache: Dict[int, np.ndarray] = {}
        runtime_goal_path_cache: Dict[int, pathlib.Path] = {}
        if align_by_proprio and test_record is not None and goal_send_key is not None:
            assert wm_root is not None and cache_dir is not None
            built = _build_goal_set(test_record)
            if built is None:
                raise ValueError(
                    "goal_align_by_init_proprio=true requires goal_set_episode_range "
                    "(and optional goal_set_excluded_episodes) to define the candidate pool."
                )
            candidate_goal_eps = list(built)
            for ep in candidate_goal_eps:
                candidate_goal_proprios[ep] = _load_first_proprio_from_libero_wm(
                    wm_root, ep
                )
            logging.info(
                f"goal_align_by_init_proprio=True: candidate pool size={len(candidate_goal_eps)} "
                f"(task_id={task_id})"
            )

        if align_by_proprio:
            # Skip the per-episode_idx preload — selection happens after set_init_state.
            pass
        elif test_record is not None and goal_send_key is not None:
            assert wm_root is not None and cache_dir is not None
            ep_to_image: Dict[int, np.ndarray] = {}
            ep_to_path: Dict[int, pathlib.Path] = {}
            for episode_idx in range(args.num_trials_per_task):
                ep_for_goal = _resolve_goal_episode_index(test_record, episode_idx)
                if ep_for_goal is None:
                    continue
                cached_image = ep_to_image.get(ep_for_goal)
                if cached_image is None:
                    goal_path = _ensure_goal_frame_from_libero_wm(
                        wm_root=wm_root,
                        cache_dir=cache_dir,
                        source_task=test_record["source_task"],
                        full_task=test_record["task"],
                        episode_index_override=ep_for_goal,
                    )
                    cached_image = _load_and_preprocess_image(
                        goal_path, resize_size, args.rotate_images_180
                    )
                    ep_to_image[ep_for_goal] = cached_image
                    ep_to_path[ep_for_goal] = goal_path
                    split_tag = "test" if ep_for_goal in test_episode_indices else "train"
                    logging.info(
                        f"Loaded goal frame for task_id={task_id} ({bddl_stem}) "
                        f"libero_wm_episode={ep_for_goal} ({split_tag}) from {goal_path}"
                    )
                task_goal_images[episode_idx] = cached_image
                task_goal_source_paths[episode_idx] = ep_to_path[ep_for_goal]
                task_goal_episode_indices[episode_idx] = int(ep_for_goal)
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
                shared_image = _load_and_preprocess_image(
                    goal_path, resize_size, args.rotate_images_180
                )
                for episode_idx in range(args.num_trials_per_task):
                    task_goal_images[episode_idx] = shared_image
                    task_goal_source_paths[episode_idx] = goal_path
                logging.info(f"Loaded goal frame for task {task_id} from {goal_path}")

        # Start episodes
        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm.tqdm(range(args.num_trials_per_task)):
            logging.info(f"\nTask: {task_description}")

            task_goal_image: Optional[np.ndarray] = task_goal_images.get(episode_idx)

            # Reset environment
            env.reset()
            action_plan = collections.deque()

            # Set initial states
            obs = env.set_init_state(initial_states[episode_idx])

            # Runtime goal selection: pick the libero_wm demo whose t=0 proprio
            # is closest to the env's t=0 proprio after set_init_state. This
            # produces a (rollout-start, goal-end) pair drawn from approximately
            # the same init state, restoring the training distribution where
            # context and goal share a trajectory.
            if (
                align_by_proprio
                and goal_send_key is not None
                and candidate_goal_eps
            ):
                online_proprio = _online_proprio_from_obs(obs)
                ep_for_goal = min(
                    candidate_goal_eps,
                    key=lambda e: float(np.linalg.norm(
                        online_proprio - candidate_goal_proprios[e]
                    )),
                )
                cached_image = runtime_goal_image_cache.get(ep_for_goal)
                if cached_image is None:
                    goal_path = _ensure_goal_frame_from_libero_wm(
                        wm_root=wm_root,
                        cache_dir=cache_dir,
                        source_task=test_record["source_task"],
                        full_task=test_record["task"],
                        episode_index_override=ep_for_goal,
                    )
                    cached_image = _load_and_preprocess_image(
                        goal_path, resize_size, args.rotate_images_180
                    )
                    runtime_goal_image_cache[ep_for_goal] = cached_image
                    runtime_goal_path_cache[ep_for_goal] = goal_path
                task_goal_image = cached_image
                task_goal_images[episode_idx] = cached_image
                task_goal_source_paths[episode_idx] = runtime_goal_path_cache[ep_for_goal]
                task_goal_episode_indices[episode_idx] = int(ep_for_goal)
                proprio_l2 = float(np.linalg.norm(
                    online_proprio - candidate_goal_proprios[ep_for_goal]
                ))
                split_tag = "test" if ep_for_goal in test_episode_indices else "train"
                logging.info(
                    f"[align_by_proprio] task_id={task_id} ({bddl_stem}) "
                    f"online_ep={episode_idx} → libero_wm_ep={ep_for_goal} "
                    f"({split_tag}, proprio_l2={proprio_l2:.4f})"
                )

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
                        _ee_pos = np.asarray(obs["robot0_eef_pos"], dtype=np.float32)
                        _ee_ori = np.asarray(
                            _quat2axisangle(obs["robot0_eef_quat"]), dtype=np.float32
                        )
                        element = {
                            "observation/image": img,
                            "observation/wrist_image": wrist_img,
                            # Full 8D proprio (back-compat) plus pre-sliced ee_pos
                            # and ee_ori in axis-angle. The split keys let policy
                            # adapters use either layout without rebaking.
                            "observation/state": np.concatenate(
                                (
                                    _ee_pos,
                                    _ee_ori,
                                    np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32),
                                )
                            ),
                            "observation/ee_pos": _ee_pos,
                            "observation/ee_ori": _ee_ori,
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
                        # [DIAG] First-chunk dump for the very first inference
                        # call of the run (regardless of task_id, since
                        # --test-tasks-json may filter away task_id=0). Used to
                        # sanity-check the policy's action ranges against the
                        # current ee pose.
                        if not getattr(env, "_diag_first_chunk_logged", False):
                            env._diag_first_chunk_logged = True
                            _ee_pos = obs["robot0_eef_pos"]
                            _ee_quat = obs["robot0_eef_quat"]
                            logging.info(
                                "[DIAG] action_chunk.shape=%s dtype=%s",
                                getattr(action_chunk, "shape", None),
                                getattr(action_chunk, "dtype", None),
                            )
                            # Dump every step of the chunk so we can see
                            # whether xyz is monotonically drifting within a
                            # single inference (the no-proprio failure mode
                            # for abs_action).
                            for _i, _a in enumerate(action_chunk):
                                _a = np.asarray(_a)
                                logging.info(
                                    "[DIAG] action_chunk[%d] xyz=%s axis_angle=%s grip=%.3f",
                                    _i,
                                    np.round(_a[:3], 5).tolist(),
                                    np.round(_a[3:6], 4).tolist(),
                                    float(_a[6]),
                                )
                            logging.info(
                                "[DIAG] obs robot0_eef_pos=%s eef_quat=%s",
                                np.asarray(_ee_pos).tolist(),
                                np.asarray(_ee_quat).tolist(),
                            )
                            # also dump first sent image's orientation signature
                            _h = img.shape[0]
                            logging.info(
                                "[DIAG] img sent shape=%s dtype=%s top-row mean=%.2f bottom-row mean=%.2f",
                                tuple(img.shape), img.dtype,
                                float(img[0].mean()), float(img[_h - 1].mean()),
                            )
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
            ep_for_goal = task_goal_episode_indices.get(episode_idx)
            if ep_for_goal is not None:
                episode_record["goal_libero_wm_episode_index"] = int(ep_for_goal)
                episode_record["goal_split"] = (
                    "test" if int(ep_for_goal) in test_episode_indices else "train"
                )

            if args.save_videos and replay_images:
                video_path = video_root / f"task_{task_id:03d}_episode_{episode_idx:03d}_{suffix}.mp4"
                rendered_frames = []
                for frame in replay_images:
                    arr = np.ascontiguousarray(np.asarray(frame)[::-1, ::-1])
                    pil = Image.fromarray(arr).resize((228, 228), Image.BILINEAR)
                    rendered_frames.append(np.asarray(pil))
                imageio.mimwrite(
                    video_path,
                    rendered_frames,
                    fps=10,
                    macro_block_size=1,
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

            # Dump the per-episode goal frame as a 228x228 PNG next to the video.
            # We re-load from the cached source PNG and rotate 180 degrees so the
            # saved goal sits in the same orientation as the rollout video, making
            # visual A/B pairing of "asked goal" vs "actually reached" trivial.
            if args.save_videos and episode_idx in task_goal_source_paths:
                goal_src_path = task_goal_source_paths[episode_idx]
                goal_png_path = (
                    video_root / f"task_{task_id:03d}_episode_{episode_idx:03d}_{suffix}_goal.png"
                )
                with Image.open(goal_src_path) as goal_img:
                    goal_img = (
                        goal_img.convert("RGB")
                        .transpose(Image.ROTATE_180)
                        .resize((228, 228), Image.BILINEAR)
                    )
                    goal_img.save(goal_png_path)
                episode_record["goal_image_path"] = str(goal_png_path)
                episode_record["goal_source_path"] = str(goal_src_path)

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


def _get_libero_env(task, resolution, seed, *, abs_action: bool = False):
    """Initializes and returns the LIBERO environment, along with the task description.

    When ``abs_action=True``, the underlying robosuite OSC_POSE controller is
    flipped to ``use_delta=False`` so it interprets policy actions as absolute
    target poses (xyz + axis-angle) instead of normalized deltas. Matches
    lpb's ``libero_image_runner`` setup for diffusion policies trained with
    abs_action=True.
    """
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    if abs_action:
        # LIBERO's OffScreenRenderEnv builds controller_configs internally
        # (env_wrapper.py:47) so the only post-hoc knob is the live OSC
        # instance. robosuite >=1.4 stores the flag as `use_delta`.
        for robot in env.env.robots:
            controller = getattr(robot, "controller", None)
            if controller is None or not hasattr(controller, "use_delta"):
                raise RuntimeError(
                    "abs_action=True requires an OSC controller with `use_delta`; "
                    f"got {type(controller).__name__ if controller else None}."
                )
            controller.use_delta = False
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


def _build_goal_set(test_record: dict) -> Optional[List[int]]:
    """Materialise the ``goal_set_episode_range`` + ``goal_set_excluded_episodes``
    schema into an explicit list of libero_wm episode indices.

    Returns ``None`` if the record does not use this schema. Otherwise the
    returned list contains every integer in the inclusive range, minus any
    excluded indices, sorted ascending. Use this to mirror "training set goal
    frames excluding validation" without listing all 49 indices by hand.
    """
    rng = test_record.get("goal_set_episode_range")
    if rng is None:
        return None
    if not (isinstance(rng, (list, tuple)) and len(rng) == 2):
        raise ValueError(
            f"goal_set_episode_range must be [start, end] (inclusive), got {rng!r}"
        )
    start, end = int(rng[0]), int(rng[1])
    if end < start:
        raise ValueError(
            f"goal_set_episode_range end={end} must be >= start={start}"
        )
    excluded = {
        int(e) for e in (test_record.get("goal_set_excluded_episodes") or [])
    }
    return [i for i in range(start, end + 1) if i not in excluded]


def _resolve_goal_episode_index(test_record: dict, episode_idx: int) -> Optional[int]:
    """Pick the libero_wm episode_index whose terminal frame should serve as the
    goal for online ``episode_idx``. Schema priority (highest first):

    1. ``goal_set_episode_range`` + optional ``goal_set_excluded_episodes``: builds
       a goal set covering ``[start, end]`` minus excluded indices, then maps
       ``episode_idx`` onto the set deterministically via modulo. Use this for
       "training-set goal frames excluding validation": set range to the full
       50-episode block and list the val index in excluded.
    2. ``goal_episode_indices``: an explicit list of libero_wm episode indices,
       one per online episode. ``goal_episode_indices[episode_idx]`` is used.
    3. ``goal_episode_index_base``: an integer; resolved to ``base + episode_idx``.
       Use when libero_wm episodes for this task are stored consecutively in the
       same order as LIBERO benchmark ``initial_states[0..49]`` and you do NOT
       need to skip the val episode.
    4. ``goal_episode_index``: a single integer; same goal frame for every
       online episode (legacy / smoke-test behaviour).
    """
    goal_set = _build_goal_set(test_record)
    if goal_set is not None:
        if not goal_set:
            raise ValueError(
                "goal_set_episode_range produced an empty goal set after "
                "applying goal_set_excluded_episodes."
            )
        return int(goal_set[episode_idx % len(goal_set)])
    indices = test_record.get("goal_episode_indices")
    if isinstance(indices, list) and indices:
        if episode_idx >= len(indices):
            raise ValueError(
                f"goal_episode_indices has {len(indices)} items but online "
                f"episode_idx={episode_idx} was requested. Either extend the list "
                f"or lower --num-trials-per-task."
            )
        return int(indices[episode_idx])
    base = test_record.get("goal_episode_index_base")
    if base is not None:
        return int(base) + int(episode_idx)
    single = test_record.get("goal_episode_index")
    if single is not None:
        return int(single)
    return None


def _load_first_proprio_from_libero_wm(
    wm_root: pathlib.Path, ep_index: int
) -> np.ndarray:
    """Read the t=0 ``state`` row of ``episode_<ep_index>.parquet``. libero_wm's
    state schema is the 8-dim concatenation of [eef_pos(3), eef_axis_angle(3),
    gripper_qpos(2)] — exactly what main.py constructs from the LIBERO env at
    runtime — so this vector can be compared 1-to-1 to the online proprio.
    """
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError(
            "pyarrow is required to extract first proprio from libero_wm parquet."
        ) from exc

    chunk = ep_index // 1000
    parquet_path = (
        wm_root / "data" / f"chunk-{chunk:03d}" / f"episode_{ep_index:06d}.parquet"
    )
    if not parquet_path.is_file():
        raise FileNotFoundError(f"libero_wm parquet not found: {parquet_path}")
    state_rows = (
        pq.ParquetFile(str(parquet_path))
        .read(columns=["state"])
        .column("state")
        .to_pylist()
    )
    if not state_rows:
        raise ValueError(f"libero_wm parquet has no state rows: {parquet_path}")
    return np.asarray(state_rows[0], dtype=np.float64)


def _online_proprio_from_obs(obs: dict) -> np.ndarray:
    """Build the same 8-dim proprio used by main.py's ``observation/state``."""
    return np.concatenate(
        (
            np.asarray(obs["robot0_eef_pos"], dtype=np.float64),
            _quat2axisangle(np.asarray(obs["robot0_eef_quat"], dtype=np.float64)),
            np.asarray(obs["robot0_gripper_qpos"], dtype=np.float64),
        )
    )


def _ensure_goal_frame_from_libero_wm(
    wm_root: pathlib.Path,
    cache_dir: pathlib.Path,
    source_task: str,
    full_task: str,
    *,
    episode_index_override: Optional[int] = None,
) -> pathlib.Path:
    """Return a path to the goal-frame PNG for `source_task`, extracting from
    libero_wm parquet data if not already cached.

    When ``episode_index_override`` is provided (e.g. a held-out validation
    episode the policy never saw during training), that exact episode is used
    instead of the first ``full_task``-matching record in episodes.jsonl.
    """
    if episode_index_override is not None:
        episode_index = int(episode_index_override)
        cache_path = cache_dir / f"{source_task}__episode_{episode_index:06d}.png"
    else:
        cache_path = cache_dir / f"{source_task}.png"
    if cache_path.is_file():
        return cache_path

    if episode_index_override is None:
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
    # `force=True` overrides any handler that robosuite / gym / jax may have
    # attached at import time. Without it, those libraries leave the root
    # logger at WARNING and every `logging.info(...)` here is silently
    # dropped (including the `[DIAG]` rollout-diagnostic prints).
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        force=True,
    )
    eval_libero(tyro.cli(Args))
