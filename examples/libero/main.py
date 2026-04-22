import collections
import dataclasses
import json
import logging
import math
import pathlib
from datetime import datetime

import imageio
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import numpy as np
from client import image_tools
from client import websocket_client_policy as _websocket_client_policy
import tqdm
import tyro

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data


@dataclasses.dataclass
class Args:
    #################################################################################################################
    # Model server parameters
    #################################################################################################################
    host: str = "0.0.0.0"
    port: int = 8000
    resize_size: int | None = None
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
    summary_out_path: str | None = "data/libero/results"
    run_name: str | None = None

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

    if server_metadata.get("causal") is False:
        raise ValueError(
            "The connected policy server is non-causal and cannot drive the online LIBERO loop. "
            "Pure IDM checkpoints need the next image and are only suitable for oracle/offline evaluation."
        )

    model_resize_size = server_metadata.get("input_resolution", None)
    resize_size = int(model_resize_size) if args.resize_size is None and model_resize_size is not None else (
        int(args.resize_size) if args.resize_size is not None else 224
    )

    action_horizon = server_metadata.get("action_horizon", None)
    if action_horizon is not None and int(action_horizon) < args.replan_steps:
        raise ValueError(
            f"replan_steps={args.replan_steps} exceeds the server action horizon={action_horizon}."
        )

    if server_metadata.get("goal_conditioned"):
        logging.warning(
            "The connected policy was trained with visual goal conditioning, but this client only sends current observations "
            "unless you extend it to provide `goal_image`. The server will fall back to goal=None."
        )

    # Start evaluation
    total_episodes, total_successes = 0, 0
    episode_records = []
    task_summaries = []
    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
        # Get task
        task = task_suite.get_task(task_id)

        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)

        # Initialize LIBERO environment and task description
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

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
