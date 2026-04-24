"""
Evaluate a Stage 1 checkpoint on fixed held-out tasks with reproducible episode
sampling.

Default behavior:
  - use the three LIBERO held-out tasks discussed during training
  - sample 5 episodes per task with a fixed seed
  - run teacher-forcing pixel MSE on all valid windows in each sampled episode
  - save 5 reproducible teacher-forcing examples per episode as
    [context | predicted | target | goal]
  - run open-loop rollout MSE from the start of each sampled episode
  - compute teacher-forcing FID on predicted vs target frames
  - save side-by-side rollout videos and JSON summaries

Example:
    python eval_heldout_tasks.py \
        --checkpoint /path/to/stage1_best.pt \
        --dataset /path/to/libero_wm \
        --seed 42 \
        --output-dir /scr2/zhaoyang/TTT-WM-outputs/stage1 \     
        --episodes-per-task 5
"""

from __future__ import annotations

import argparse
import io
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.models import Inception_V3_Weights, inception_v3
from torchvision import transforms
from tqdm import tqdm

from eval import frames_to_uint8, load_model_from_checkpoint, save_video


DEFAULT_TASKS = [
    "KITCHEN_SCENE10: put the butter at the back in the top drawer of the cabinet and close it",
    "KITCHEN_SCENE2: put the middle black bowl on the plate",
    "STUDY_SCENE3: pick up the book and place it in the front compartment of the caddy",
]
DEFAULT_EXAMPLES_PER_EPISODE = 5


def slugify(text: str, max_len: int = 96) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_").lower()
    return (slug[:max_len].rstrip("_") or "task")


def build_transform(resolution: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((resolution, resolution)),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ]
    )


def extract_image_bytes(value):
    if isinstance(value, dict) and "bytes" in value:
        return value["bytes"]
    return value


def decode_image(value, transform: transforms.Compose) -> torch.Tensor:
    raw = extract_image_bytes(value)
    if isinstance(raw, (bytes, bytearray)):
        image = Image.open(io.BytesIO(raw)).convert("RGB")
    elif isinstance(raw, Image.Image):
        image = raw.convert("RGB")
    else:
        image = Image.fromarray(np.asarray(raw)).convert("RGB")
    return transform(image)


def load_dataset_metadata(dataset_root: str):
    root = Path(dataset_root)
    info_path = root / "meta" / "info.json"
    episodes_path = root / "meta" / "episodes.jsonl"
    if not info_path.is_file() or not episodes_path.is_file():
        raise FileNotFoundError(
            f"{root} is missing meta/info.json or meta/episodes.jsonl."
        )

    with info_path.open() as f:
        info = json.load(f)

    task_to_episodes: dict[str, list[int]] = defaultdict(list)
    episode_lengths: dict[int, int] = {}
    with episodes_path.open() as f:
        for line in f:
            record = json.loads(line)
            episode_idx = int(record["episode_index"])
            tasks = record.get("tasks", [])
            task_name = tasks[0] if tasks else None
            episode_lengths[episode_idx] = int(record["length"])
            if task_name is not None:
                task_to_episodes[task_name].append(episode_idx)

    for episodes in task_to_episodes.values():
        episodes.sort()

    return {
        "root": root,
        "chunks_size": int(info["chunks_size"]),
        "task_to_episodes": dict(task_to_episodes),
        "episode_lengths": episode_lengths,
    }


def resolve_episode_path(root: Path, chunks_size: int, episode_idx: int) -> Path:
    return (
        root
        / "data"
        / f"chunk-{episode_idx // chunks_size:03d}"
        / f"episode_{episode_idx:06d}.parquet"
    )


def load_episode_frames(
    root: Path,
    chunks_size: int,
    episode_idx: int,
    image_key: str,
    transform: transforms.Compose,
    flip_vertical: bool = False,
) -> list[torch.Tensor]:
    parquet_path = resolve_episode_path(root, chunks_size, episode_idx)
    if not parquet_path.is_file():
        raise FileNotFoundError(f"Episode parquet not found: {parquet_path}")

    df = pd.read_parquet(parquet_path, columns=[image_key])
    frames = [decode_image(value, transform) for value in df[image_key].tolist()]
    if flip_vertical:
        frames = [torch.flip(frame, dims=(-2,)) for frame in frames]
    return frames


class RunningFeatureStats:
    def __init__(self, dim: int):
        self.dim = int(dim)
        self.count = 0
        self.sum = torch.zeros(self.dim, dtype=torch.float64)
        self.sum_outer = torch.zeros(self.dim, self.dim, dtype=torch.float64)

    def update(self, features: torch.Tensor):
        if features.numel() == 0:
            return
        x = features.detach().cpu().to(torch.float64)
        self.count += int(x.shape[0])
        self.sum += x.sum(dim=0)
        self.sum_outer += x.transpose(0, 1) @ x

    def mean_and_cov(self):
        if self.count < 2:
            return None, None
        mean = self.sum / self.count
        cov = (self.sum_outer - self.count * torch.outer(mean, mean)) / (self.count - 1)
        return mean, cov


class FIDAccumulator:
    def __init__(self, feature_dim: int = 2048):
        self.real = RunningFeatureStats(feature_dim)
        self.fake = RunningFeatureStats(feature_dim)

    def update_features(self, real_features: torch.Tensor, fake_features: torch.Tensor):
        if real_features.shape[0] != fake_features.shape[0]:
            raise ValueError(
                f"FID feature count mismatch: real={real_features.shape[0]}, fake={fake_features.shape[0]}"
            )
        self.real.update(real_features)
        self.fake.update(fake_features)

    def compute(self) -> float | None:
        mu_real, cov_real = self.real.mean_and_cov()
        mu_fake, cov_fake = self.fake.mean_and_cov()
        if mu_real is None or mu_fake is None:
            return None

        diff = mu_real - mu_fake
        cov_prod = cov_real @ cov_fake
        eigvals = torch.linalg.eigvals(cov_prod)
        trace_sqrt = torch.sqrt(eigvals.real.clamp(min=0)).sum()
        fid = diff.dot(diff) + torch.trace(cov_real) + torch.trace(cov_fake) - 2.0 * trace_sqrt
        return float(max(fid.item(), 0.0))


class InceptionFeatureExtractor:
    def __init__(self, device: torch.device):
        self.device = device
        self.model = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)
        self.model.fc = torch.nn.Identity()
        self.model.eval().to(device)
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    def extract(self, images: torch.Tensor, batch_size: int = 32) -> torch.Tensor:
        if images.numel() == 0:
            return torch.empty((0, 2048), dtype=torch.float64)

        features: list[torch.Tensor] = []
        with torch.no_grad():
            for start in range(0, images.shape[0], int(batch_size)):
                batch = images[start : start + int(batch_size)].to(
                    self.device, dtype=torch.float32, non_blocking=True
                )
                batch = batch.clamp(-1, 1) * 0.5 + 0.5
                batch = F.interpolate(
                    batch, size=(299, 299), mode="bilinear", align_corners=False
                )
                batch = (batch - self.mean) / self.std
                output = self.model(batch)
                if isinstance(output, tuple):
                    output = output[0]
                elif hasattr(output, "logits"):
                    output = output.logits
                features.append(output.detach().cpu().to(torch.float64))

        return torch.cat(features, dim=0)


def tensor_to_uint8_image(frame: torch.Tensor) -> np.ndarray:
    if frame.dim() == 4:
        return frames_to_uint8(frame[:1])[0]
    if frame.dim() == 3:
        return frames_to_uint8(frame.unsqueeze(0))[0]
    raise ValueError(f"Expected 3D or 4D frame tensor, got shape {tuple(frame.shape)}")


def upscale_uint8(image: np.ndarray, scale: int) -> np.ndarray:
    if scale is None or int(scale) <= 1:
        return image
    scale = int(scale)
    pil = Image.fromarray(image)
    resized = pil.resize(
        (pil.width * scale, pil.height * scale), resample=Image.NEAREST
    )
    return np.asarray(resized)


def save_uint8_image(image: np.ndarray, path: Path, scale: int = 1):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(upscale_uint8(image, scale)).save(path)


def build_example_window_starts(
    n_windows: int,
    example_count: int,
    seed: int,
    task_index: int,
    episode_idx: int,
) -> list[int]:
    if n_windows <= 0 or example_count <= 0:
        return []

    rng = np.random.default_rng(
        np.random.SeedSequence([int(seed), int(task_index), int(episode_idx), int(n_windows)])
    )
    starts = [0]
    need = int(example_count) - 1
    remaining_pool = list(range(1, n_windows))

    if need > 0 and remaining_pool:
        take = min(need, len(remaining_pool))
        sampled = rng.choice(remaining_pool, size=take, replace=False).tolist()
        starts.extend(int(x) for x in sampled)
        need -= take

    if need > 0:
        fallback_pool = remaining_pool if remaining_pool else [0]
        extra = rng.choice(fallback_pool, size=need, replace=True).tolist()
        starts.extend(int(x) for x in extra)

    return starts


def save_teacher_forcing_examples(
    episode_dir: Path,
    goal_frame: torch.Tensor,
    examples: list[dict],
    image_scale: int = 1,
) -> tuple[str, list[dict]]:
    goal_path = episode_dir / "goal.png"
    save_uint8_image(tensor_to_uint8_image(goal_frame), goal_path, scale=image_scale)

    examples_dir = episode_dir / "examples"
    examples_dir.mkdir(parents=True, exist_ok=True)
    saved_examples: list[dict] = []

    for example in sorted(examples, key=lambda item: item["slot_index"]):
        context_img = upscale_uint8(tensor_to_uint8_image(example["context"]), image_scale)
        pred_img = upscale_uint8(tensor_to_uint8_image(example["predicted"]), image_scale)
        target_img = upscale_uint8(tensor_to_uint8_image(example["target"]), image_scale)
        goal_img = upscale_uint8(tensor_to_uint8_image(goal_frame), image_scale)

        height = context_img.shape[0]
        sep_width = max(4, 4 * max(1, int(image_scale)))
        separator = np.full((height, sep_width, 3), 128, dtype=np.uint8)
        combined = np.concatenate(
            [
                context_img,
                separator,
                pred_img,
                separator,
                target_img,
                separator,
                goal_img,
            ],
            axis=1,
        )

        example_path = examples_dir / (
            f"example_{example['slot_index']:02d}_start_{example['window_start']:04d}.png"
        )
        save_uint8_image(combined, example_path)
        saved_examples.append(
            {
                "slot_index": int(example["slot_index"]),
                "window_start": int(example["window_start"]),
                "mse": float(example["mse"]),
                "path": str(example_path.resolve()),
            }
        )

    return str(goal_path.resolve()), saved_examples


def sample_task_episodes(
    task_to_episodes: dict[str, list[int]],
    tasks: list[str],
    episodes_per_task: int,
    seed: int,
) -> dict[str, list[int]]:
    rng = np.random.default_rng(seed)
    sampled: dict[str, list[int]] = {}
    missing_tasks: list[str] = []
    undersized_tasks: list[str] = []

    for task in tasks:
        pool = list(task_to_episodes.get(task, []))
        if not pool:
            missing_tasks.append(task)
            continue
        if len(pool) < episodes_per_task:
            undersized_tasks.append(
                f"{task} (requested {episodes_per_task}, found {len(pool)})"
            )
            continue
        chosen = rng.choice(pool, size=episodes_per_task, replace=False)
        sampled[task] = sorted(int(x) for x in chosen.tolist())

    if missing_tasks or undersized_tasks:
        problems: list[str] = []
        if missing_tasks:
            problems.append(
                "Missing tasks in dataset root:\n  - " + "\n  - ".join(missing_tasks)
            )
        if undersized_tasks:
            problems.append(
                "Tasks with too few episodes:\n  - "
                + "\n  - ".join(undersized_tasks)
            )
        raise ValueError("\n".join(problems))

    return sampled


def evaluate_teacher_forcing_episode(
    model,
    frames: list[torch.Tensor],
    frames_in: int,
    frames_out: int,
    frame_gap: int,
    use_goal: bool,
    device: torch.device,
    use_amp: bool,
    task_index: int,
    episode_idx: int,
    example_seed: int,
    example_count: int,
    episode_dir: Path,
    image_scale: int = 1,
):
    span = frames_in + frame_gap + frames_out - 1
    target_offset = frames_in - 1 + frame_gap
    if len(frames) < span:
        goal_path = None
        if frames:
            goal_path = episode_dir / "goal.png"
            save_uint8_image(
                tensor_to_uint8_image(frames[-1].detach().cpu()),
                goal_path,
                scale=image_scale,
            )
        return {
            "n_windows": 0,
            "avg_mse": None,
            "first_mse": None,
            "last_mse": None,
            "goal_path": str(goal_path.resolve()) if goal_path else None,
            "examples": [],
            "fid_pred_frames": torch.empty(0, 3, 0, 0),
            "fid_target_frames": torch.empty(0, 3, 0, 0),
        }

    goal_batch = frames[-1].unsqueeze(0).to(device) if use_goal else None
    goal_frame = frames[-1].detach().cpu()
    window_mses: list[float] = []
    pred_batches: list[torch.Tensor] = []
    target_batches: list[torch.Tensor] = []
    n_windows = len(frames) - span + 1
    example_starts = build_example_window_starts(
        n_windows=n_windows,
        example_count=example_count,
        seed=example_seed,
        task_index=task_index,
        episode_idx=episode_idx,
    )
    start_to_slots: dict[int, list[int]] = defaultdict(list)
    for slot_index, start in enumerate(example_starts):
        start_to_slots[int(start)].append(int(slot_index))
    example_records: list[dict] = []

    model.eval()
    with torch.no_grad():
        for start in range(n_windows):
            context = torch.stack(frames[start : start + frames_in]).unsqueeze(0).to(device)
            target_start = start + target_offset
            target = torch.stack(
                frames[target_start : target_start + frames_out]
            ).unsqueeze(0).to(device)

            with torch.amp.autocast(
                "cuda",
                enabled=(use_amp and device.type == "cuda"),
                dtype=torch.bfloat16,
            ):
                pred_frames, _ = model(context, target, goal_batch)

            mse = F.mse_loss(pred_frames.float(), target.float()).item()
            window_mses.append(float(mse))
            pred_frames_cpu = pred_frames.detach().cpu()
            target_cpu = target.detach().cpu()
            pred_batches.append(pred_frames_cpu)
            target_batches.append(target_cpu)

            if start in start_to_slots:
                context_cpu = context[0].detach().cpu()
                for slot_index in start_to_slots[start]:
                    example_records.append(
                        {
                            "slot_index": int(slot_index),
                            "window_start": int(start),
                            "mse": float(mse),
                            "context": context_cpu[:1],
                            "predicted": pred_frames_cpu[0, :1],
                            "target": target_cpu[0, :1],
                        }
                    )

    goal_path, saved_examples = save_teacher_forcing_examples(
        episode_dir=episode_dir,
        goal_frame=goal_frame,
        image_scale=image_scale,
        examples=example_records,
    )
    fid_pred_frames = torch.cat(pred_batches, dim=0).flatten(0, 1) if pred_batches else None
    fid_target_frames = torch.cat(target_batches, dim=0).flatten(0, 1) if target_batches else None

    return {
        "n_windows": len(window_mses),
        "avg_mse": float(np.mean(window_mses)) if window_mses else None,
        "first_mse": window_mses[0] if window_mses else None,
        "last_mse": window_mses[-1] if window_mses else None,
        "goal_path": goal_path,
        "examples": saved_examples,
        "fid_pred_frames": fid_pred_frames,
        "fid_target_frames": fid_target_frames,
    }


def rollout_episode(
    model,
    frames: list[torch.Tensor],
    frame_gap: int,
    use_goal: bool,
    device: torch.device,
    use_amp: bool,
    max_steps: int,
):
    if getattr(model.cfg, "frames_in", None) != 1 or getattr(model.cfg, "frames_out", None) != 1:
        raise NotImplementedError(
            "Open-loop rollout in this script currently supports only frames_in=1 and frames_out=1."
        )
    if frame_gap <= 0:
        raise ValueError(f"Open-loop rollout requires frame_gap >= 1, got {frame_gap}.")

    available_steps = (len(frames) - 1) // frame_gap
    n_steps = available_steps if max_steps <= 0 else min(int(max_steps), available_steps)
    if n_steps <= 0:
        return {
            "n_steps": 0,
            "avg_mse": None,
            "first_mse": None,
            "last_mse": None,
            "gt_frames": [],
            "pred_frames": [],
        }

    goal_batch = frames[-1].unsqueeze(0).to(device) if use_goal else None
    window = frames[0].unsqueeze(0).unsqueeze(0).to(device)

    init_uint8 = frames_to_uint8(frames[0].unsqueeze(0))[0]
    gt_uint8 = [init_uint8]
    pred_uint8 = [init_uint8]
    step_mse: list[float] = []

    model.eval()
    with torch.no_grad():
        for step_idx in range(n_steps):
            with torch.amp.autocast(
                "cuda",
                enabled=(use_amp and device.type == "cuda"),
                dtype=torch.bfloat16,
            ):
                pred_frames = model.generate(window, goal=goal_batch)

            gt_frame = frames[(step_idx + 1) * frame_gap].unsqueeze(0).unsqueeze(0).to(device)
            mse = F.mse_loss(pred_frames.float(), gt_frame.float()).item()
            step_mse.append(float(mse))

            pred_uint8.append(frames_to_uint8(pred_frames[0])[0])
            gt_uint8.append(frames_to_uint8(gt_frame[0])[0])
            window = pred_frames

    return {
        "n_steps": len(step_mse),
        "avg_mse": float(np.mean(step_mse)) if step_mse else None,
        "first_mse": step_mse[0] if step_mse else None,
        "last_mse": step_mse[-1] if step_mse else None,
        "step_mse": step_mse,
        "gt_frames": gt_uint8,
        "pred_frames": pred_uint8,
    }


def save_rollout_comparison(
    gt_frames: list[np.ndarray],
    pred_frames: list[np.ndarray],
    output_path: Path,
    fps: int,
    image_scale: int = 1,
):
    if int(image_scale) > 1:
        gt_frames = [upscale_uint8(f, image_scale) for f in gt_frames]
        pred_frames = [upscale_uint8(f, image_scale) for f in pred_frames]
    gt_arr = np.stack(gt_frames)
    pred_arr = np.stack(pred_frames)
    height = gt_arr.shape[1]
    sep_width = max(4, 4 * max(1, int(image_scale)))
    separator = np.full((len(gt_arr), height, sep_width, 3), 128, dtype=np.uint8)
    combined = np.concatenate([gt_arr, separator, pred_arr], axis=2)
    save_video(combined, str(output_path), fps=fps)


def save_rollout_frames_grid(
    gt_frames: list[np.ndarray],
    pred_frames: list[np.ndarray],
    output_path: Path,
    n_snapshots: int = 8,
    image_scale: int = 1,
) -> None:
    """Static PNG grid: top row = GT frames, bottom row = rollout predictions,
    at `n_snapshots` evenly-spaced timesteps. Fast way to eyeball rollout
    quality without playing the mp4."""
    n_total = min(len(gt_frames), len(pred_frames))
    if n_total == 0:
        return
    n = min(int(n_snapshots), n_total)
    if n <= 1:
        idxs = [0]
    else:
        idxs = np.linspace(0, n_total - 1, num=n, dtype=int).tolist()

    def _pick(frames: list[np.ndarray]) -> list[np.ndarray]:
        out = [frames[i] for i in idxs]
        if int(image_scale) > 1:
            out = [upscale_uint8(f, image_scale) for f in out]
        return out

    gt_pick = _pick(gt_frames)
    pred_pick = _pick(pred_frames)
    h, w = gt_pick[0].shape[:2]
    sep_w = max(2, 2 * max(1, int(image_scale)))

    def _row(frames: list[np.ndarray]) -> np.ndarray:
        sep_v = np.full((h, sep_w, 3), 180, dtype=np.uint8)
        pieces: list[np.ndarray] = []
        for i, img in enumerate(frames):
            pieces.append(img)
            if i < len(frames) - 1:
                pieces.append(sep_v)
        return np.concatenate(pieces, axis=1)

    gt_row = _row(gt_pick)
    pred_row = _row(pred_pick)

    # Horizontal separator between GT row and pred row.
    h_sep = np.full((sep_w, gt_row.shape[1], 3), 180, dtype=np.uint8)
    grid = np.concatenate([gt_row, h_sep, pred_row], axis=0)

    # Add a thin caption strip at the top and bottom (white background, dark
    # text) labelling the row meaning and the step indices.
    from PIL import Image as _Img, ImageDraw as _Draw
    pil = _Img.fromarray(grid)
    label_h = max(14, 10 * max(1, int(image_scale)))
    canvas = _Img.new(
        "RGB", (pil.width, pil.height + 2 * label_h), color=(255, 255, 255)
    )
    canvas.paste(pil, (0, label_h))
    draw = _Draw.Draw(canvas)
    # Top caption: step indices
    step_labels = "  |  ".join(f"step {i}" for i in idxs)
    draw.text((4, 2), f"GT (top) vs Rollout Pred (bottom)   |   {step_labels}",
              fill=(0, 0, 0))
    # Bottom caption: GT/Pred row markers
    draw.text((4, pil.height + label_h + 2),
              "GT is top row,  Rollout Pred is bottom row",
              fill=(80, 80, 80))
    canvas.save(output_path)


def save_rollout_mse_plot(
    step_mse: list[float],
    output_path: Path,
    title: str | None = None,
) -> None:
    """Matplotlib line plot of rollout step MSE vs step index. Also marks
    the first/avg/last values for quick reading."""
    if not step_mse:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xs = np.arange(len(step_mse))
    ys = np.asarray(step_mse, dtype=np.float64)

    fig, ax = plt.subplots(figsize=(6.4, 3.6), dpi=120)
    ax.plot(xs, ys, marker="o", markersize=3, linewidth=1.2, color="#1f77b4")
    ax.axhline(float(ys.mean()), linestyle="--", linewidth=0.8, color="#ff7f0e",
               label=f"mean = {ys.mean():.4f}")
    ax.set_xlabel("Rollout step")
    ax.set_ylabel("MSE (pred vs GT frame)")
    ax.set_title(title or "Rollout per-step MSE")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    # Annotate first and last points
    ax.annotate(f"{ys[0]:.4f}", xy=(xs[0], ys[0]),
                xytext=(4, 6), textcoords="offset points", fontsize=8)
    ax.annotate(f"{ys[-1]:.4f}", xy=(xs[-1], ys[-1]),
                xytext=(4, 6), textcoords="offset points", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def summarize_numeric(values: list[float | None]) -> float | None:
    clean = [float(x) for x in values if x is not None]
    return float(np.mean(clean)) if clean else None


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a Stage 1 checkpoint on reproducibly sampled held-out tasks."
    )
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Dataset root. If omitted, use data.root stored in the checkpoint config.",
    )
    parser.add_argument("--image-key", type=str, default=None)
    weight_group = parser.add_mutually_exclusive_group()
    weight_group.add_argument(
        "--live-weights",
        action="store_true",
        help="Force live model weights even if EMA weights exist in the checkpoint.",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes-per-task", type=int, default=5)
    parser.add_argument(
        "--n-steps",
        type=int,
        default=0,
        help="Max rollout steps per episode. Use 0 to roll out as far as possible.",
    )
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to store example images and metrics JSON for this run.",
    )
    parser.add_argument(
        "--examples-per-episode",
        type=int,
        default=DEFAULT_EXAMPLES_PER_EPISODE,
        help="Number of teacher-forcing comparison examples to save per episode.",
    )
    parser.add_argument(
        "--fid-batch-size",
        type=int,
        default=32,
        help="Batch size used when extracting Inception features for FID.",
    )
    parser.add_argument(
        "--task",
        dest="tasks",
        action="append",
        default=None,
        help="Repeatable. If omitted, uses the three default held-out tasks.",
    )
    parser.add_argument(
        "--no-amp",
        action="store_true",
        help="Disable AMP during inference.",
    )
    parser.add_argument(
        "--flip-vertical",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Flip each loaded frame vertically. LIBERO-90 HDF5 ingests store agentview "
            "upside down; held-out tasks come from that source, so flipping is on by default."
        ),
    )
    parser.add_argument(
        "--image-scale",
        type=int,
        default=2,
        help=(
            "Integer upscale factor (nearest-neighbor) applied when saving PNGs and rollout "
            "videos. Use 1 to save at native model resolution."
        ),
    )
    parser.add_argument(
        "--grid-snapshots",
        type=int,
        default=8,
        help=(
            "Number of evenly-spaced frames sampled from the rollout to build the "
            "rollout_grid.png summary image (2 rows: GT and Pred). Set 0 to disable."
        ),
    )
    args = parser.parse_args()
    if args.image_scale < 1:
        parser.error("--image-scale must be >= 1")

    device = torch.device(
        args.device if (torch.cuda.is_available() or args.device != "cuda") else "cpu"
    )
    use_amp = not args.no_amp
    print(f"Device: {device}")


    model, model_cfg, train_cfg = load_model_from_checkpoint(
        args.checkpoint, device, use_ema=None
    )

    cfg_data = train_cfg.get("data", {})
    dataset_root = args.dataset or cfg_data.get("root")
    if not dataset_root:
        raise ValueError("Dataset root is missing. Pass --dataset or save data.root in the checkpoint.")
    image_key = args.image_key or cfg_data.get("image_key", "image")
    frame_gap = int(cfg_data.get("frame_gap", 1))
    use_goal = bool(cfg_data.get("use_goal", True))
    tasks = list(args.tasks) if args.tasks else list(DEFAULT_TASKS)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = load_dataset_metadata(dataset_root)
    sampled = sample_task_episodes(
        metadata["task_to_episodes"],
        tasks,
        episodes_per_task=int(args.episodes_per_task),
        seed=int(args.seed),
    )

    sampled_payload = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "dataset_root": str(Path(dataset_root).resolve()),
        "seed": int(args.seed),
        "episodes_per_task": int(args.episodes_per_task),
        "image_key": image_key,
        "frame_gap": frame_gap,
        "frames_in": int(model_cfg.frames_in),
        "frames_out": int(model_cfg.frames_out),
        "use_goal": use_goal,
        "tasks": tasks,
        "sampled_episodes": sampled,
    }
    sampled_path = output_dir / "sampled_episodes.json"
    with sampled_path.open("w") as f:
        json.dump(sampled_payload, f, indent=2)

    print(f"Dataset root: {dataset_root}")
    print(f"Image key   : {image_key}")
    print(f"Frame gap   : {frame_gap}")
    print(f"Use goal    : {use_goal}")
    print(f"Seed        : {args.seed}")
    print(f"Saved sample manifest to: {sampled_path}")
    for task in tasks:
        print(f"- {task}")
        print(f"  episodes: {sampled[task]}")

    transform = build_transform(int(model_cfg.resolution))
    results: list[dict] = []
    fid_extractor = InceptionFeatureExtractor(device=device)
    overall_fid = FIDAccumulator()
    task_fid = {task: FIDAccumulator() for task in tasks}

    total_episodes = sum(len(ep_list) for ep_list in sampled.values())
    progress = tqdm(total=total_episodes, desc="Evaluating episodes", unit="episode")

    try:
        for task_idx, task in enumerate(tasks, start=1):
            task_slug = f"task{task_idx:02d}_{slugify(task)}"
            task_dir = output_dir / task_slug
            task_dir.mkdir(parents=True, exist_ok=True)

            for episode_idx in sampled[task]:
                episode_dir = task_dir / f"episode_{episode_idx:06d}"
                episode_dir.mkdir(parents=True, exist_ok=True)
                frames = load_episode_frames(
                    metadata["root"],
                    metadata["chunks_size"],
                    episode_idx,
                    image_key,
                    transform,
                    flip_vertical=bool(args.flip_vertical),
                )

                tf_metrics = evaluate_teacher_forcing_episode(
                    model=model,
                    frames=frames,
                    frames_in=int(model_cfg.frames_in),
                    frames_out=int(model_cfg.frames_out),
                    frame_gap=frame_gap,
                    use_goal=use_goal,
                    device=device,
                    use_amp=use_amp,
                    task_index=task_idx,
                    episode_idx=episode_idx,
                    example_seed=int(args.seed),
                    example_count=int(args.examples_per_episode),
                    episode_dir=episode_dir,
                    image_scale=int(args.image_scale),
                )

                fid_pred_frames = tf_metrics.pop("fid_pred_frames")
                fid_target_frames = tf_metrics.pop("fid_target_frames")
                if fid_pred_frames is not None and fid_target_frames is not None:
                    real_features = fid_extractor.extract(
                        fid_target_frames, batch_size=int(args.fid_batch_size)
                    )
                    fake_features = fid_extractor.extract(
                        fid_pred_frames, batch_size=int(args.fid_batch_size)
                    )
                    task_fid[task].update_features(real_features, fake_features)
                    overall_fid.update_features(real_features, fake_features)

                rollout_metrics = rollout_episode(
                    model=model,
                    frames=frames,
                    frame_gap=frame_gap,
                    use_goal=use_goal,
                    device=device,
                    use_amp=use_amp,
                    max_steps=int(args.n_steps),
                )

                video_path = episode_dir / "rollout.mp4"
                grid_path: Path | None = None
                mse_plot_path: Path | None = None
                if rollout_metrics["n_steps"] > 0:
                    save_rollout_comparison(
                        rollout_metrics["gt_frames"],
                        rollout_metrics["pred_frames"],
                        video_path,
                        fps=int(args.fps),
                        image_scale=int(args.image_scale),
                    )
                    # Static snapshot grid — quick visual summary of the whole
                    # rollout without playing the video.
                    grid_path = episode_dir / "rollout_grid.png"
                    save_rollout_frames_grid(
                        rollout_metrics["gt_frames"],
                        rollout_metrics["pred_frames"],
                        grid_path,
                        n_snapshots=int(args.grid_snapshots),
                        image_scale=int(args.image_scale),
                    )
                    # Per-step MSE curve plot.
                    mse_plot_path = episode_dir / "rollout_mse.png"
                    save_rollout_mse_plot(
                        rollout_metrics.get("step_mse", []),
                        mse_plot_path,
                        title=f"{task_slug}  ep={episode_idx}  "
                              f"avg_mse={rollout_metrics.get('avg_mse') or float('nan'):.4f}",
                    )
                else:
                    video_path = None

                episode_result = {
                    "task": task,
                    "task_dir": task_slug,
                    "episode_index": int(episode_idx),
                    "episode_length": int(len(frames)),
                    "teacher_forcing": tf_metrics,
                    "rollout": {
                        "n_steps": int(rollout_metrics["n_steps"]),
                        "avg_mse": rollout_metrics["avg_mse"],
                        "first_mse": rollout_metrics["first_mse"],
                        "last_mse": rollout_metrics["last_mse"],
                        "step_mse": rollout_metrics.get("step_mse", []),
                        "video_path": str(video_path.resolve()) if video_path else None,
                        "grid_path": str(grid_path.resolve()) if grid_path else None,
                        "mse_plot_path": str(mse_plot_path.resolve()) if mse_plot_path else None,
                    },
                }
                results.append(episode_result)

                episode_metrics_path = episode_dir / "metrics.json"
                with episode_metrics_path.open("w") as f:
                    json.dump(episode_result, f, indent=2)

                print(
                    f"[{task_idx}/{len(tasks)}] episode {episode_idx}: "
                    f"TF={tf_metrics['avg_mse'] if tf_metrics['avg_mse'] is not None else 'NA'} "
                    f"| rollout={rollout_metrics['avg_mse'] if rollout_metrics['avg_mse'] is not None else 'NA'}"
                )
                progress.update(1)
    finally:
        progress.close()

    task_summaries: list[dict] = []
    for task in tasks:
        task_rows = [row for row in results if row["task"] == task]
        task_summaries.append(
            {
                "task": task,
                "n_episodes": len(task_rows),
                "sampled_episodes": [row["episode_index"] for row in task_rows],
                "teacher_forcing_avg_mse": summarize_numeric(
                    [row["teacher_forcing"]["avg_mse"] for row in task_rows]
                ),
                "teacher_forcing_fid": task_fid[task].compute(),
                "rollout_avg_mse": summarize_numeric(
                    [row["rollout"]["avg_mse"] for row in task_rows]
                ),
                "rollout_first_mse": summarize_numeric(
                    [row["rollout"]["first_mse"] for row in task_rows]
                ),
                "rollout_last_mse": summarize_numeric(
                    [row["rollout"]["last_mse"] for row in task_rows]
                ),
            }
        )

    summary = {
        **sampled_payload,
        "results": results,
        "task_summaries": task_summaries,
        "overall": {
            "n_tasks": len(tasks),
            "n_episodes": len(results),
            "teacher_forcing_avg_mse": summarize_numeric(
                [row["teacher_forcing"]["avg_mse"] for row in results]
            ),
            "teacher_forcing_fid": overall_fid.compute(),
            "rollout_avg_mse": summarize_numeric(
                [row["rollout"]["avg_mse"] for row in results]
            ),
            "rollout_first_mse": summarize_numeric(
                [row["rollout"]["first_mse"] for row in results]
            ),
            "rollout_last_mse": summarize_numeric(
                [row["rollout"]["last_mse"] for row in results]
            ),
        },
    }

    summary_path = output_dir / "summary.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)

    print("\nPer-task summary")
    for row in task_summaries:
        print(f"- {row['task']}")
        print(
            "  "
            + f"episodes={row['sampled_episodes']}, "
            + f"tf_mse={row['teacher_forcing_avg_mse']}, "
            + f"tf_fid={row['teacher_forcing_fid']}, "
            + f"rollout_mse={row['rollout_avg_mse']}"
        )

    print("\nOverall")
    print(f"- teacher-forcing avg MSE: {summary['overall']['teacher_forcing_avg_mse']}")
    print(f"- teacher-forcing FID    : {summary['overall']['teacher_forcing_fid']}")
    print(f"- rollout avg MSE        : {summary['overall']['rollout_avg_mse']}")
    print(f"- summary json           : {summary_path}")


if __name__ == "__main__":
    main()
