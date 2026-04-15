from __future__ import annotations

import copy
import io
import json
import logging
import math
import os
import sys
import time
import warnings
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn as nn
import wandb
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torchvision import transforms
from tqdm import tqdm

from cosmos_model import (
    ARPatchConfig as _CosmosConfig,
    ARVideoPatchTransformer as _CosmosModel,
)
from idm_model import (
    InverseDynamicsModel as _IDMModel,
    InverseDynamicsModelDP as _IDMModelDP,
)

OmegaConf.register_new_resolver("if", lambda cond, t, f: t if cond else f, replace=True)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Logging / warning / wandb silence helpers
# ---------------------------------------------------------------------------


def _is_rank0() -> bool:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank() == 0
    return int(os.environ.get("RANK", "0")) == 0


def configure_logging_for_ddp(is_main: bool) -> None:
    """
    Make DDP training quiet:
    - rank0: normal INFO logging
    - non-rank0: errors only
    """
    root = logging.getLogger()

    if is_main:
        if not root.handlers:
            logging.basicConfig(
                level=logging.INFO,
                format="[%(asctime)s][%(name)s][%(levelname)s] - %(message)s",
            )
        else:
            root.setLevel(logging.INFO)
            for h in root.handlers:
                h.setLevel(logging.INFO)
        log.setLevel(logging.INFO)
        warnings.filterwarnings("default")
    else:
        # silence warnings on non-main ranks
        warnings.filterwarnings("ignore")

        # remove all existing handlers and replace with ERROR-only handler
        for h in list(root.handlers):
            root.removeHandler(h)

        handler = logging.StreamHandler(sys.stderr)
        handler.setLevel(logging.ERROR)
        handler.setFormatter(
            logging.Formatter("[%(asctime)s][%(name)s][%(levelname)s] - %(message)s")
        )
        root.addHandler(handler)
        root.setLevel(logging.ERROR)

        # silence common noisy libraries
        logging.getLogger(__name__).setLevel(logging.ERROR)
        logging.getLogger("wandb").setLevel(logging.ERROR)
        logging.getLogger("urllib3").setLevel(logging.ERROR)
        logging.getLogger("matplotlib").setLevel(logging.ERROR)
        logging.getLogger("PIL").setLevel(logging.ERROR)
        logging.getLogger("torch").setLevel(logging.ERROR)
        logging.getLogger("torch.distributed").setLevel(logging.ERROR)
        logging.getLogger("torch._dynamo").setLevel(logging.ERROR)
        logging.getLogger("torch._inductor").setLevel(logging.ERROR)
        logging.getLogger("hydra").setLevel(logging.ERROR)

        # reduce external noise
        os.environ["WANDB_SILENT"] = "true"
        os.environ["PYTHONWARNINGS"] = "ignore"


# ---------------------------------------------------------------------------
# Checkpoint state_dict helpers
# ---------------------------------------------------------------------------


def _strip_prefix(sd: dict, prefix: str) -> dict:
    if any(k.startswith(prefix) for k in sd):
        return {k.removeprefix(prefix): v for k, v in sd.items()}
    return sd


def _clean_state_dict(sd: dict) -> dict:
    sd = _strip_prefix(sd, "_orig_mod.")
    sd = _strip_prefix(sd, "module.")
    sd = _strip_prefix(sd, "_orig_mod.")
    return sd


# ---------------------------------------------------------------------------
# NCCL environment hardening
# ---------------------------------------------------------------------------


def _set_nccl_env_defaults():
    defaults = {
        "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
        "NCCL_IB_TIMEOUT": "23",
        "NCCL_IB_RETRY_CNT": "7",
    }
    for key, value in defaults.items():
        if key not in os.environ:
            os.environ[key] = value


# ---------------------------------------------------------------------------
# DDP helpers
# ---------------------------------------------------------------------------


def setup_ddp(nccl_timeout_min: int = 30):
    if "RANK" not in os.environ:
        return 0, 0, 1

    _set_nccl_env_defaults()

    dist.init_process_group(
        backend="nccl",
        timeout=timedelta(minutes=nccl_timeout_min),
    )
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size


def cleanup_ddp():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def ddp_enabled() -> bool:
    return dist.is_available() and dist.is_initialized()


def ddp_barrier() -> None:
    if ddp_enabled():
        dist.barrier()


def ddp_all_reduce_bool(flag: bool, op: str = "or") -> bool:
    if not ddp_enabled():
        return flag
    t = torch.tensor(
        1 if flag else 0,
        device=(
            torch.device("cuda", torch.cuda.current_device())
            if torch.cuda.is_available()
            else torch.device("cpu")
        ),
        dtype=torch.int32,
    )
    if op == "or":
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
    elif op == "and":
        dist.all_reduce(t, op=dist.ReduceOp.MIN)
    else:
        raise ValueError(f"Unsupported reduce op for bool sync: {op}")
    return bool(t.item())


def ddp_all_reduce_scalar(value: float, op=dist.ReduceOp.AVG) -> float:
    if not ddp_enabled():
        return float(value)
    t = torch.tensor(
        float(value),
        device=(
            torch.device("cuda", torch.cuda.current_device())
            if torch.cuda.is_available()
            else torch.device("cpu")
        ),
        dtype=torch.float32,
    )
    dist.all_reduce(t, op=op)
    return float(t.item())


def unwrap_model(model: nn.Module) -> nn.Module:
    if hasattr(model, "_orig_mod"):
        model = model._orig_mod
    if isinstance(model, DDP):
        model = model.module
    if hasattr(model, "_orig_mod"):
        model = model._orig_mod
    return model


def resolve_amp_dtype(amp_dtype: str) -> torch.dtype:
    amp_dtype = str(amp_dtype).lower()
    if amp_dtype == "fp16":
        return torch.float16
    if amp_dtype == "bf16":
        return torch.bfloat16
    raise ValueError(f"Unsupported train.amp_dtype: {amp_dtype!r}")


# ---------------------------------------------------------------------------
# EMA
# ---------------------------------------------------------------------------


class EMA:
    def __init__(
        self,
        model: nn.Module,
        decay: float | None = None,
        *,
        update_after_step: int = 0,
        inv_gamma: float = 1.0,
        power: float = 0.75,
        min_value: float = 0.0,
        max_value: float = 0.9999,
    ):
        self.decay = decay
        self.update_after_step = int(update_after_step)
        self.inv_gamma = float(inv_gamma)
        self.power = float(power)
        self.min_value = float(min_value)
        self.max_value = float(max_value)
        self.num_updates = 0
        self.shadow = copy.deepcopy(model)
        self.shadow.eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    def _get_decay(self) -> float:
        if self.decay is not None:
            return min(self.decay, (1.0 + self.num_updates) / (10.0 + self.num_updates))

        step = max(int(self.num_updates) - self.update_after_step - 1, 0)
        if step <= 0:
            return 0.0
        value = 1.0 - (1.0 + step / self.inv_gamma) ** (-self.power)
        return min(max(value, self.min_value), self.max_value)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        d = self._get_decay()
        self.num_updates += 1
        for s, m in zip(self.shadow.parameters(), model.parameters()):
            s.data.lerp_(m.data, 1.0 - d)
        for s_buf, m_buf in zip(self.shadow.buffers(), model.buffers()):
            s_buf.data.copy_(m_buf.data)

    def apply(self, model: nn.Module) -> None:
        self._backup = [p.data.clone() for p in model.parameters()]
        for m, s in zip(model.parameters(), self.shadow.parameters()):
            m.data.copy_(s.data)

    def restore(self, model: nn.Module) -> None:
        for m, b in zip(model.parameters(), self._backup):
            m.data.copy_(b)
        del self._backup

    def state_dict(self) -> dict:
        return {
            "shadow": self.shadow.state_dict(),
            "num_updates": self.num_updates,
        }

    def load_state_dict(self, state: dict) -> None:
        if "shadow" in state:
            self.shadow.load_state_dict(state["shadow"])
            self.num_updates = int(state.get("num_updates", 0))
            return
        self.shadow.load_state_dict(state)
        self.num_updates = 0


# ---------------------------------------------------------------------------
# Optimizer & Scheduler
# ---------------------------------------------------------------------------


def build_optimizer(
    model: nn.Module, opt_cfg: DictConfig, is_main: bool
) -> torch.optim.Optimizer:
    eps = float(opt_cfg.get("eps", 1.0e-8))
    if not opt_cfg.no_decay_norm:
        return torch.optim.AdamW(
            model.parameters(),
            lr=opt_cfg.lr,
            weight_decay=opt_cfg.weight_decay,
            betas=tuple(opt_cfg.betas),
            eps=eps,
        )

    no_decay_types = (nn.LayerNorm, nn.GroupNorm, nn.Embedding)
    decay_params, no_decay_params = [], []
    for module in model.modules():
        for param_name, param in module.named_parameters(recurse=False):
            if not param.requires_grad:
                continue
            if isinstance(module, no_decay_types) or param_name == "bias":
                no_decay_params.append(param)
            else:
                decay_params.append(param)

    if is_main:
        log.info(
            f"Optimizer groups — decay: {len(decay_params)} | no_decay: {len(no_decay_params)}"
        )

    return torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": opt_cfg.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=opt_cfg.lr,
        betas=tuple(opt_cfg.betas),
        eps=eps,
    )


def build_scheduler(
    optimizer,
    sched_cfg,
    base_lr,
    total_steps,
    is_main: bool,
    *,
    sched_type_override: str | None = None,
    num_warmup_steps: int | None = None,
):
    sched_type = str(
        sched_type_override if sched_type_override is not None else sched_cfg.get("type", "cosine")
    ).lower()
    min_lr = float(sched_cfg.get("min_lr", 0.0))

    total_steps = max(1, int(total_steps))
    if num_warmup_steps is not None:
        warmup_steps = min(total_steps - 1, max(int(num_warmup_steps), 0)) if total_steps > 1 else 0
    else:
        warmup_fraction = float(sched_cfg.get("warmup_fraction", 0.0))
        warmup_steps = (
            min(total_steps - 1, int(total_steps * warmup_fraction))
            if total_steps > 1
            else 0
        )
    min_lr_ratio = min_lr / base_lr if base_lr > 0 else 0.0

    if sched_type != "cosine":
        raise ValueError(f"Unsupported scheduler type: {sched_type!r}")

    def lr_lambda(step):
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        if total_steps <= warmup_steps + 1:
            return 1.0
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps - 1)
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    scheduler = LambdaLR(optimizer, lr_lambda)
    if is_main:
        log.info(
            f"Scheduler — type={sched_type}, total={total_steps}, "
            f"warmup={warmup_steps}, min_lr={min_lr:.3e}"
        )
    return scheduler


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------


class VideoFrameDataset(Dataset):
    _CACHE_MAX = 64

    def __init__(
        self,
        data_cfg: DictConfig,
        model_cfg: DictConfig,
        split: str,
        is_main: bool = True,
    ):
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Unknown split: {split!r}")

        self.is_main = is_main
        self.fin = model_cfg.frames_in
        self.fout = model_cfg.frames_out
        self.gap = int(data_cfg.frame_gap)
        self.image_key = data_cfg.get("image_key", "image")
        self.action_key = data_cfg.get("action_key", "actions")
        self.use_goal = data_cfg.get("use_goal", True)
        self._res = model_cfg.resolution
        self.transform = transforms.Compose(
            [
                transforms.Resize((model_cfg.resolution, model_cfg.resolution)),
                transforms.ToTensor(),
                transforms.Normalize([0.5] * 3, [0.5] * 3),
            ]
        )

        root = Path(data_cfg.root)
        self._data_root = root

        with open(root / "meta" / "info.json") as f:
            info = json.load(f)
        total_episodes = info["total_episodes"]
        chunks_size = info["chunks_size"]

        task_names = []
        tasks_path = root / "meta" / "tasks.jsonl"
        if tasks_path.exists():
            with open(tasks_path) as f:
                for line in f:
                    rec = json.loads(line)
                    if "task" in rec:
                        task_names.append(rec["task"])

        episode_files = [
            root
            / "data"
            / f"chunk-{ep_idx // chunks_size:03d}"
            / f"episode_{ep_idx:06d}.parquet"
            for ep_idx in range(total_episodes)
        ]

        episode_meta = []
        with open(root / "meta" / "episodes.jsonl") as f:
            for line in f:
                rec = json.loads(line)
                tasks = rec.get("tasks", [])
                episode_meta.append(
                    {
                        "episode_index": int(rec["episode_index"]),
                        "length": int(rec["length"]),
                        "task": tasks[0] if tasks else None,
                    }
                )

        if not task_names:
            seen = set()
            for rec in episode_meta:
                t = rec["task"]
                if t and t not in seen:
                    seen.add(t)
                    task_names.append(t)

        test_tasks_meta_path = root / "meta" / "test_tasks.json"
        configured_test_tasks = list(data_cfg.get("test_tasks", []))
        if not configured_test_tasks and test_tasks_meta_path.exists():
            with open(test_tasks_meta_path) as f:
                stored = json.load(f)
            configured_test_tasks = list(
                stored.get("tasks", []) if isinstance(stored, dict) else stored
            )

        test_task_count = int(data_cfg.get("test_task_count", 0))
        if configured_test_tasks:
            unknown = sorted(set(configured_test_tasks) - set(task_names))
            if unknown:
                raise ValueError("Missing test tasks: " + ", ".join(unknown))
            selected_test_tasks = configured_test_tasks
        elif test_task_count > 0:
            if test_task_count >= len(task_names):
                raise ValueError(
                    f"test_task_count={test_task_count} >= total tasks ({len(task_names)})"
                )
            selected_test_tasks = task_names[:test_task_count]
        else:
            selected_test_tasks = []

        self.test_tasks = tuple(selected_test_tasks)
        test_task_set = set(self.test_tasks)

        in_domain_eps = [
            r["episode_index"] for r in episode_meta if r["task"] not in test_task_set
        ]
        test_eps = [
            r["episode_index"] for r in episode_meta if r["task"] in test_task_set
        ]
        ep_lengths = {r["episode_index"]: r["length"] for r in episode_meta}
        ep_tasks = {r["episode_index"]: r["task"] for r in episode_meta}

        ep_indices = in_domain_eps if split == "train" else test_eps
        self.episode_indices = list(ep_indices)
        self.episode_files = {ep_idx: str(episode_files[ep_idx]) for ep_idx in range(total_episodes)}

        self.samples = []
        span = self.fin + self.gap + self.fout - 1
        self._target_offset = self.fin - 1 + self.gap
        self._action_offset = self.fin - 1

        for ep_idx in ep_indices:
            length = ep_lengths[ep_idx]
            if length < span:
                continue
            ep_path_str = str(episode_files[ep_idx])
            for start in range(length - span + 1):
                self.samples.append((ep_path_str, start, length))

        split_tasks = sorted(
            {ep_tasks[i] for i in ep_indices if ep_tasks[i] is not None}
        )
        if self.test_tasks and split == "train" and self.is_main:
            log.info(
                f"Held-out test tasks ({len(self.test_tasks)}): "
                + "; ".join(self.test_tasks)
            )
        if self.is_main:
            log.info(
                f"[{split}] {len(self.samples)} windows / {len(ep_indices)} episodes / {len(split_tasks)} tasks"
            )

        self._parquet_cache: dict[str, pd.DataFrame] = {}
        self._action_stats_cache: dict[str, np.ndarray] | None = None

    def _read_parquet(self, path: str) -> pd.DataFrame:
        if path in self._parquet_cache:
            return self._parquet_cache[path]
        if len(self._parquet_cache) >= self._CACHE_MAX:
            oldest = next(iter(self._parquet_cache))
            del self._parquet_cache[oldest]
        df = pd.read_parquet(path, columns=[self.image_key, self.action_key])
        self._parquet_cache[path] = df
        return df

    def __len__(self):
        return len(self.samples)

    def get_action_stats(self) -> dict[str, np.ndarray]:
        if self._action_stats_cache is not None:
            return {k: v.copy() for k, v in self._action_stats_cache.items()}

        # Try loading from disk cache
        cache_path = self._data_root / "meta" / "action_stats.json"
        if cache_path.exists():
            if self.is_main:
                log.info(f"Loading cached action stats from {cache_path}")
            with open(cache_path) as f:
                saved = json.load(f)
            self._action_stats_cache = {
                k: np.array(v, dtype=np.float32) for k, v in saved.items()
            }
            return {k: v.copy() for k, v in self._action_stats_cache.items()}

        action_min = None
        action_max = None
        action_sum = None
        action_sq_sum = None
        count = 0

        for ep_idx in tqdm(self.episode_indices, desc="Computing action stats", disable=not self.is_main):
            path = self.episode_files[ep_idx]
            df = self._read_parquet(path)
            actions = np.stack(
                [np.array(value, copy=False) for value in df[self.action_key].tolist()],
                axis=0,
            ).astype(np.float64, copy=False)

            batch_min = actions.min(axis=0)
            batch_max = actions.max(axis=0)
            batch_sum = actions.sum(axis=0)
            batch_sq_sum = np.square(actions).sum(axis=0)

            action_min = batch_min if action_min is None else np.minimum(action_min, batch_min)
            action_max = batch_max if action_max is None else np.maximum(action_max, batch_max)
            action_sum = batch_sum if action_sum is None else (action_sum + batch_sum)
            action_sq_sum = (
                batch_sq_sum if action_sq_sum is None else (action_sq_sum + batch_sq_sum)
            )
            count += actions.shape[0]

        if count == 0:
            raise RuntimeError("Cannot compute action stats from an empty dataset.")

        mean = action_sum / count
        var = np.maximum(action_sq_sum / count - np.square(mean), 0.0)
        std = np.sqrt(var)
        self._action_stats_cache = {
            "min": action_min.astype(np.float32),
            "max": action_max.astype(np.float32),
            "mean": mean.astype(np.float32),
            "std": std.astype(np.float32),
        }

        # Save to disk for next run
        if self.is_main:
            saved = {k: v.tolist() for k, v in self._action_stats_cache.items()}
            with open(cache_path, "w") as f:
                json.dump(saved, f)
            log.info(f"Saved action stats cache to {cache_path}")

        return {k: v.copy() for k, v in self._action_stats_cache.items()}

    def __getitem__(self, idx):
        parquet_path, start, ep_length = self.samples[idx]
        df = self._read_parquet(parquet_path)

        ctx_imgs = []
        for t in range(self.fin):
            img_data = df.iloc[start + t][self.image_key]
            img = Image.open(io.BytesIO(img_data["bytes"])).convert("RGB")
            ctx_imgs.append(self.transform(img))

        tgt_imgs = []
        for t in range(self.fout):
            frame_idx = start + self._target_offset + t
            img_data = df.iloc[frame_idx][self.image_key]
            img = Image.open(io.BytesIO(img_data["bytes"])).convert("RGB")
            tgt_imgs.append(self.transform(img))

        act_start = start + self._action_offset
        actions = torch.stack(
            [
                torch.tensor(
                    np.array(df.iloc[act_start + i][self.action_key], copy=True),
                    dtype=torch.float32,
                )
                for i in range(self.gap)
            ]
        )

        if self.use_goal:
            goal_data = df.iloc[ep_length - 1][self.image_key]
            goal = Image.open(io.BytesIO(goal_data["bytes"])).convert("RGB")
            goal = self.transform(goal)
        else:
            goal = torch.zeros(3, self._res, self._res)

        return torch.stack(ctx_imgs), torch.stack(tgt_imgs), actions, goal


class SyntheticVideoDataset(Dataset):
    def __init__(self, data_cfg, model_cfg, split):
        self.T = model_cfg.frames_in + model_cfg.frames_out
        self.fin = model_cfg.frames_in
        self.cfg = model_cfg
        self.moving_dot = data_cfg.moving_dot
        self.n = data_cfg.n_train if split == "train" else data_cfg.n_val
        self.gap = data_cfg.get("frame_gap", 1)
        self.action_dim = data_cfg.get("action_dim", 7)
        self.use_goal = data_cfg.get("use_goal", True)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        C, H, W, T = (
            self.cfg.num_channels,
            self.cfg.resolution,
            self.cfg.resolution,
            self.T,
        )
        if self.moving_dot:
            frames = torch.full((T, C, H, W), -0.9)
            speed = torch.randint(2, 6, (1,)).item()
            x0 = torch.randint(0, W // 2, (1,)).item()
            y0 = torch.randint(0, H // 2, (1,)).item()
            for t in range(T):
                cx = min(x0 + t * speed, W - 5)
                cy = min(y0 + t * speed, H - 5)
                frames[t, :, cy : cy + 5, cx : cx + 5] = 0.9
        else:
            frames = torch.rand(T, C, H, W) * 2 - 1
        actions = torch.zeros(self.gap, self.action_dim)
        goal = frames[-1].clone() if self.use_goal else torch.zeros(C, H, W)
        return frames[: self.fin], frames[self.fin :], actions, goal


def build_datasets(data_cfg, model_cfg, is_main: bool):
    cls = {"real": VideoFrameDataset, "synthetic": SyntheticVideoDataset}
    if data_cfg.type not in cls:
        raise ValueError(f"Unknown data.type: {data_cfg.type!r}")

    if data_cfg.type == "real":
        train_ds = VideoFrameDataset(data_cfg, model_cfg, "train", is_main=is_main)
        val_ds = VideoFrameDataset(data_cfg, model_cfg, "val", is_main=is_main)
    else:
        train_ds = SyntheticVideoDataset(data_cfg, model_cfg, "train")
        val_ds = SyntheticVideoDataset(data_cfg, model_cfg, "val")

    return train_ds, val_ds, None


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


class Trainer:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.seed = int(
            OmegaConf.select(cfg, "seed", default=OmegaConf.select(cfg, "training.seed", default=42))
        )

        nccl_timeout_min = int(cfg.train.get("nccl_timeout_min", 30))
        self.rank, self.local_rank, self.world_size = setup_ddp(
            nccl_timeout_min=nccl_timeout_min
        )
        self.is_main = self.rank == 0

        configure_logging_for_ddp(self.is_main)

        if torch.cuda.is_available():
            self.device = torch.device(f"cuda:{self.local_rank}")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        if self.is_main:
            log.info(f"Device: {self.device} | world_size: {self.world_size}")
            log.info(
                f"NCCL env: ASYNC_ERROR_HANDLING="
                f"{os.environ.get('TORCH_NCCL_ASYNC_ERROR_HANDLING', 'unset')}, "
                f"IB_TIMEOUT={os.environ.get('NCCL_IB_TIMEOUT', 'unset')}, "
                f"IB_RETRY_CNT={os.environ.get('NCCL_IB_RETRY_CNT', 'unset')}"
            )

        if self.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True

        torch.manual_seed(self.seed + self.rank)

        self.grad_accum_steps = int(
            OmegaConf.select(
                cfg,
                "train.gradient_accumulate_every",
                default=cfg.train.get("grad_accum_steps", 1),
            )
        )
        self.current_epoch = 0
        self.compile_enabled = False
        self.idm_type = str(cfg.train.get("idm_type", "mlp")).lower()
        self.num_epochs = int(
            OmegaConf.select(cfg, "training.num_epochs", default=cfg.train.get("epochs", 1))
        )
        self.train_batch_size = int(
            OmegaConf.select(cfg, "dataloader.batch_size", default=cfg.train.get("batch_size", 1))
        )
        self.val_batch_size = int(
            OmegaConf.select(
                cfg,
                "val_dataloader.batch_size",
                default=self.train_batch_size * 2,
            )
        )
        self.val_every = int(
            OmegaConf.select(
                cfg,
                "train.val_every",
                default=OmegaConf.select(cfg, "training.val_every", default=1),
            )
        )
        self.sample_every = int(
            OmegaConf.select(
                cfg,
                "train.sample_every",
                default=OmegaConf.select(cfg, "training.sample_every", default=1),
            )
        )
        self.checkpoint_every = int(
            OmegaConf.select(
                cfg,
                "train.checkpoint_every",
                default=OmegaConf.select(cfg, "training.checkpoint_every", default=1),
            )
        )
        self.max_train_steps = OmegaConf.select(
            cfg,
            "train.max_train_steps",
            default=OmegaConf.select(cfg, "training.max_train_steps", default=None),
        )
        self.max_val_steps = OmegaConf.select(
            cfg,
            "train.max_val_steps",
            default=OmegaConf.select(cfg, "training.max_val_steps", default=None),
        )
        self.ema_update_every = 1

        if self.is_main:
            eff = self.train_batch_size * self.grad_accum_steps * self.world_size
            log.info(
                f"micro_batch={self.train_batch_size} | accum={self.grad_accum_steps} | "
                f"world={self.world_size} | eff_batch={eff}"
            )

        # --- Model ---
        mcfg = cfg.model
        self.stage = int(cfg.train.get("stage", 1))
        has_goal = bool(cfg.data.get("use_goal", True))

        config_kwargs = dict(
            resolution=mcfg.resolution,
            num_channels=mcfg.num_channels,
            patch_size=mcfg.patch_size,
            d_model=mcfg.d_model,
            n_heads=mcfg.n_heads,
            n_layers=mcfg.n_layers,
            mlp_ratio=mcfg.mlp_ratio,
            dropout=mcfg.dropout,
            frames_in=mcfg.frames_in,
            frames_out=mcfg.frames_out,
            action_dim=cfg.data.get("action_dim", 7),
        )

        config_kwargs["qk_norm"] = mcfg.get("qk_norm", True)
        config_kwargs["parallel_attn"] = mcfg.get("parallel_attn", False)
        self.model_cfg = _CosmosConfig(**config_kwargs)
        raw_model = _CosmosModel(self.model_cfg).to(self.device)

        if self.stage == 2:
            s1 = str(cfg.train.get("stage1_ckpt", ""))
            if s1:
                self._load_stage1_weights(raw_model, s1)
            if self.idm_type in {"dp", "diffusion", "diffusion_policy"}:
                raw_model = _IDMModelDP(
                    raw_model,
                    n_actions=int(cfg.data.get("frame_gap", 0)),
                    **OmegaConf.to_container(cfg.train.get("idm_dp", {}), resolve=True),
                ).to(self.device)
            else:
                raw_model = _IDMModel(
                    raw_model,
                    n_actions=int(cfg.data.get("frame_gap", 0)),
                ).to(self.device)
            raw_model.prebuild_mask(device=self.device, has_goal=has_goal)
        else:
            raw_model.prebuild_mask(device=self.device, has_goal=has_goal)

        if self.is_main:
            np_ = sum(p.numel() for p in raw_model.parameters()) / 1e6
            nt_ = (
                sum(p.numel() for p in raw_model.parameters() if p.requires_grad) / 1e6
            )
            log.info(
                f"Stage: {self.stage} | Params: {np_:.2f}M | Trainable: {nt_:.2f}M"
            )

        using_diffusion_policy_idm = (
            self.stage == 2 and self.idm_type in {"dp", "diffusion", "diffusion_policy"}
        )
        self._stage2_dp_ddp_safe_mode = self.world_size > 1 and using_diffusion_policy_idm
        if self.world_size > 1 and using_diffusion_policy_idm and hasattr(torch, "_dynamo"):
            torch._dynamo.config.optimize_ddp = False
            if self.is_main:
                log.info(
                    "Disabled TorchDynamo DDP optimizer for stage=2 diffusion_policy IDM "
                    "to avoid flex_attention compile failures under DDP."
                )

        # --- DDP ---
        if self.world_size > 1:
            self.model = DDP(
                raw_model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                broadcast_buffers=not self._stage2_dp_ddp_safe_mode,
                find_unused_parameters=bool(
                    cfg.train.get("find_unused_parameters", False)
                ),
            )
            if self.is_main:
                log.info(f"DDP enabled ({self.world_size} GPUs)")
                if self._stage2_dp_ddp_safe_mode:
                    log.info(
                        "Disabled DDP buffer broadcasts for stage=2 diffusion_policy IDM "
                        "to avoid collective mismatches from buffer sync."
                    )
        else:
            self.model = raw_model

        # --- torch.compile ---
        compile_requested = bool(cfg.train.get("compile", False))
        if compile_requested and hasattr(torch, "compile"):
            if using_diffusion_policy_idm:
                if self.is_main:
                    log.info(
                        "Skipping torch.compile for stage=2 diffusion_policy IDM "
                        "because the robomimic/diffusion_policy vision stack is not compile-safe."
                    )
            else:
                torch._dynamo.config.optimize_ddp = False
                self.model = torch.compile(self.model)
                self.compile_enabled = True
                if self.is_main:
                    log.info("torch.compile enabled")

        # --- EMA ---
        dp_ema_cfg = OmegaConf.select(cfg, "ema", default=None)
        legacy_ema_cfg = OmegaConf.select(cfg, "train.ema", default=None)
        raw = unwrap_model(self.model)
        if dp_ema_cfg is not None:
            use_ema = bool(OmegaConf.select(cfg, "training.use_ema", default=True))
            self.ema = (
                EMA(
                    raw,
                    update_after_step=int(dp_ema_cfg.update_after_step),
                    inv_gamma=float(dp_ema_cfg.inv_gamma),
                    power=float(dp_ema_cfg.power),
                    min_value=float(dp_ema_cfg.min_value),
                    max_value=float(dp_ema_cfg.max_value),
                )
                if use_ema
                else None
            )
            self.ema_update_every = 1
        else:
            self.ema = (
                EMA(raw, decay=float(legacy_ema_cfg.decay))
                if legacy_ema_cfg is not None and legacy_ema_cfg.enabled
                else None
            )
            self.ema_update_every = int(
                legacy_ema_cfg.update_every if legacy_ema_cfg is not None else 1
            )
        if self.ema and self.is_main:
            if dp_ema_cfg is not None:
                log.info(
                    "EMA enabled"
                    f" update_after_step={int(dp_ema_cfg.update_after_step)}"
                    f" inv_gamma={float(dp_ema_cfg.inv_gamma)}"
                    f" power={float(dp_ema_cfg.power)}"
                    f" max_value={float(dp_ema_cfg.max_value)}"
                )
            else:
                log.info(
                    f"EMA decay={float(legacy_ema_cfg.decay)} update_every={self.ema_update_every}"
                )

        # --- Optimizer ---
        self.optimizer = build_optimizer(
            unwrap_model(self.model), cfg.train.optimizer, self.is_main
        )

        # --- AMP ---
        self.use_amp = cfg.train.amp and self.device.type == "cuda"
        amp_dtype_name = str(cfg.train.get("amp_dtype", "fp16"))
        self.amp_dtype = resolve_amp_dtype(amp_dtype_name)
        self.use_grad_scaler = self.use_amp and self.amp_dtype == torch.float16
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_grad_scaler)
        if self.is_main:
            log.info(
                f"AMP {'on' if self.use_amp else 'off'}"
                + (
                    f" dtype={amp_dtype_name} scaler={'on' if self.use_grad_scaler else 'off'}"
                    if self.use_amp
                    else ""
                )
            )

        # --- Data ---
        self.use_goal = cfg.data.get("use_goal", True)
        train_ds, val_ds, test_ds = build_datasets(cfg.data, cfg.model, self.is_main)
        raw = unwrap_model(self.model)
        if self.stage == 2 and hasattr(raw, "set_action_stats") and hasattr(train_ds, "get_action_stats"):
            action_stats = train_ds.get_action_stats()
            raw.set_action_stats(action_stats)
            if self.ema is not None and hasattr(self.ema.shadow, "set_action_stats"):
                self.ema.shadow.set_action_stats(action_stats)
            if self.is_main:
                log.info("Loaded train action stats into stage-2 diffusion-policy IDM.")
        self.test_task_names = list(getattr(train_ds, "test_tasks", ()))
        dl_timeout = int(cfg.data.get("loader_timeout", 600))

        def _loader_kwargs(cfg_path: str, *, fallback_workers: int | None = None) -> dict:
            n_workers = int(
                OmegaConf.select(
                    cfg,
                    f"{cfg_path}.num_workers",
                    default=(
                        fallback_workers
                        if fallback_workers is not None
                        else cfg.data.num_workers
                    ),
                )
            )
            pin_memory = bool(
                OmegaConf.select(
                    cfg,
                    f"{cfg_path}.pin_memory",
                    default=cfg.data.pin_memory,
                )
            ) and self.device.type == "cuda"
            persistent_workers = bool(
                OmegaConf.select(
                    cfg,
                    f"{cfg_path}.persistent_workers",
                    default=cfg.data.get("persistent_workers", True),
                )
            ) and n_workers > 0
            prefetch_factor = (
                int(
                    OmegaConf.select(
                        cfg,
                        f"{cfg_path}.prefetch_factor",
                        default=cfg.data.get("prefetch_factor", 4),
                    )
                )
                if n_workers > 0
                else None
            )
            return dict(
                num_workers=n_workers,
                pin_memory=pin_memory,
                persistent_workers=persistent_workers,
                prefetch_factor=prefetch_factor,
                timeout=dl_timeout if n_workers > 0 else 0,
            )

        train_loader_kw = _loader_kwargs("dataloader")
        val_loader_kw = _loader_kwargs(
            "val_dataloader",
            fallback_workers=train_loader_kw["num_workers"],
        )
        if self.is_main:
            log.info(
                f"Train DataLoader: workers={train_loader_kw['num_workers']}, "
                f"persistent={train_loader_kw['persistent_workers']}, "
                f"prefetch={train_loader_kw.get('prefetch_factor', 'N/A')}, "
                f"timeout={dl_timeout}s"
            )
            log.info(
                f"Val DataLoader: workers={val_loader_kw['num_workers']}, "
                f"persistent={val_loader_kw['persistent_workers']}, "
                f"prefetch={val_loader_kw.get('prefetch_factor', 'N/A')}, "
                f"timeout={dl_timeout}s"
            )
            log.info("Creating train DataLoader...")

        if self.world_size > 1:
            train_sampler = DistributedSampler(
                train_ds, num_replicas=self.world_size, rank=self.rank, shuffle=True
            )
            val_sampler = DistributedSampler(
                val_ds, num_replicas=self.world_size, rank=self.rank, shuffle=False
            )
            self.train_loader = DataLoader(
                train_ds,
                batch_size=self.train_batch_size,
                sampler=train_sampler,
                drop_last=True,
                **train_loader_kw,
            )
            self.val_loader = DataLoader(
                val_ds,
                batch_size=self.val_batch_size,
                sampler=val_sampler,
                **val_loader_kw,
            )
            self.test_loader = None
            self._train_sampler = train_sampler
            if self.is_main:
                log.info("DataLoaders created.")
        else:
            self.train_loader = DataLoader(
                train_ds,
                batch_size=self.train_batch_size,
                shuffle=True,
                drop_last=True,
                **train_loader_kw,
            )
            self.val_loader = DataLoader(
                val_ds,
                batch_size=self.val_batch_size,
                shuffle=False,
                **val_loader_kw,
            )
            self.test_loader = None
            self._train_sampler = None

        if self.is_main and self.test_task_names:
            log.info(
                f"Test tasks ({len(self.test_task_names)}): "
                + "; ".join(self.test_task_names)
            )

        # --- Scheduler ---
        steps_per_epoch = max(1, len(self.train_loader) // self.grad_accum_steps)
        total_steps = self.num_epochs * steps_per_epoch
        self.scheduler = build_scheduler(
            self.optimizer,
            cfg.train.get("scheduler", {}),
            cfg.train.optimizer.lr,
            total_steps,
            self.is_main,
            sched_type_override=OmegaConf.select(cfg, "train.lr_scheduler", default=None),
            num_warmup_steps=OmegaConf.select(cfg, "train.lr_warmup_steps", default=None),
        )

        # --- State ---
        self.ckpt_dir = Path(cfg.train.ckpt_dir)
        if self.is_main:
            self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        self.start_epoch = 0
        self.best_val_loss = float("inf")
        self.best_ckpt_path: Path | None = None
        self.global_step = 0

        if cfg.train.resume:
            self._load_checkpoint(cfg.train.resume)

        # --- wandb ---
        self.wandb_cfg = cfg.get("wandb", None) or cfg.get("logging", {})
        self.wandb_enabled = bool(self.wandb_cfg.get("enabled", True))
        self.use_wandb = self.is_main and self.wandb_enabled
        if self.use_wandb:
            if self.is_main:
                log.info("Initializing wandb...")
            wandb_resume = bool(
                self.wandb_cfg.get("resume", bool(cfg.train.get("resume", "")))
            )
            wandb_tags = self.wandb_cfg.get("tags", None)
            wandb.init(
                project=self.wandb_cfg.get("project", "TTT-WM"),
                name=self.wandb_cfg.get("name", cfg.experiment_name),
                config=OmegaConf.to_container(cfg, resolve=True),
                mode=self.wandb_cfg.get("mode", None),
                tags=list(wandb_tags) if wandb_tags is not None else None,
                group=self.wandb_cfg.get("group", None),
                id=self.wandb_cfg.get("id", None),
                resume="allow" if wandb_resume else None,
            )
            if self.is_main:
                log.info("wandb initialized.")

    # --- Checkpointing ---

    def _save_checkpoint(self, epoch, val_loss, tag="last"):
        raw = unwrap_model(self.model)
        payload = {
            "epoch": epoch,
            "global_step": self.global_step,
            "model": raw.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "scaler": self.scaler.state_dict(),
            "val_loss": val_loss,
            "cfg": OmegaConf.to_container(self.cfg, resolve=True),
        }
        if self.ema is not None:
            payload["ema"] = self.ema.state_dict()
        path = self.ckpt_dir / f"{tag}.pt"
        torch.save(payload, path)
        if self.is_main:
            log.info(f"Saved: {path} | val_loss={val_loss:.6f}")

    def _load_checkpoint(self, path):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        unwrap_model(self.model).load_state_dict(_clean_state_dict(ckpt["model"]))
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.scheduler.load_state_dict(ckpt["scheduler"])
        self.scaler.load_state_dict(ckpt["scaler"])
        if self.ema is not None and "ema" in ckpt:
            self.ema.load_state_dict(_clean_state_dict(ckpt["ema"]))
        self.start_epoch = ckpt["epoch"] + 1
        self.global_step = ckpt.get("global_step", 0)
        self.best_val_loss = ckpt.get("val_loss", float("inf"))
        if self.is_main:
            log.info(f"Resumed from {path} | start_epoch={self.start_epoch}")

    def _load_stage1_weights(self, model, path):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        if "ema" in ckpt:
            ema_state = ckpt["ema"]
            raw_sd = ema_state["shadow"] if "shadow" in ema_state else ema_state
            weight_source = "EMA"
        else:
            raw_sd = ckpt["model"]
            weight_source = "live"
        model.load_state_dict(_clean_state_dict(raw_sd))
        if self.is_main:
            log.info(f"Loaded Stage 1 ({weight_source} weights) from {path}")

    # --- Epoch loop ---

    @staticmethod
    def _is_dataloader_timeout_error(exc: BaseException) -> bool:
        s = str(exc).lower()
        return ("timed out" in s and "dataloader" in s) or "dataloader timed out" in s

    @staticmethod
    def _is_fatal_dist_error(exc: BaseException) -> bool:
        s = str(exc)
        return any(
            kw in s for kw in ("NCCL", "CUDA error", "SIGABRT", "DistBackendError")
        )

    def _sync_fatal_flag(self, fatal: bool) -> bool:
        return ddp_all_reduce_bool(fatal, op="or")

    def _maybe_finish_wandb(self, exit_code: int | None = None):
        if self.use_wandb:
            if exit_code is None:
                wandb.finish()
            else:
                wandb.finish(exit_code=exit_code)

    def _abort_all_ranks(self, message: str, epoch: int | None = None):
        log.error(f"[Rank {self.rank}] {message}")
        if self.is_main:
            try:
                self._save_checkpoint(
                    -1 if epoch is None else epoch, float("inf"), "emergency"
                )
            except Exception:
                pass
        self._maybe_finish_wandb(exit_code=1)
        cleanup_ddp()
        raise RuntimeError(message)

    def _run_epoch(self, loader, train):
        self.model.train(train)
        total_loss, n_opt_steps = 0.0, 0

        pbar = tqdm(
            loader,
            desc=f"{'train' if train else 'val'} | epoch {self.current_epoch}",
            leave=False,
            disable=not self.is_main,
            dynamic_ncols=True,
        )

        grad_ctx = torch.enable_grad() if train else torch.no_grad()
        accum_loss, accum_step = 0.0, 0

        with grad_ctx:
            for batch_idx, (context, target, actions, goal) in enumerate(pbar):
                if batch_idx == 0 and self.is_main:
                    warmup_note = (
                        "compiling if first epoch"
                        if self.compile_enabled
                        else "no torch.compile on this run"
                    )
                    log.info(
                        f"First batch loaded ({'train' if train else 'val'}), "
                        f"running forward pass ({warmup_note})..."
                    )
                context = context.to(self.device, non_blocking=True)
                target = target.to(self.device, non_blocking=True)
                actions = actions.to(self.device, non_blocking=True)
                goal = (
                    goal.to(self.device, non_blocking=True) if self.use_goal else None
                )

                if train:
                    is_last = (accum_step == self.grad_accum_steps - 1) or (
                        batch_idx == len(loader) - 1
                    )
                    sync_ctx = (
                        nullcontext()
                        if (not isinstance(self.model, DDP) or is_last)
                        else self.model.no_sync()
                    )

                    try:
                        with sync_ctx, torch.amp.autocast(
                            "cuda", enabled=self.use_amp, dtype=self.amp_dtype
                        ):
                            if self.stage == 1:
                                _, loss = self.model(context, target, goal)
                            else:
                                _, _, loss = self.model(context, target, actions, goal=goal)
                            scaled = loss / self.grad_accum_steps
                        self.scaler.scale(scaled).backward()
                    except RuntimeError as e:
                        if self._is_fatal_dist_error(e):
                            log.error(
                                f"[Rank {self.rank}] Fatal @ step {self.global_step}: {e}"
                            )
                        raise

                    accum_loss += loss.detach().item()
                    accum_step += 1

                    if is_last:
                        if self.cfg.train.grad_clip > 0:
                            self.scaler.unscale_(self.optimizer)
                            nn.utils.clip_grad_norm_(
                                unwrap_model(self.model).parameters(),
                                self.cfg.train.grad_clip,
                            )

                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        self.optimizer.zero_grad(set_to_none=True)
                        self.scheduler.step()
                        self.global_step += 1

                        if self.ema and self.global_step % self.ema_update_every == 0:
                            self.ema.update(unwrap_model(self.model))

                        avg = accum_loss / self.grad_accum_steps
                        total_loss += avg
                        n_opt_steps += 1

                        if self.is_main:
                            lr = self.scheduler.get_last_lr()[0]

                            if self.use_wandb:
                                wandb.log(
                                    {
                                        "train/loss": avg,
                                        "train/lr": lr,
                                        "train/epoch": self.current_epoch,
                                    },
                                )

                            pbar.set_postfix(
                                loss=f"{avg:.4f}",
                                lr=f"{lr:.2e}",
                                epoch=self.current_epoch,
                            )

                        if n_opt_steps == 1 and self.is_main:
                            log.info(f"First optimizer step done (loss={avg:.4f}), training is running normally.")
                        accum_loss, accum_step = 0.0, 0

                        if self.max_train_steps is not None and (batch_idx + 1) >= int(self.max_train_steps):
                            break

                else:
                    try:
                        with torch.amp.autocast(
                            "cuda", enabled=self.use_amp, dtype=self.amp_dtype
                        ):
                            if self.stage == 1:
                                _, loss = self.model(context, target, goal)
                            else:
                                _, _, loss = self.model(context, target, actions, goal=goal)
                    except RuntimeError as e:
                        if self._is_fatal_dist_error(e):
                            log.error(f"[Rank {self.rank}] Eval error: {e}")
                        raise

                    if self.is_main:
                        pbar.set_postfix(
                            loss=f"{loss.item():.4f}", epoch=self.current_epoch
                        )

                    total_loss += loss.item()
                    n_opt_steps += 1

                    if self.max_val_steps is not None and (batch_idx + 1) >= int(self.max_val_steps):
                        break

        avg = total_loss / max(n_opt_steps, 1)
        if self.world_size > 1 and not train:
            avg = ddp_all_reduce_scalar(avg, op=dist.ReduceOp.AVG)
        return avg

    def _val_loss(self):
        return self._eval_loader_loss(self.val_loader)

    def _test_loss(self):
        return self._eval_loader_loss(self.test_loader) if self.test_loader else None

    def _eval_loader_loss(self, loader):
        raw = unwrap_model(self.model)
        if self.ema:
            self.ema.apply(raw)
        loss = self._run_epoch(loader, train=False)
        if self.ema:
            self.ema.restore(raw)
        return loss

    # --- Video logging ---

    @staticmethod
    def _frames_to_uint8(frames):
        x = (frames.float().clamp(-1, 1) * 0.5 + 0.5) * 255.0
        arr = x.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
        arr = np.rot90(arr, 2, axes=(1, 2)).copy()   # rotate 180 degrees
        return arr

    def _sync_model_before_logging(self):
        if self.world_size <= 1:
            return
        if getattr(self, "_stage2_dp_ddp_safe_mode", False):
            return
        raw = unwrap_model(self.model)
        for p in raw.parameters():
            dist.broadcast(p.data, src=0)
        for b in raw.buffers():
            dist.broadcast(b.data, src=0)

    def _sample_indices_for_logging(self, ds, n_samples: int):
        n = min(n_samples, len(ds))
        if n <= 0:
            return []
        if self.world_size > 1:
            if self.is_main:
                idx = torch.randperm(len(ds), device=self.device)[:n]
            else:
                idx = torch.empty(n, dtype=torch.long, device=self.device)
            dist.broadcast(idx, src=0)
            return idx.cpu().tolist()
        return torch.randperm(len(ds))[:n].tolist()

    @torch.no_grad()
    def _log_val_videos(self, n_samples=3, val_loss=None):
        if not self.wandb_enabled:
            return

        ddp_barrier()
        self._sync_model_before_logging()

        prev_mode = self.model.training
        self.model.eval()
        raw = unwrap_model(self.model)
        if self.ema:
            self.ema.apply(raw)

        ds = self.val_loader.dataset
        indices = self._sample_indices_for_logging(ds, n_samples)
        if not indices:
            if self.ema:
                self.ema.restore(raw)
            self.model.train(prev_mode)
            ddp_barrier()
            return

        ctx_list, tgt_list, act_list, goal_list = zip(*[ds[i] for i in indices])
        context = torch.stack(ctx_list).to(self.device)
        target = torch.stack(tgt_list).to(self.device)
        actions = torch.stack(act_list).to(self.device)
        goals = torch.stack(goal_list).to(self.device) if self.use_goal else None

        with torch.amp.autocast("cuda", enabled=self.use_amp, dtype=self.amp_dtype):
            if self.stage == 1:
                pred_frames, _ = raw(context, target, goals)
            else:
                pred_frames, _, _ = raw(context, target, actions, goal=goals)

        if self.ema:
            self.ema.restore(raw)

        self.model.train(prev_mode)

        if self.is_main:
            n = min(n_samples, context.shape[0])
            total_frames = context.shape[1] + target.shape[1]
            media = {}

            for i in range(n):
                ctx_np = self._frames_to_uint8(context[i])
                tgt_np = self._frames_to_uint8(target[i])
                pred_np = self._frames_to_uint8(pred_frames[i])

                if total_frames <= 2:
                    parts = (
                        [ctx_np[f] for f in range(ctx_np.shape[0])]
                        + [tgt_np[f] for f in range(tgt_np.shape[0])]
                        + [pred_np[f] for f in range(pred_np.shape[0])]
                    )
                    caption = "ctx|tgt|pred"
                    if self.use_goal and goals is not None:
                        goal_np = self._frames_to_uint8(goals[i].unsqueeze(0))
                        parts.append(goal_np[0])
                        caption = "ctx|tgt|pred|goal"
                    media[f"val/sample_{i}"] = wandb.Image(
                        np.concatenate(parts, axis=1), caption=caption
                    )
                else:
                    media[f"val/sample_{i}_target"] = wandb.Video(
                        np.concatenate([ctx_np, tgt_np], axis=0).transpose(0, 3, 1, 2),
                        fps=4,
                        format="mp4",
                    )
                    media[f"val/sample_{i}_pred"] = wandb.Video(
                        np.concatenate([ctx_np, pred_np], axis=0).transpose(0, 3, 1, 2),
                        fps=4,
                        format="mp4",
                    )
                    if self.use_goal and goals is not None:
                        goal_np = self._frames_to_uint8(goals[i].unsqueeze(0))
                        media[f"val/sample_{i}_goal"] = wandb.Image(
                            goal_np[0], caption="goal"
                        )

            if val_loss is not None:
                media["val/loss"] = val_loss
            wandb.log(media)

        ddp_barrier()

    @torch.no_grad()
    def _log_train_samples(self, n_samples=3):
        if not self.wandb_enabled:
            return

        ddp_barrier()
        self._sync_model_before_logging()

        prev_mode = self.model.training
        self.model.eval()
        raw = unwrap_model(self.model)
        if self.ema:
            self.ema.apply(raw)

        ds = self.train_loader.dataset
        indices = self._sample_indices_for_logging(ds, n_samples)
        if not indices:
            if self.ema:
                self.ema.restore(raw)
            self.model.train(prev_mode)
            ddp_barrier()
            return

        ctx_list, tgt_list, act_list, goal_list = zip(*[ds[i] for i in indices])
        context = torch.stack(ctx_list).to(self.device)
        target = torch.stack(tgt_list).to(self.device)
        actions = torch.stack(act_list).to(self.device)
        goals = torch.stack(goal_list).to(self.device) if self.use_goal else None

        with torch.amp.autocast("cuda", enabled=self.use_amp, dtype=self.amp_dtype):
            if self.stage == 1:
                pred_frames, _ = raw(context, target, goals)
            else:
                pred_frames, _, _ = raw(context, target, actions, goal=goals)

        if self.ema:
            self.ema.restore(raw)

        self.model.train(prev_mode)

        if self.is_main:
            n = min(n_samples, context.shape[0])
            total_frames = context.shape[1] + target.shape[1]
            media = {}

            for i in range(n):
                ctx_np = self._frames_to_uint8(context[i])
                tgt_np = self._frames_to_uint8(target[i])
                pred_np = self._frames_to_uint8(pred_frames[i])

                if total_frames <= 2:
                    parts = (
                        [ctx_np[f] for f in range(ctx_np.shape[0])]
                        + [tgt_np[f] for f in range(tgt_np.shape[0])]
                        + [pred_np[f] for f in range(pred_np.shape[0])]
                    )
                    caption = "ctx|tgt|pred"
                    if self.use_goal and goals is not None:
                        goal_np = self._frames_to_uint8(goals[i].unsqueeze(0))
                        parts.append(goal_np[0])
                        caption = "ctx|tgt|pred|goal"
                    media[f"train/sample_{i}"] = wandb.Image(
                        np.concatenate(parts, axis=1), caption=caption
                    )
                else:
                    media[f"train/sample_{i}_target"] = wandb.Video(
                        np.concatenate([ctx_np, tgt_np], axis=0).transpose(0, 3, 1, 2),
                        fps=4,
                        format="mp4",
                    )
                    media[f"train/sample_{i}_pred"] = wandb.Video(
                        np.concatenate([ctx_np, pred_np], axis=0).transpose(0, 3, 1, 2),
                        fps=4,
                        format="mp4",
                    )
                    if self.use_goal and goals is not None:
                        goal_np = self._frames_to_uint8(goals[i].unsqueeze(0))
                        media[f"train/sample_{i}_goal"] = wandb.Image(
                            goal_np[0], caption="goal"
                        )

            wandb.log(media)

        ddp_barrier()

    # --- Main loop ---

    def train(self):
        tcfg = self.cfg.train
        max_dl_timeouts = int(tcfg.get("max_dl_timeouts", 3))

        if self.is_main:
            log.info(
                f"Training {self.num_epochs} epochs | batch={self.train_batch_size} | "
                f"accum={self.grad_accum_steps} | world={self.world_size}"
            )

        if self.is_main:
            if self.compile_enabled:
                log.info("Entering training loop (first batch may be slow due to torch.compile)...")
            else:
                log.info("Entering training loop...")

        try:
            for epoch in range(self.start_epoch, self.num_epochs):
                self.current_epoch = epoch

                if self._train_sampler:
                    self._train_sampler.set_epoch(epoch)

                t0 = time.time()

                train_loss = float("inf")
                train_ok = False

                for attempt in range(1, max_dl_timeouts + 1):
                    local_dl_timeout = False
                    local_fatal = False
                    try:
                        train_loss = self._run_epoch(self.train_loader, train=True)
                        train_ok = True
                    except RuntimeError as e:
                        if self._is_dataloader_timeout_error(e):
                            local_dl_timeout = True
                            log.warning(
                                f"[Rank {self.rank}] DL timeout epoch {epoch} attempt {attempt}/{max_dl_timeouts}: {e}"
                            )
                            self.optimizer.zero_grad(set_to_none=True)
                        elif self._is_fatal_dist_error(e):
                            local_fatal = True
                            log.error(f"[Rank {self.rank}] Fatal at epoch {epoch}: {e}")
                        else:
                            raise

                    if self._sync_fatal_flag(local_fatal):
                        self._abort_all_ranks(
                            "Synchronized fatal distributed/CUDA failure during training.",
                            epoch=epoch,
                        )

                    any_timeout = ddp_all_reduce_bool(local_dl_timeout, op="or")
                    all_ok = ddp_all_reduce_bool(train_ok, op="and")

                    if all_ok:
                        break

                    if any_timeout:
                        if attempt >= max_dl_timeouts:
                            if self.is_main:
                                log.error(
                                    f"Max DataLoader timeouts reached at epoch {epoch}; marking train loss as inf and continuing."
                                )
                            train_loss = float("inf")
                            break
                        ddp_barrier()
                        continue

                    self._abort_all_ranks(
                        "Ranks diverged during training without a recognized synchronized timeout.",
                        epoch=epoch,
                    )

                ran_val = (epoch % self.val_every) == 0
                val_loss = float("nan")
                val_loss_raw = float("nan")
                if ran_val:
                    try:
                        val_loss = self._val_loss()
                        val_loss_raw = val_loss
                    except RuntimeError as e:
                        if self._is_dataloader_timeout_error(e):
                            log.warning(
                                f"[Rank {self.rank}] DL timeout during val, using inf: {e}"
                            )
                            val_loss = float("inf")
                            val_loss_raw = float("inf")
                        elif self._is_fatal_dist_error(e):
                            self._abort_all_ranks(
                                "Fatal distributed/CUDA failure during validation.",
                                epoch=epoch,
                            )
                        else:
                            raise

                test_every = int(tcfg.get("test_every", 1))
                test_loss = (
                    self._test_loss()
                    if (
                        self.test_loader
                        and ran_val
                        and test_every > 0
                        and (epoch + 1) % test_every == 0
                    )
                    else None
                )
                elapsed = time.time() - t0

                ddp_barrier()

                if self.is_main:
                    ema_tag = " [EMA]" if self.ema else ""
                    s = f"epoch {epoch:04d} | train {train_loss:.6f} | "
                    if ran_val:
                        s += f"val{ema_tag} {val_loss:.6f} | val[raw] {val_loss_raw:.6f} | "
                    if test_loss is not None:
                        s += f"test{ema_tag} {test_loss:.6f} | "
                    log.info(s + f"{elapsed:.1f}s")

                if (epoch % self.sample_every) == 0:
                    if ran_val:
                        self._log_val_videos(n_samples=3, val_loss=val_loss)
                    self._log_train_samples(n_samples=3)

                if self.is_main:
                    if (epoch % self.checkpoint_every) == 0:
                        self._save_checkpoint(epoch, val_loss, tag="last")

                    if ran_val and val_loss < self.best_val_loss:
                        if self.best_ckpt_path and self.best_ckpt_path.exists():
                            self.best_ckpt_path.unlink()
                        self.best_val_loss = val_loss
                        tag = f"best_epoch{epoch:04d}_loss{val_loss:.6f}"
                        self._save_checkpoint(epoch, val_loss, tag=tag)
                        self.best_ckpt_path = self.ckpt_dir / f"{tag}.pt"

                    if (epoch % self.checkpoint_every) == 0:
                        self._save_checkpoint(epoch, val_loss, tag=f"epoch_{epoch:04d}")

                ddp_barrier()

        finally:
            self._maybe_finish_wandb()
            cleanup_ddp()


@hydra.main(config_path="configs", config_name="stage1_wm_config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    Trainer(cfg).train()


if __name__ == "__main__":
    main()
