"""
train.py — ARVideoPatchTransformer training with Hydra + DDP
============================================================

Single-GPU:
    python train.py data=synthetic

Multi-GPU with DDP:
    torchrun --nproc_per_node=8 train.py data=synthetic
    torchrun --nproc_per_node=8 train.py data.type=real data.root=/clips

Memory optimizations (applied automatically):
    1. fp16 AMP              — train.amp=true
    2. Gradient checkpointing — enabled inside model.py during training
    3. Gradient accumulation  — train.grad_accum_steps

Effective batch size:
    batch_size × grad_accum_steps × world_size

Scheduler config example:
    scheduler:
      warmup_fraction: 0.10
      type: cosine
      min_lr: 1.0e-6

Notes
-----
- Launch DDP with torchrun; RANK / LOCAL_RANK / WORLD_SIZE are set automatically.
- EMA, checkpoint save/load, and wandb run only on rank 0.
- DistributedSampler shards the dataset; set_epoch is called each epoch.
- no_sync() skips redundant gradient all-reduce on intermediate accumulation steps.
- Scheduler total_steps is computed per-rank, so scheduling is GPU-count-independent.
- torch.compile is applied after DDP wrapping for checkpointing compatibility.

Checkpoint format
-----------------
Always saves the unwrapped module state dict:
- model.module for DDP
- model otherwise

_load_checkpoint strips both:
- "_orig_mod."  (torch.compile prefix)
- "module."     (DDP prefix)
"""

from __future__ import annotations

import copy
import functools
import io
import json
import logging
import math
import os
import time
from contextlib import nullcontext
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

from model import ARPatchConfig as _EmuConfig, ARVideoPatchTransformer as _EmuModel
from cosmos_model import ARPatchConfig as _CosmosConfig, ARVideoPatchTransformer as _CosmosModel

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# DDP helpers
# ---------------------------------------------------------------------------

def setup_ddp():
    """
    Initialize the distributed process group when launched via torchrun.

    Returns:
        (rank, local_rank, world_size)

    For plain python launch:
        rank=0, local_rank=0, world_size=1
    """
    if "RANK" not in os.environ:
        return 0, 0, 1

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size


def cleanup_ddp():
    """Destroy the process group if DDP is initialized."""
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def unwrap_model(model: nn.Module) -> nn.Module:
    """Return the underlying model by removing the DDP wrapper if present."""
    return model.module if isinstance(model, DDP) else model


# ---------------------------------------------------------------------------
# EMA
# ---------------------------------------------------------------------------

class EMA:
    """
    Exponential Moving Average of model weights.

    EMA is always applied to the unwrapped model, never to the DDP wrapper.
    """

    def __init__(self, model: nn.Module, decay: float):
        self.decay = decay
        self.num_updates = 0
        self.shadow = copy.deepcopy(model)
        self.shadow.eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """
        Update EMA weights.

        A short adaptive ramp is used at the beginning to avoid overly stale EMA.
        """
        d = min(self.decay, (1.0 + self.num_updates) / (10.0 + self.num_updates))
        self.num_updates += 1

        for s, m in zip(self.shadow.parameters(), model.parameters()):
            s.data.mul_(d).add_(m.data, alpha=1.0 - d)

        for s_buf, m_buf in zip(self.shadow.buffers(), model.buffers()):
            s_buf.data.copy_(m_buf.data)

    def apply(self, model: nn.Module) -> None:
        """Temporarily copy EMA weights into the live model."""
        self._backup = [p.data.clone() for p in model.parameters()]
        for m, s in zip(model.parameters(), self.shadow.parameters()):
            m.data.copy_(s.data)

    def restore(self, model: nn.Module) -> None:
        """Restore original live model weights after apply()."""
        for m, b in zip(model.parameters(), self._backup):
            m.data.copy_(b)
        del self._backup

    def state_dict(self) -> dict:
        return self.shadow.state_dict()

    def load_state_dict(self, state: dict) -> None:
        self.shadow.load_state_dict(state)


# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------

def build_optimizer(model: nn.Module, opt_cfg: DictConfig) -> torch.optim.Optimizer:
    """
    Build AdamW optimizer.

    If no_decay_norm=True, LayerNorm / GroupNorm / Embedding / bias parameters
    are excluded from weight decay.
    """
    if not opt_cfg.no_decay_norm:
        return torch.optim.AdamW(
            model.parameters(),
            lr=opt_cfg.lr,
            weight_decay=opt_cfg.weight_decay,
            betas=tuple(opt_cfg.betas),
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

    log.info(
        f"Optimizer groups — decay: {len(decay_params)} params | "
        f"no_decay: {len(no_decay_params)} params"
    )

    return torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": opt_cfg.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=opt_cfg.lr,
        betas=tuple(opt_cfg.betas),
    )


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------

def build_scheduler(
    optimizer: torch.optim.Optimizer,
    sched_cfg: DictConfig,
    base_lr: float,
    total_steps: int,
):
    """
    Build training scheduler.

    Supported scheduler types:
        - cosine: linear warmup + cosine decay to min_lr

    The LambdaLR multiplier is defined relative to base_lr.

    Args:
        optimizer: optimizer instance
        sched_cfg: scheduler config
        base_lr: optimizer base learning rate
        total_steps: total optimizer update steps across all epochs
    """
    sched_type = sched_cfg.get("type", "cosine").lower()
    warmup_fraction = float(sched_cfg.get("warmup_fraction", 0.0))
    min_lr = float(sched_cfg.get("min_lr", 0.0))

    total_steps = max(1, int(total_steps))
    warmup_steps = min(total_steps - 1, int(total_steps * warmup_fraction)) if total_steps > 1 else 0
    min_lr_ratio = min_lr / base_lr if base_lr > 0 else 0.0

    if sched_type != "cosine":
        raise ValueError(f"Unsupported scheduler type: {sched_type!r}. Expected 'cosine'.")

    def lr_lambda(current_step: int) -> float:
        """
        Return LR multiplier for LambdaLR.

        Behavior:
        - Warmup: linearly increase from a very small value to 1.0
        - Decay: cosine decay from 1.0 to min_lr/base_lr
        """
        # Warmup phase
        if warmup_steps > 0 and current_step < warmup_steps:
            return float(current_step + 1) / float(warmup_steps)

        # No decay region if total_steps is too small
        if total_steps <= warmup_steps + 1:
            return 1.0

        # Cosine decay phase
        decay_steps = total_steps - warmup_steps
        progress = (current_step - warmup_steps) / max(1, decay_steps - 1)
        progress = min(max(progress, 0.0), 1.0)

        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

    log.info(
        f"Scheduler — type={sched_type}, total_steps={total_steps}, "
        f"warmup_steps={warmup_steps}, min_lr={min_lr:.3e}"
    )

    return scheduler


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

class VideoFrameDataset(Dataset):
    """LeRobot-v2 parquet video dataset.

    Each sample returns:
        context    : (frames_in, C, H, W)   input frames
        target     : (frames_out, C, H, W)  target frames (after action gap)
        actions    : (gap, action_dim)       actions between last input and first target
        goal       : (C, H, W)              last frame of the episode (target condition)

    Splits:
        - train / val: in-domain episodes, split at the episode level
        - test       : held-out tasks, excluded from train / val entirely
    """

    def __init__(self, data_cfg: DictConfig, model_cfg: DictConfig, split: str):
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Unknown split: {split!r}")

        self.fin        = model_cfg.frames_in
        self.fout       = model_cfg.frames_out
        self.gap        = int(data_cfg.frame_gap)
        self.image_key  = data_cfg.get("image_key", "image")
        self.action_key = data_cfg.get("action_key", "actions")
        self.use_goal   = data_cfg.get("use_goal", True)
        self._res       = model_cfg.resolution
        self.transform  = transforms.Compose([
            transforms.Resize((model_cfg.resolution, model_cfg.resolution)),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ])

        root = Path(data_cfg.root)

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
            root / "data" / f"chunk-{ep_idx // chunks_size:03d}" / f"episode_{ep_idx:06d}.parquet"
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
            task_names = []
            seen_tasks = set()
            for rec in episode_meta:
                task_name = rec["task"]
                if task_name and task_name not in seen_tasks:
                    seen_tasks.add(task_name)
                    task_names.append(task_name)

        configured_test_tasks = list(data_cfg.get("test_tasks", []))
        test_task_count = int(data_cfg.get("test_task_count", 0))
        if configured_test_tasks:
            unknown_tasks = sorted(set(configured_test_tasks) - set(task_names))
            if unknown_tasks:
                raise ValueError(
                    "Configured data.test_tasks are missing from the dataset: "
                    + ", ".join(unknown_tasks)
                )
            selected_test_tasks = configured_test_tasks
        elif test_task_count > 0:
            if test_task_count >= len(task_names):
                raise ValueError(
                    f"data.test_task_count={test_task_count} must be smaller than the "
                    f"number of tasks in the dataset ({len(task_names)})."
                )
            selected_test_tasks = task_names[:test_task_count]
        else:
            selected_test_tasks = []

        self.test_tasks = tuple(selected_test_tasks)
        test_task_set = set(self.test_tasks)

        in_domain_eps = [rec["episode_index"] for rec in episode_meta if rec["task"] not in test_task_set]
        test_eps = [rec["episode_index"] for rec in episode_meta if rec["task"] in test_task_set]
        ep_lengths = {rec["episode_index"]: rec["length"] for rec in episode_meta}
        ep_tasks = {rec["episode_index"]: rec["task"] for rec in episode_meta}

        if split == "test":
            ep_indices = test_eps
        else:
            n_train_val = len(in_domain_eps)
            if n_train_val <= 1:
                n_val = 0
            else:
                n_val = max(1, int(n_train_val * data_cfg.val_split))
                n_val = min(n_val, n_train_val - 1)

            val_eps = in_domain_eps[:n_val]
            train_eps = in_domain_eps[n_val:]
            ep_indices = train_eps if split == "train" else val_eps

        self.samples = []
        # span = fin + gap + fout - 1  (consecutive frames, no stride)
        span = self.fin + self.gap + self.fout - 1
        self._target_offset = self.fin - 1 + self.gap   # first target frame relative to start
        self._action_offset = self.fin - 1               # first action relative to start

        for ep_idx in ep_indices:
            length = ep_lengths[ep_idx]
            if length < span:
                continue
            ep_path_str = str(episode_files[ep_idx])
            for start in range(length - span + 1):
                self.samples.append((ep_path_str, start, length))

        split_tasks = sorted({ep_tasks[ep_idx] for ep_idx in ep_indices if ep_tasks[ep_idx] is not None})
        if self.test_tasks and split == "train":
            log.info(
                f"Held-out test tasks ({len(self.test_tasks)}): "
                + "; ".join(self.test_tasks)
            )
        log.info(
            f"[{split}] {len(self.samples)} windows / {len(ep_indices)} episodes / "
            f"{len(split_tasks)} tasks"
        )

        self._read_parquet = functools.lru_cache(maxsize=32)(
            lambda path: pd.read_parquet(path, columns=[self.image_key, self.action_key])
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        parquet_path, start, ep_length = self.samples[idx]
        df = self._read_parquet(parquet_path)

        # --- Input frames: [start, start+1, ..., start+fin-1] ---
        ctx_imgs = []
        for t in range(self.fin):
            img_data = df.iloc[start + t][self.image_key]
            img = Image.open(io.BytesIO(img_data["bytes"])).convert("RGB")
            ctx_imgs.append(self.transform(img))

        # --- Target frames: [start+fin-1+gap, ..., start+fin-1+gap+fout-1] ---
        tgt_imgs = []
        for t in range(self.fout):
            frame_idx = start + self._target_offset + t
            img_data = df.iloc[frame_idx][self.image_key]
            img = Image.open(io.BytesIO(img_data["bytes"])).convert("RGB")
            tgt_imgs.append(self.transform(img))

        # --- Actions between last input frame and first target frame (m = gap) ---
        act_start = start + self._action_offset
        actions = torch.stack([
            torch.tensor(
                np.array(df.iloc[act_start + i][self.action_key], copy=True),
                dtype=torch.float32,
            )
            for i in range(self.gap)
        ])  # (gap, action_dim)

        # --- Goal frame: last frame of the episode (optional) ---
        if self.use_goal:
            goal_data = df.iloc[ep_length - 1][self.image_key]
            goal = Image.open(io.BytesIO(goal_data["bytes"])).convert("RGB")
            goal = self.transform(goal)
        else:
            goal = torch.zeros(3, self._res, self._res)  # dummy, ignored by trainer

        return torch.stack(ctx_imgs), torch.stack(tgt_imgs), actions, goal


class SyntheticVideoDataset(Dataset):
    """Synthetic moving-dot dataset for debugging and smoke tests."""

    def __init__(self, data_cfg: DictConfig, model_cfg: DictConfig, split: str):
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
        C, H, W, T = self.cfg.num_channels, self.cfg.resolution, self.cfg.resolution, self.T

        if self.moving_dot:
            frames = torch.full((T, C, H, W), -0.9)
            speed = torch.randint(2, 6, (1,)).item()
            x0 = torch.randint(0, W // 2, (1,)).item()
            y0 = torch.randint(0, H // 2, (1,)).item()

            for t in range(T):
                cx = min(x0 + t * speed, W - 5)
                cy = min(y0 + t * speed, H - 5)
                frames[t, :, cy:cy + 5, cx:cx + 5] = 0.9
        else:
            frames = torch.rand(T, C, H, W) * 2 - 1

        actions = torch.zeros(self.gap, self.action_dim)
        goal = frames[-1].clone() if self.use_goal else torch.zeros(C, H, W)
        return frames[:self.fin], frames[self.fin:], actions, goal


def build_datasets(data_cfg: DictConfig, model_cfg: DictConfig):
    """Build train / validation / test datasets from config."""
    cls = {
        "real": VideoFrameDataset,
        "synthetic": SyntheticVideoDataset,
    }
    if data_cfg.type not in cls:
        raise ValueError(f"Unknown data.type: {data_cfg.type!r}")

    dataset_cls = cls[data_cfg.type]
    train_ds = dataset_cls(data_cfg, model_cfg, "train")
    val_ds = dataset_cls(data_cfg, model_cfg, "val")
    test_ds = dataset_cls(data_cfg, model_cfg, "test") if data_cfg.type == "real" else None
    return train_ds, val_ds, test_ds


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg

        # -------------------------------------------------------------------
        # DDP setup
        # -------------------------------------------------------------------
        self.rank, self.local_rank, self.world_size = setup_ddp()
        self.is_main = (self.rank == 0)

        # -------------------------------------------------------------------
        # Device setup
        # -------------------------------------------------------------------
        if torch.cuda.is_available():
            self.device = torch.device(f"cuda:{self.local_rank}")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        if self.is_main:
            log.info(f"Device: {self.device} | world_size: {self.world_size}")

        # Enable TF32 for faster training on modern NVIDIA GPUs
        if self.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True

        # Use different seeds across ranks so stochastic ops differ per GPU
        torch.manual_seed(cfg.seed + self.rank)

        # -------------------------------------------------------------------
        # Gradient accumulation
        # -------------------------------------------------------------------
        self.grad_accum_steps = int(cfg.train.get("grad_accum_steps", 1))
        if self.is_main:
            effective_batch = cfg.train.batch_size * self.grad_accum_steps * self.world_size
            log.info(
                f"micro_batch={cfg.train.batch_size} | "
                f"accum_steps={self.grad_accum_steps} | "
                f"world_size={self.world_size} | "
                f"effective_batch={effective_batch}"
            )

        # -------------------------------------------------------------------
        # Model
        # -------------------------------------------------------------------
        mcfg = cfg.model
        use_cosmos = mcfg.get("arch", "emu3") == "cosmos"

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

        if use_cosmos:
            config_kwargs["qk_norm"] = mcfg.get("qk_norm", True)
            config_kwargs["parallel_attn"] = mcfg.get("parallel_attn", False)
            self.model_cfg = _CosmosConfig(**config_kwargs)
            raw_model = _CosmosModel(self.model_cfg).to(self.device)
            arch_name = "Cosmos"
        else:
            self.model_cfg = _EmuConfig(**config_kwargs)
            raw_model = _EmuModel(self.model_cfg).to(self.device)
            arch_name = "Emu3"

        if self.is_main:
            n_params = sum(p.numel() for p in raw_model.parameters()) / 1e6
            log.info(f"Arch: {arch_name} | Parameters: {n_params:.2f} M")

        # -------------------------------------------------------------------
        # DDP wrapping
        # -------------------------------------------------------------------
        if self.world_size > 1:
            self.model = DDP(
                raw_model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=False,
            )
            if self.is_main:
                log.info(f"DDP enabled ({self.world_size} GPUs)")
        else:
            self.model = raw_model

        # -------------------------------------------------------------------
        # torch.compile
        # -------------------------------------------------------------------
        if cfg.train.get("compile", False) and hasattr(torch, "compile"):
            torch._dynamo.config.optimize_ddp = False
            self.model = torch.compile(self.model)
            if self.is_main:
                log.info("torch.compile enabled (optimize_ddp=False)")

        # -------------------------------------------------------------------
        # EMA
        # -------------------------------------------------------------------
        ema_cfg = cfg.train.ema
        raw = unwrap_model(self.model)
        self.ema = EMA(raw, decay=ema_cfg.decay) if (ema_cfg.enabled and self.is_main) else None
        if self.ema is not None and self.is_main:
            log.info(f"EMA enabled | decay={ema_cfg.decay} | update_every={ema_cfg.update_every}")

        # -------------------------------------------------------------------
        # Optimizer
        # -------------------------------------------------------------------
        self.optimizer = build_optimizer(unwrap_model(self.model), cfg.train.optimizer)

        # -------------------------------------------------------------------
        # AMP
        # -------------------------------------------------------------------
        self.use_amp = cfg.train.amp and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        # -------------------------------------------------------------------
        # Data
        # -------------------------------------------------------------------
        self.use_goal = cfg.data.get("use_goal", True)
        train_ds, val_ds, test_ds = build_datasets(cfg.data, cfg.model)
        self.test_task_names = list(getattr(train_ds, "test_tasks", ()))
        n_workers = cfg.data.num_workers

        loader_kw = dict(
            num_workers=n_workers,
            pin_memory=cfg.data.pin_memory and self.device.type == "cuda",
            persistent_workers=cfg.data.get("persistent_workers", False) and n_workers > 0,
            prefetch_factor=cfg.data.get("prefetch_factor", 2) if n_workers > 0 else None,
        )

        if self.world_size > 1:
            train_sampler = DistributedSampler(
                train_ds,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=True,
            )
            val_sampler = DistributedSampler(
                val_ds,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=False,
            )
            test_sampler = None
            if test_ds is not None and len(test_ds) > 0:
                test_sampler = DistributedSampler(
                    test_ds,
                    num_replicas=self.world_size,
                    rank=self.rank,
                    shuffle=False,
                )

            self.train_loader = DataLoader(
                train_ds,
                batch_size=cfg.train.batch_size,
                sampler=train_sampler,
                drop_last=True,
                **loader_kw,
            )
            self.val_loader = DataLoader(
                val_ds,
                batch_size=cfg.train.batch_size * 2,
                sampler=val_sampler,
                **loader_kw,
            )
            self.test_loader = (
                DataLoader(
                    test_ds,
                    batch_size=cfg.train.batch_size * 2,
                    sampler=test_sampler,
                    **loader_kw,
                )
                if test_sampler is not None
                else None
            )
            self._train_sampler = train_sampler
        else:
            self.train_loader = DataLoader(
                train_ds,
                batch_size=cfg.train.batch_size,
                shuffle=True,
                drop_last=True,
                **loader_kw,
            )
            self.val_loader = DataLoader(
                val_ds,
                batch_size=cfg.train.batch_size * 2,
                shuffle=False,
                **loader_kw,
            )
            self.test_loader = (
                DataLoader(
                    test_ds,
                    batch_size=cfg.train.batch_size * 2,
                    shuffle=False,
                    **loader_kw,
                )
                if test_ds is not None and len(test_ds) > 0
                else None
            )
            self._train_sampler = None

        if self.is_main and self.test_task_names:
            log.info(
                f"Task-heldout test split enabled ({len(self.test_task_names)} tasks): "
                + "; ".join(self.test_task_names)
            )

        # -------------------------------------------------------------------
        # Scheduler
        # -------------------------------------------------------------------
        # Steps per epoch are computed per-rank. DistributedSampler already
        # partitions the dataset, so this is correct for all world sizes.
        steps_per_epoch = max(1, len(self.train_loader) // self.grad_accum_steps)
        total_steps = cfg.train.epochs * steps_per_epoch

        self.scheduler = build_scheduler(
            optimizer=self.optimizer,
            sched_cfg=cfg.train.scheduler,
            base_lr=cfg.train.optimizer.lr,
            total_steps=total_steps,
        )

        # -------------------------------------------------------------------
        # State
        # -------------------------------------------------------------------
        self.ckpt_dir = Path(cfg.train.ckpt_dir)
        if self.is_main:
            self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        self.start_epoch = 0
        self.best_val_loss = float("inf")
        self.best_ckpt_path: Path | None = None
        self.global_step = 0

        if cfg.train.resume:
            self._load_checkpoint(cfg.train.resume)

        # -------------------------------------------------------------------
        # wandb
        # -------------------------------------------------------------------
        wandb_cfg = cfg.get("wandb", {})
        self.use_wandb = self.is_main and wandb_cfg.get("enabled", True)
        if self.use_wandb:
            wandb.init(
                project=wandb_cfg.get("project", "TTT-WM"),
                name=wandb_cfg.get("name", cfg.experiment_name),
                config=OmegaConf.to_container(cfg, resolve=True),
                resume="allow" if cfg.train.resume else None,
            )
            wandb.define_metric("epoch")
            wandb.define_metric("epoch/*", step_metric="epoch")

    # -----------------------------------------------------------------------
    # Checkpointing
    # -----------------------------------------------------------------------

    def _save_checkpoint(self, epoch: int, val_loss: float, tag: str = "last"):
        """Save the unwrapped model state dict without DDP or compile prefixes."""
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
        log.info(f"Saved checkpoint: {path} | val_loss={val_loss:.6f}")

    def _load_checkpoint(self, path: str):
        """Load checkpoint and strip optional compile / DDP prefixes."""
        ckpt = torch.load(path, map_location=self.device, weights_only=False)

        def _strip(sd, prefix):
            if any(k.startswith(prefix) for k in sd):
                return {k.removeprefix(prefix): v for k, v in sd.items()}
            return sd

        model_sd = _strip(_strip(ckpt["model"], "_orig_mod."), "module.")
        unwrap_model(self.model).load_state_dict(model_sd)

        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.scheduler.load_state_dict(ckpt["scheduler"])
        self.scaler.load_state_dict(ckpt["scaler"])

        if self.ema is not None and "ema" in ckpt:
            self.ema.load_state_dict(ckpt["ema"])

        self.start_epoch = ckpt["epoch"] + 1
        self.global_step = ckpt.get("global_step", 0)
        self.best_val_loss = ckpt.get("val_loss", float("inf"))

        if self.is_main:
            log.info(f"Resumed from {path} | start_epoch={self.start_epoch}")

    # -----------------------------------------------------------------------
    # Epoch loop
    # -----------------------------------------------------------------------

    def _run_epoch(self, loader: DataLoader, train: bool) -> float:
        """
        Run one epoch.

        Training logic:
        - Zero gradients only at the start of an accumulation window
        - Divide loss by grad_accum_steps before backward()
        - Under DDP, suppress all-reduce on non-final micro-steps with no_sync()
        - Perform optimizer step, scheduler step, and EMA update only on the
          final micro-step of each accumulation window
        """
        self.model.train(train)
        total_loss, n_opt_steps = 0.0, 0
        ema_cfg = self.cfg.train.ema
        log_every = self.cfg.train.log_every

        pbar = tqdm(
            loader,
            desc="train" if train else "val",
            leave=False,
            disable=not self.is_main,
        )

        grad_ctx = torch.enable_grad() if train else torch.no_grad()

        accum_loss = 0.0
        accum_step = 0

        with grad_ctx:
            for batch_idx, (context, target, actions, goal) in enumerate(pbar):
                context = context.to(self.device, non_blocking=True)
                target = target.to(self.device, non_blocking=True)
                actions = actions.to(self.device, non_blocking=True)
                goal = goal.to(self.device, non_blocking=True) if self.use_goal else None

                if train:
                    is_last_accum = (
                        (accum_step == self.grad_accum_steps - 1)
                        or (batch_idx == len(loader) - 1)
                    )

                    ddp_sync_ctx = (
                        nullcontext()
                        if (not isinstance(self.model, DDP) or is_last_accum)
                        else self.model.no_sync()
                    )

                    with ddp_sync_ctx, torch.amp.autocast("cuda", enabled=self.use_amp):
                        _, _, loss = self.model(context, target, actions, goal)
                        scaled_loss = loss / self.grad_accum_steps

                    self.scaler.scale(scaled_loss).backward()

                    accum_loss += loss.detach().item()
                    accum_step += 1

                    if is_last_accum:
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

                        if self.ema is not None and self.global_step % ema_cfg.update_every == 0:
                            self.ema.update(unwrap_model(self.model))

                        avg_loss = accum_loss / self.grad_accum_steps
                        total_loss += avg_loss
                        n_opt_steps += 1

                        if self.is_main:
                            lr = self.scheduler.get_last_lr()[0]
                            if self.use_wandb:
                                wandb.log(
                                    {
                                        "train/loss": avg_loss,
                                        "train/lr": lr,
                                        "global_step": self.global_step,
                                    },
                                    step=self.global_step,
                                )
                            pbar.set_postfix(loss=f"{avg_loss:.4f}", lr=f"{lr:.2e}")

                            if self.global_step % log_every == 0:
                                log.info(
                                    f"step {self.global_step:06d} | "
                                    f"loss {avg_loss:.6f} | lr {lr:.2e}"
                                )

                        accum_loss = 0.0
                        accum_step = 0
                else:
                    with torch.amp.autocast("cuda", enabled=self.use_amp):
                        _, _, loss = self.model(context, target, actions, goal)

                    if self.is_main:
                        pbar.set_postfix(loss=f"{loss.item():.4f}")

                    total_loss += loss.item()
                    n_opt_steps += 1

        avg = total_loss / max(n_opt_steps, 1)

        # Aggregate validation loss across all ranks
        if self.world_size > 1 and not train:
            t = torch.tensor(avg, device=self.device)
            dist.all_reduce(t, op=dist.ReduceOp.AVG)
            avg = t.item()

        return avg

    def _val_loss(self) -> float:
        """Evaluate validation loss using EMA weights when EMA is enabled."""
        return self._eval_loader_loss(self.val_loader)

    def _test_loss(self) -> float | None:
        """Evaluate held-out test loss using EMA weights when EMA is enabled."""
        if self.test_loader is None:
            return None
        return self._eval_loader_loss(self.test_loader)

    def _eval_loader_loss(self, loader: DataLoader) -> float:
        """Evaluate a loader using EMA weights when EMA is enabled."""
        raw = unwrap_model(self.model)
        if self.ema is not None:
            self.ema.apply(raw)

        loss = self._run_epoch(loader, train=False)

        if self.ema is not None:
            self.ema.restore(raw)

        return loss

    # -----------------------------------------------------------------------
    # Video logging helpers
    # -----------------------------------------------------------------------

    @staticmethod
    def _frames_to_uint8(frames: torch.Tensor) -> np.ndarray:
        """
        Convert normalized float tensor in [-1, 1] to uint8 HWC frames.

        Output shape:
            [T, H, W, C]
        """
        x = (frames.float().clamp(-1, 1) * 0.5 + 0.5) * 255.0
        return x.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)

    @torch.no_grad()
    def _log_val_videos(self, n_samples: int = 4):
        """Log validation targets and predictions to wandb. Rank 0 only."""
        if not self.use_wandb:
            return

        self.model.eval()
        raw = unwrap_model(self.model)

        if self.ema is not None:
            self.ema.apply(raw)

        ds = self.val_loader.dataset
        indices = torch.randperm(len(ds))[:n_samples].tolist()
        ctx_list, tgt_list, act_list, goal_list = zip(*[ds[i] for i in indices])

        context = torch.stack(ctx_list).to(self.device)
        target = torch.stack(tgt_list).to(self.device)
        actions = torch.stack(act_list).to(self.device)
        goals = torch.stack(goal_list).to(self.device) if self.use_goal else None

        with torch.amp.autocast("cuda", enabled=self.use_amp):
            pred_frames, _, _ = self.model(context, target, actions, goals)

        if self.ema is not None:
            self.ema.restore(raw)

        n = min(n_samples, context.shape[0])
        total_frames = context.shape[1] + target.shape[1]
        media = {}

        for i in range(n):
            ctx_np = self._frames_to_uint8(context[i])
            tgt_np = self._frames_to_uint8(target[i])
            pred_np = self._frames_to_uint8(pred_frames[i])

            if total_frames <= 2:
                parts = [ctx_np[f] for f in range(ctx_np.shape[0])]
                parts += [tgt_np[f] for f in range(tgt_np.shape[0])]
                parts += [pred_np[f] for f in range(pred_np.shape[0])]
                composite = np.concatenate(parts, axis=1)

                media[f"val/sample_{i}"] = wandb.Image(
                    composite,
                    caption="context | target | prediction",
                )
            else:
                gt_vid = np.concatenate([ctx_np, tgt_np], axis=0).transpose(0, 3, 1, 2)
                pred_vid = np.concatenate([ctx_np, pred_np], axis=0).transpose(0, 3, 1, 2)

                media[f"val/sample_{i}_target"] = wandb.Video(gt_vid, fps=4, format="mp4")
                media[f"val/sample_{i}_pred"] = wandb.Video(pred_vid, fps=4, format="mp4")

        wandb.log(media, step=self.global_step)

    # -----------------------------------------------------------------------
    # Main training loop
    # -----------------------------------------------------------------------

    def train(self):
        tcfg = self.cfg.train

        if self.is_main:
            log.info(
                f"Training {tcfg.epochs} epochs | "
                f"micro_batch={tcfg.batch_size} | "
                f"accum={self.grad_accum_steps} | "
                f"world_size={self.world_size}"
            )

        for epoch in range(self.start_epoch, tcfg.epochs):
            if self._train_sampler is not None:
                # Required so DistributedSampler reshuffles differently each epoch
                self._train_sampler.set_epoch(epoch)

            t0 = time.time()

            train_loss = self._run_epoch(self.train_loader, train=True)
            val_loss = self._val_loss()
            val_loss_raw = self._run_epoch(self.val_loader, train=False)
            test_loss = self._test_loss()

            elapsed = time.time() - t0

            if self.is_main:
                ema_tag = " [EMA]" if self.ema is not None else ""
                summary = (
                    f"epoch {epoch:04d} | "
                    f"train {train_loss:.6f} | "
                    f"val{ema_tag} {val_loss:.6f} | "
                    f"val[raw] {val_loss_raw:.6f} | "
                )
                if test_loss is not None:
                    summary += f"test{ema_tag} {test_loss:.6f} | "
                summary += f"{elapsed:.1f}s"
                log.info(summary)

                metrics = {
                    "epoch": epoch,
                    "epoch/train_loss": train_loss,
                    "epoch/val_loss": val_loss,
                    "epoch/val_loss_raw": val_loss_raw,
                    "epoch/lr": self.scheduler.get_last_lr()[0],
                }
                if test_loss is not None:
                    metrics["epoch/test_loss"] = test_loss

                if self.use_wandb:
                    wandb.log(
                        metrics,
                        step=self.global_step,
                    )
                    if self.test_task_names:
                        wandb.run.summary["test_tasks"] = list(self.test_task_names)

                self._log_val_videos(n_samples=4)
                self._save_checkpoint(epoch, val_loss, tag="last")

                if val_loss < self.best_val_loss:
                    if self.best_ckpt_path is not None and self.best_ckpt_path.exists():
                        self.best_ckpt_path.unlink()

                    self.best_val_loss = val_loss
                    best_tag = f"best_epoch{epoch:04d}_loss{val_loss:.6f}"
                    self._save_checkpoint(epoch, val_loss, tag=best_tag)
                    self.best_ckpt_path = self.ckpt_dir / f"{best_tag}.pt"

                    if self.use_wandb:
                        wandb.run.summary["best_val_loss"] = val_loss
                        wandb.run.summary["best_epoch"] = epoch

                if (epoch + 1) % tcfg.save_every == 0:
                    self._save_checkpoint(epoch, val_loss, tag=f"epoch_{epoch:04d}")

        if self.use_wandb:
            wandb.finish()

        cleanup_ddp()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

@hydra.main(config_path="configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    Trainer(cfg).train()


if __name__ == "__main__":
    main()
