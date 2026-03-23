"""
train.py — ARVideoPatchTransformer training with Hydra + DDP
=============================================================

Single-GPU (original behaviour, unchanged):
    python train.py data=synthetic

Multi-GPU with DDP (new):
    torchrun --nproc_per_node=8 train.py data=synthetic
    torchrun --nproc_per_node=8 train.py data.type=real data.root=/clips

Memory optimisations (applied automatically):
    1. fp16 AMP          — train.amp=true  (default)
    2. Gradient checkpointing — inside CausalTransformer in model.py,
       active whenever self.training=True, no extra config needed
    3. Gradient accumulation — train.grad_accum_steps  (default 1)

       Effective batch = batch_size × grad_accum_steps × world_size
       Example: batch_size=8, grad_accum_steps=4, 8 GPUs → effective B=256

DDP notes
---------
- Launch with torchrun; RANK / LOCAL_RANK / WORLD_SIZE are set automatically.
- EMA, checkpoint save/load, wandb → rank 0 only.
- DistributedSampler shards dataset; set_epoch called each epoch.
- no_sync() context skips gradient all-reduce on intermediate accumulation
  micro-steps — avoids (grad_accum_steps-1) wasted all-reduces per step.
- OneCycleLR total_steps computed per-rank (DistributedSampler already
  divides dataset by world_size), so LR schedule is GPU-count-independent.
- torch.compile is applied AFTER DDP wrapping for grad-checkpoint compatibility.

Checkpoint format
-----------------
Always saves unwrapped module state dict (model.module for DDP, model otherwise).
_load_checkpoint strips both '_orig_mod.' (compile) and 'module.' (DDP) prefixes.
"""

from __future__ import annotations

import copy
import functools
import io
import json
import logging
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
    Initialise process group when launched via torchrun.
    Returns (rank, local_rank, world_size).
    Plain python launch: rank=0, local_rank=0, world_size=1.
    """
    if "RANK" not in os.environ:
        return 0, 0, 1

    dist.init_process_group(backend="nccl")
    rank       = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size


def cleanup_ddp():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def unwrap_model(model: nn.Module) -> nn.Module:
    """Strip DDP wrapper to reach the underlying module."""
    return model.module if isinstance(model, DDP) else model


# ---------------------------------------------------------------------------
# EMA
# ---------------------------------------------------------------------------

class EMA:
    """
    Exponential Moving Average of model weights.
    Always operated on the *unwrapped* module (not a DDP wrapper).
    """

    def __init__(self, model: nn.Module, decay: float):
        self.decay       = decay
        self.num_updates = 0
        self.shadow      = copy.deepcopy(model)
        self.shadow.eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        d = min(self.decay, (1.0 + self.num_updates) / (10.0 + self.num_updates))
        self.num_updates += 1
        for s, m in zip(self.shadow.parameters(), model.parameters()):
            s.data.mul_(d).add_(m.data, alpha=1.0 - d)
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
        return self.shadow.state_dict()

    def load_state_dict(self, state: dict) -> None:
        self.shadow.load_state_dict(state)


# ---------------------------------------------------------------------------
# Optimizer with weight-decay grouping
# ---------------------------------------------------------------------------

def build_optimizer(model: nn.Module, opt_cfg: DictConfig) -> torch.optim.Optimizer:
    """AdamW with optional weight-decay param groups. Pass unwrapped module."""
    if not opt_cfg.no_decay_norm:
        return torch.optim.AdamW(
            model.parameters(),
            lr           = opt_cfg.lr,
            weight_decay = opt_cfg.weight_decay,
            betas        = tuple(opt_cfg.betas),
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
        f"Optimizer — decay: {len(decay_params)} params | "
        f"no_decay: {len(no_decay_params)} params"
    )
    return torch.optim.AdamW(
        [
            {"params": decay_params,    "weight_decay": opt_cfg.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr    = opt_cfg.lr,
        betas = tuple(opt_cfg.betas),
    )


# ---------------------------------------------------------------------------
# Datasets  (identical to original — DDP sharding done via sampler)
# ---------------------------------------------------------------------------

class VideoFrameDataset(Dataset):
    """LeRobot-v2 parquet video dataset (e.g. LIBERO)."""

    def __init__(self, data_cfg: DictConfig, model_cfg: DictConfig, split: str):
        self.T         = model_cfg.frames_in + model_cfg.frames_out
        self.fin       = model_cfg.frames_in
        self.stride    = data_cfg.get("frame_stride", 1)
        self.gap       = data_cfg.get("frame_gap", self.stride)
        self.image_key = data_cfg.get("image_key", "image")
        self.transform = transforms.Compose([
            transforms.Resize((model_cfg.resolution, model_cfg.resolution)),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ])

        root = Path(data_cfg.root)
        with open(root / "meta" / "info.json") as f:
            info = json.load(f)
        total_episodes = info["total_episodes"]
        chunks_size    = info["chunks_size"]

        episode_files = [
            root / "data" / f"chunk-{ep_idx // chunks_size:03d}" / f"episode_{ep_idx:06d}.parquet"
            for ep_idx in range(total_episodes)
        ]

        ep_lengths = {}
        with open(root / "meta" / "episodes.jsonl") as f:
            for line in f:
                rec = json.loads(line)
                ep_lengths[rec["episode_index"]] = rec["length"]

        n_val = max(1, int(total_episodes * data_cfg.val_split))
        ep_indices = list(range(n_val, total_episodes)) if split == "train" else list(range(n_val))

        self.samples = []
        fout = self.T - self.fin
        span = (self.fin - 1) * self.stride + self.gap + (fout - 1) * self.stride + 1
        self._target_offset = (self.fin - 1) * self.stride + self.gap
        for ep_idx in ep_indices:
            length = ep_lengths[ep_idx]
            if length < span:
                continue
            ep_path_str = str(episode_files[ep_idx])
            for start in range(length - span + 1):
                self.samples.append((ep_path_str, start))

        log.info(f"[{split}] {len(self.samples)} windows / {len(ep_indices)} episodes")

        self._read_parquet = functools.lru_cache(maxsize=32)(
            lambda path: pd.read_parquet(path, columns=[self.image_key])
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        parquet_path, start = self.samples[idx]
        df   = self._read_parquet(parquet_path)
        imgs = []
        for t in range(self.T):
            frame_idx = (
                start + t * self.stride if t < self.fin
                else start + self._target_offset + (t - self.fin) * self.stride
            )
            img_data  = df.iloc[frame_idx][self.image_key]
            img       = Image.open(io.BytesIO(img_data["bytes"])).convert("RGB")
            imgs.append(self.transform(img))
        frames = torch.stack(imgs)
        return frames[: self.fin], frames[self.fin :]


class SyntheticVideoDataset(Dataset):
    """Synthetic moving-dot dataset — no files needed."""

    def __init__(self, data_cfg: DictConfig, model_cfg: DictConfig, split: str):
        self.T          = model_cfg.frames_in + model_cfg.frames_out
        self.fin        = model_cfg.frames_in
        self.cfg        = model_cfg
        self.moving_dot = data_cfg.moving_dot
        self.n          = data_cfg.n_train if split == "train" else data_cfg.n_val

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        C, H, W, T = self.cfg.num_channels, self.cfg.resolution, self.cfg.resolution, self.T
        if self.moving_dot:
            frames = torch.full((T, C, H, W), -0.9)
            speed  = torch.randint(2, 6, (1,)).item()
            x0     = torch.randint(0, W // 2, (1,)).item()
            y0     = torch.randint(0, H // 2, (1,)).item()
            for t in range(T):
                cx = min(x0 + t * speed, W - 5)
                cy = min(y0 + t * speed, H - 5)
                frames[t, :, cy:cy + 5, cx:cx + 5] = 0.9
        else:
            frames = torch.rand(T, C, H, W) * 2 - 1
        return frames[: self.fin], frames[self.fin :]


def build_datasets(data_cfg: DictConfig, model_cfg: DictConfig):
    cls = {"real": VideoFrameDataset, "synthetic": SyntheticVideoDataset}
    if data_cfg.type not in cls:
        raise ValueError(f"Unknown data.type: {data_cfg.type!r}")
    Ds = cls[data_cfg.type]
    return Ds(data_cfg, model_cfg, "train"), Ds(data_cfg, model_cfg, "val")


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg

        # ── DDP setup ───────────────────────────────────────────────────
        self.rank, self.local_rank, self.world_size = setup_ddp()
        self.is_main = (self.rank == 0)

        # ── device ──────────────────────────────────────────────────────
        if torch.cuda.is_available():
            self.device = torch.device(f"cuda:{self.local_rank}")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        if self.is_main:
            log.info(f"Device: {self.device}  |  world_size: {self.world_size}")

        # ── H100 performance flags ─────────────────────────────────────
        if self.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32        = True
            torch.backends.cudnn.benchmark         = True

        # Different seed per rank so data augmentation differs across GPUs
        torch.manual_seed(cfg.seed + self.rank)

        # ── gradient accumulation ──────────────────────────────────────
        self.grad_accum_steps = int(cfg.train.get("grad_accum_steps", 1))
        if self.is_main:
            eff = cfg.train.batch_size * self.grad_accum_steps * self.world_size
            log.info(
                f"micro_batch={cfg.train.batch_size}  "
                f"accum_steps={self.grad_accum_steps}  "
                f"world_size={self.world_size}  "
                f"→ effective_batch={eff}"
            )

        # ── model ───────────────────────────────────────────────────────
        mcfg       = cfg.model
        use_cosmos = mcfg.get("arch", "emu3") == "cosmos"

        config_kwargs = dict(
            resolution   = mcfg.resolution,
            num_channels = mcfg.num_channels,
            patch_size   = mcfg.patch_size,
            d_model      = mcfg.d_model,
            n_heads      = mcfg.n_heads,
            n_layers     = mcfg.n_layers,
            mlp_ratio    = mcfg.mlp_ratio,
            dropout      = mcfg.dropout,
            frames_in    = mcfg.frames_in,
            frames_out   = mcfg.frames_out,
        )
        if use_cosmos:
            config_kwargs["qk_norm"]       = mcfg.get("qk_norm",       True)
            config_kwargs["parallel_attn"] = mcfg.get("parallel_attn", False)
            self.model_cfg = _CosmosConfig(**config_kwargs)
            raw_model      = _CosmosModel(self.model_cfg).to(self.device)
            arch_name      = "Cosmos"
        else:
            self.model_cfg = _EmuConfig(**config_kwargs)
            raw_model      = _EmuModel(self.model_cfg).to(self.device)
            arch_name      = "Emu3"

        if self.is_main:
            n_params = sum(p.numel() for p in raw_model.parameters()) / 1e6
            log.info(f"Arch: {arch_name}  |  Parameters: {n_params:.2f} M")

        # compile is applied AFTER DDP — see below

        # ── DDP wrapping ────────────────────────────────────────────────
        if self.world_size > 1:
            self.model = DDP(
                raw_model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=False,
            )
            if self.is_main:
                log.info(f"DDP enabled  ({self.world_size} GPUs)")
        else:
            self.model = raw_model

        # ── torch.compile ───────────────────────────────────────────────
        # gradient checkpointing uses higher-order ops that are incompatible
        # with DDPOptimizer's bucket-splitting pass regardless of compile order.
        # optimize_ddp=False disables that pass; dynamo compiles the full graph
        # as one unit. Cost: one large all-reduce bucket instead of N small ones
        # — negligible for 2 GPUs, and compile kernel fusion more than covers it.
        if cfg.train.get("compile", False) and hasattr(torch, "compile"):
            torch._dynamo.config.optimize_ddp = False
            self.model = torch.compile(self.model)
            if self.is_main:
                log.info("torch.compile enabled (optimize_ddp=False, grad-ckpt compatible)")

        # ── EMA — rank 0 only, on unwrapped module ──────────────────────
        ema_cfg  = cfg.train.ema
        raw      = unwrap_model(self.model)
        self.ema = EMA(raw, decay=ema_cfg.decay) if (ema_cfg.enabled and self.is_main) else None
        if self.ema and self.is_main:
            log.info(f"EMA  decay={ema_cfg.decay}  update_every={ema_cfg.update_every}")

        # ── optimizer (unwrapped params) ─────────────────────────────────
        self.optimizer = build_optimizer(unwrap_model(self.model), cfg.train.optimizer)

        # ── AMP ─────────────────────────────────────────────────────────
        self.use_amp = cfg.train.amp and self.device.type == "cuda"
        self.scaler  = torch.amp.GradScaler('cuda', enabled=self.use_amp)

        # ── data ─────────────────────────────────────────────────────────
        train_ds, val_ds = build_datasets(cfg.data, cfg.model)
        n_workers = cfg.data.num_workers
        loader_kw = dict(
            num_workers        = n_workers,
            pin_memory         = cfg.data.pin_memory and self.device.type == "cuda",
            persistent_workers = cfg.data.get("persistent_workers", False) and n_workers > 0,
            prefetch_factor    = cfg.data.get("prefetch_factor", 2) if n_workers > 0 else None,
        )

        if self.world_size > 1:
            train_sampler = DistributedSampler(
                train_ds, num_replicas=self.world_size, rank=self.rank, shuffle=True,
            )
            val_sampler = DistributedSampler(
                val_ds, num_replicas=self.world_size, rank=self.rank, shuffle=False,
            )
            self.train_loader = DataLoader(
                train_ds, batch_size=cfg.train.batch_size,
                sampler=train_sampler, drop_last=True, **loader_kw,
            )
            self.val_loader = DataLoader(
                val_ds, batch_size=cfg.train.batch_size * 2,
                sampler=val_sampler, **loader_kw,
            )
            self._train_sampler = train_sampler
        else:
            self.train_loader = DataLoader(
                train_ds, batch_size=cfg.train.batch_size,
                shuffle=True, drop_last=True, **loader_kw,
            )
            self.val_loader = DataLoader(
                val_ds, batch_size=cfg.train.batch_size * 2,
                shuffle=False, **loader_kw,
            )
            self._train_sampler = None

        # ── scheduler ────────────────────────────────────────────────────
        # steps_per_epoch = batches per rank / accum_steps
        # DistributedSampler already divides dataset by world_size, so this
        # is correct regardless of GPU count.
        steps_per_epoch = max(1, len(self.train_loader) // self.grad_accum_steps)
        total_steps     = cfg.train.epochs * steps_per_epoch
        self.scheduler  = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr          = cfg.train.optimizer.lr,
            total_steps     = total_steps,
            pct_start       = cfg.train.scheduler.warmup_fraction,
            anneal_strategy = "cos",
        )

        # ── state ─────────────────────────────────────────────────────────
        self.ckpt_dir      = Path(cfg.train.ckpt_dir)
        if self.is_main:
            self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.start_epoch    = 0
        self.best_val_loss  = float("inf")
        self.best_ckpt_path: Path | None = None
        self.global_step    = 0

        if cfg.train.resume:
            self._load_checkpoint(cfg.train.resume)

        # ── wandb (rank 0 only) ──────────────────────────────────────────
        if self.is_main:
            wandb_cfg = cfg.get("wandb", {})
            wandb.init(
                project = wandb_cfg.get("project", "TTT-WM"),
                name    = wandb_cfg.get("name",    cfg.experiment_name),
                config  = OmegaConf.to_container(cfg, resolve=True),
                resume  = "allow" if cfg.train.resume else None,
            )
            wandb.define_metric("epoch")
            wandb.define_metric("epoch/*", step_metric="epoch")

    # ── checkpoint ────────────────────────────────────────────────────────

    def _save_checkpoint(self, epoch: int, val_loss: float, tag: str = "last"):
        """Save unwrapped module state dict (no DDP / compile prefix)."""
        raw = unwrap_model(self.model)
        payload = {
            "epoch"       : epoch,
            "global_step" : self.global_step,
            "model"       : raw.state_dict(),
            "optimizer"   : self.optimizer.state_dict(),
            "scheduler"   : self.scheduler.state_dict(),
            "scaler"      : self.scaler.state_dict(),
            "val_loss"    : val_loss,
            "cfg"         : OmegaConf.to_container(self.cfg, resolve=True),
        }
        if self.ema is not None:
            payload["ema"] = self.ema.state_dict()
        path = self.ckpt_dir / f"{tag}.pt"
        torch.save(payload, path)
        log.info(f"Saved {path}  val_loss={val_loss:.6f}")

    def _load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)

        def _strip(sd, prefix):
            if any(k.startswith(prefix) for k in sd):
                return {k.removeprefix(prefix): v for k, v in sd.items()}
            return sd

        # Strip compile prefix '_orig_mod.' and DDP prefix 'module.' if present
        model_sd = _strip(_strip(ckpt["model"], "_orig_mod."), "module.")
        unwrap_model(self.model).load_state_dict(model_sd)

        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.scheduler.load_state_dict(ckpt["scheduler"])
        self.scaler.load_state_dict(ckpt["scaler"])
        if self.ema is not None and "ema" in ckpt:
            self.ema.load_state_dict(ckpt["ema"])
        self.start_epoch   = ckpt["epoch"] + 1
        self.global_step   = ckpt.get("global_step", 0)
        self.best_val_loss = ckpt.get("val_loss", float("inf"))
        if self.is_main:
            log.info(f"Resumed from {path}  (epoch {self.start_epoch})")

    # ── epoch ──────────────────────────────────────────────────────────────

    def _run_epoch(self, loader: DataLoader, train: bool) -> float:
        """
        Run one epoch.

        Gradient accumulation strategy
        --------------------------------
        - zero_grad() at the start of each accumulation window.
        - Loss divided by grad_accum_steps so gradient magnitude equals a
          single large-batch forward pass.
        - Under DDP, model.no_sync() suppresses the all-reduce on all but the
          last micro-step of each window → eliminates (accum_steps-1) redundant
          collective operations.
        - optimizer step / scheduler step / EMA update only on the final
          micro-step of each window (global_step increments once per window).
        """
        self.model.train(train)
        total_loss, n_opt_steps = 0.0, 0
        ema_cfg   = self.cfg.train.ema
        log_every = self.cfg.train.log_every

        pbar = tqdm(loader, desc="train" if train else "val",
                    leave=False, disable=not self.is_main)

        grad_ctx = torch.enable_grad() if train else torch.no_grad()

        # Accumulation state (only used in train mode)
        accum_loss = 0.0
        accum_step = 0

        with grad_ctx:
            for batch_idx, (context, target) in enumerate(pbar):
                context = context.to(self.device, non_blocking=True)
                target  = target.to(self.device,  non_blocking=True)

                if train:
                    is_last_accum = (
                        (accum_step == self.grad_accum_steps - 1)
                        or (batch_idx == len(loader) - 1)
                    )

                    # DDP: suppress all-reduce on intermediate micro-steps
                    ddp_sync = (
                        nullcontext()
                        if (not isinstance(self.model, DDP) or is_last_accum)
                        else self.model.no_sync()
                    )

                    with ddp_sync, torch.amp.autocast('cuda', enabled=self.use_amp):
                        _, loss = self.model(context, target)
                        scaled_loss = loss / self.grad_accum_steps

                    self.scaler.scale(scaled_loss).backward()

                    accum_loss += loss.detach().item()
                    accum_step += 1

                    if is_last_accum:
                        # ── optimiser step ──────────────────────────────
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

                        # EMA update (rank 0, unwrapped model)
                        if (self.ema is not None
                                and self.global_step % ema_cfg.update_every == 0):
                            self.ema.update(unwrap_model(self.model))

                        # Logging (rank 0)
                        if self.is_main:
                            lr       = self.scheduler.get_last_lr()[0]
                            avg_loss = accum_loss / self.grad_accum_steps
                            wandb.log({
                                "train/loss"  : avg_loss,
                                "train/lr"    : lr,
                                "global_step" : self.global_step,
                            }, step=self.global_step)
                            pbar.set_postfix(loss=f"{avg_loss:.4f}", lr=f"{lr:.2e}")
                            if self.global_step % log_every == 0:
                                log.info(
                                    f"step {self.global_step:06d} | "
                                    f"loss {avg_loss:.6f} | lr {lr:.2e}"
                                )

                        total_loss  += accum_loss / self.grad_accum_steps
                        n_opt_steps += 1
                        accum_loss   = 0.0
                        accum_step   = 0
                else:
                    # Validation — no accumulation
                    with torch.amp.autocast('cuda', enabled=self.use_amp):
                        _, loss = self.model(context, target)
                    if self.is_main:
                        pbar.set_postfix(loss=f"{loss.item():.4f}")
                    total_loss  += loss.item()
                    n_opt_steps += 1

        avg = total_loss / max(n_opt_steps, 1)

        # Average val loss across all ranks so rank 0 sees the global value
        if self.world_size > 1 and not train:
            t = torch.tensor(avg, device=self.device)
            dist.all_reduce(t, op=dist.ReduceOp.AVG)
            avg = t.item()

        return avg

    def _val_loss(self) -> float:
        """Validation loss using EMA weights if available (rank 0 only for swap)."""
        raw = unwrap_model(self.model)
        if self.ema is not None:
            self.ema.apply(raw)
        loss = self._run_epoch(self.val_loader, train=False)
        if self.ema is not None:
            self.ema.restore(raw)
        return loss

    # ── wandb video helpers ────────────────────────────────────────────────

    @staticmethod
    def _frames_to_uint8(frames: torch.Tensor) -> np.ndarray:
        x = (frames.clamp(-1, 1) * 0.5 + 0.5) * 255.0
        return x.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)

    @torch.no_grad()
    def _log_val_videos(self, n_samples: int = 4):
        """Log prediction vs target videos to wandb.  Rank 0 only."""
        if not self.is_main:
            return
        self.model.eval()
        raw = unwrap_model(self.model)
        if self.ema is not None:
            self.ema.apply(raw)

        ds      = self.val_loader.dataset
        indices = torch.randperm(len(ds))[:n_samples].tolist()
        ctx_list, tgt_list = zip(*[ds[i] for i in indices])
        context = torch.stack(ctx_list).to(self.device)
        target  = torch.stack(tgt_list).to(self.device)

        with torch.amp.autocast('cuda', enabled=self.use_amp):
            pred_frames, _ = self.model(context, target)

        if self.ema is not None:
            self.ema.restore(raw)

        n            = min(n_samples, context.shape[0])
        total_frames = context.shape[1] + target.shape[1]
        media        = {}

        for i in range(n):
            ctx_np  = self._frames_to_uint8(context[i])
            tgt_np  = self._frames_to_uint8(target[i])
            pred_np = self._frames_to_uint8(pred_frames[i])

            if total_frames <= 2:
                parts     = [ctx_np[f] for f in range(ctx_np.shape[0])]
                parts    += [tgt_np[f] for f in range(tgt_np.shape[0])]
                parts    += [pred_np[f] for f in range(pred_np.shape[0])]
                composite = np.concatenate(parts, axis=1)
                media[f"val/sample_{i}"] = wandb.Image(
                    composite, caption="context | target | prediction"
                )
            else:
                gt_vid   = np.concatenate([ctx_np, tgt_np],  axis=0).transpose(0, 3, 1, 2)
                pred_vid = np.concatenate([ctx_np, pred_np], axis=0).transpose(0, 3, 1, 2)
                media[f"val/sample_{i}_target"] = wandb.Video(gt_vid,   fps=4, format="mp4")
                media[f"val/sample_{i}_pred"]   = wandb.Video(pred_vid, fps=4, format="mp4")

        wandb.log(media, step=self.global_step)

    # ── main loop ──────────────────────────────────────────────────────────

    def train(self):
        tcfg = self.cfg.train
        if self.is_main:
            log.info(
                f"Training {tcfg.epochs} epochs | "
                f"micro_batch {tcfg.batch_size} | accum {self.grad_accum_steps} | "
                f"world_size {self.world_size}"
            )

        for epoch in range(self.start_epoch, tcfg.epochs):
            if self._train_sampler is not None:
                # Must call set_epoch so DistributedSampler re-shuffles each epoch
                self._train_sampler.set_epoch(epoch)

            t0 = time.time()

            train_loss   = self._run_epoch(self.train_loader, train=True)
            val_loss     = self._val_loss()
            val_loss_raw = self._run_epoch(self.val_loader, train=False)

            elapsed = time.time() - t0

            if self.is_main:
                ema_tag = " [EMA]" if self.ema else ""
                log.info(
                    f"epoch {epoch:04d} | train {train_loss:.6f} | "
                    f"val{ema_tag} {val_loss:.6f} | val[raw] {val_loss_raw:.6f} | "
                    f"{elapsed:.1f}s"
                )
                wandb.log({
                    "epoch"              : epoch,
                    "epoch/train_loss"   : train_loss,
                    "epoch/val_loss"     : val_loss,
                    "epoch/val_loss_raw" : val_loss_raw,
                    "epoch/lr"           : self.scheduler.get_last_lr()[0],
                }, step=self.global_step)

                self._log_val_videos(n_samples=4)
                self._save_checkpoint(epoch, val_loss, tag="last")

                if val_loss < self.best_val_loss:
                    if self.best_ckpt_path is not None and self.best_ckpt_path.exists():
                        self.best_ckpt_path.unlink()
                    self.best_val_loss  = val_loss
                    best_tag = f"best_epoch{epoch:04d}_loss{val_loss:.6f}"
                    self._save_checkpoint(epoch, val_loss, tag=best_tag)
                    self.best_ckpt_path = self.ckpt_dir / f"{best_tag}.pt"
                    wandb.run.summary["best_val_loss"] = val_loss
                    wandb.run.summary["best_epoch"]    = epoch

                if (epoch + 1) % tcfg.save_every == 0:
                    self._save_checkpoint(epoch, val_loss, tag=f"epoch_{epoch:04d}")

        if self.is_main:
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