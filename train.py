"""
train.py — ARVideoPatchTransformer training with Hydra
=======================================================

Usage
-----
    python train.py data=synthetic                           # synthetic data (default)
    python train.py data.type=real data.root=/clips          # real data
    python train.py train.batch_size=32                      # override any key
    python train.py --multirun train.optimizer.lr=1e-4,3e-4  # sweep
    python train.py train.resume=checkpoints/last.pt         # resume

EMA
---
    When train.ema.enabled=true, an EMA shadow model is maintained.
    Validation loss is computed on the EMA model.
    Checkpoints save both the live model and EMA weights.

Weight decay grouping
---------------------
    When train.optimizer.no_decay_norm=true, bias terms and all params
    inside LayerNorm / GroupNorm / Embedding get weight_decay=0.
    All other params use train.optimizer.weight_decay.
"""

import copy
import functools
import io
import json
import logging
import time
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import wandb
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

try:
    from cosmos_model import ARPatchConfig, ARVideoPatchTransformer
    _NEW_MODEL = True
except ImportError:
    from model import ARPatchConfig, ARVideoPatchTransformer
    _NEW_MODEL = False

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# EMA
# ---------------------------------------------------------------------------

class EMA:
    """
    Exponential Moving Average of model weights.

        ema_w = decay * ema_w + (1 - decay) * model_w

    Usage:
        ema = EMA(model, decay=0.999)
        # after each optimizer step:
        ema.update(model)
        # for validation:
        ema.apply(model)
        val_loss = evaluate(model)
        ema.restore(model)
    """

    def __init__(self, model: nn.Module, decay: float):
        self.decay        = decay
        self.num_updates  = 0
        self.shadow = copy.deepcopy(model)
        self.shadow.eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        # Bias-correction warmup: effective decay starts low and ramps to self.decay
        # Prevents EMA from lagging far behind early in training (DiT / Karras style)
        d = min(self.decay, (1.0 + self.num_updates) / (10.0 + self.num_updates))
        self.num_updates += 1
        for s, m in zip(self.shadow.parameters(), model.parameters()):
            s.data.mul_(d).add_(m.data, alpha=1.0 - d)
        for s_buf, m_buf in zip(self.shadow.buffers(), model.buffers()):
            s_buf.data.copy_(m_buf.data)

    def apply(self, model: nn.Module) -> None:
        """Copy EMA weights into model (call before eval)."""
        self._backup = [p.data.clone() for p in model.parameters()]
        for m, s in zip(model.parameters(), self.shadow.parameters()):
            m.data.copy_(s.data)

    def restore(self, model: nn.Module) -> None:
        """Restore live weights after apply()."""
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
    """
    AdamW with optional weight-decay parameter groups.

    When opt_cfg.no_decay_norm=true:
      - bias params          → weight_decay = 0
      - LayerNorm/GroupNorm/Embedding params → weight_decay = 0
      - everything else      → weight_decay = opt_cfg.weight_decay

    This follows the standard recipe from GPT / LLaMA training.
    """
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
# Datasets
# ---------------------------------------------------------------------------

class VideoFrameDataset(Dataset):
    """
    LeRobot-v2 parquet video dataset (e.g. LIBERO).

    Directory layout (LeRobot v2.0):
        data.root/
            meta/
                info.json          # total_episodes, total_chunks, chunks_size, …
                episodes.jsonl     # per-episode metadata (length, tasks, …)
            data/
                chunk-000/
                    episode_000000.parquet
                    episode_000001.parquet
                    …
                chunk-001/
                    …

    Each parquet row has an `image` column stored as
    ``{"bytes": <png bytes>, "path": "frame_XXXXXX.png"}``.

    Each sample is a sliding window of ``frames_in + frames_out``
    consecutive frames from one episode.  Returns (context, target) in [-1, 1].
    """

    def __init__(self, data_cfg: DictConfig, model_cfg: DictConfig, split: str):
        self.T      = model_cfg.frames_in + model_cfg.frames_out
        self.fin    = model_cfg.frames_in
        self.stride = data_cfg.get("frame_stride", 1)
        self.gap    = data_cfg.get("frame_gap", self.stride)  # extra gap between last context and first target
        self.image_key = data_cfg.get("image_key", "image")
        self.transform = transforms.Compose([
            transforms.Resize((model_cfg.resolution, model_cfg.resolution)),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ])

        root = Path(data_cfg.root)

        # ── read meta ──────────────────────────────────────────────────
        with open(root / "meta" / "info.json") as f:
            info = json.load(f)
        total_episodes = info["total_episodes"]
        chunks_size    = info["chunks_size"]

        # ── build episode file list ────────────────────────────────────
        episode_files = []
        for ep_idx in range(total_episodes):
            chunk_idx = ep_idx // chunks_size
            path = root / "data" / f"chunk-{chunk_idx:03d}" / f"episode_{ep_idx:06d}.parquet"
            episode_files.append(path)

        # ── read episode lengths from episodes.jsonl ───────────────────
        ep_lengths = {}
        with open(root / "meta" / "episodes.jsonl") as f:
            for line in f:
                rec = json.loads(line)
                ep_lengths[rec["episode_index"]] = rec["length"]

        # ── train / val split ──────────────────────────────────────────
        n_val = max(1, int(total_episodes * data_cfg.val_split))
        if split == "train":
            ep_indices = list(range(n_val, total_episodes))
        else:
            ep_indices = list(range(n_val))

        # ── build sliding-window sample index ──────────────────────────
        # Each sample is (episode_file_path_str, start_frame_index)
        self.samples = []
        # span: total frames covered by one window
        # context occupies fin*stride frames, then a gap, then fout*stride frames
        fout = self.T - self.fin
        span = (self.fin - 1) * self.stride + self.gap + (fout - 1) * self.stride + 1
        self._target_offset = (self.fin - 1) * self.stride + self.gap  # index of first target frame
        for ep_idx in ep_indices:
            length = ep_lengths[ep_idx]
            if length < span:
                continue
            ep_path_str = str(episode_files[ep_idx])
            for start in range(length - span + 1):
                self.samples.append((ep_path_str, start))

        log.info(
            f"[{split}] {len(self.samples)} windows / "
            f"{len(ep_indices)} episodes"
        )

        # LRU cache for parquet reads (avoids re-reading the same file
        # for different windows within the same episode)
        self._read_parquet = functools.lru_cache(maxsize=32)(
            lambda path: pd.read_parquet(path, columns=[self.image_key])
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        parquet_path, start = self.samples[idx]
        df = self._read_parquet(parquet_path)
        imgs = []
        for t in range(self.T):
            if t < self.fin:
                frame_idx = start + t * self.stride
            else:
                frame_idx = start + self._target_offset + (t - self.fin) * self.stride
            img_data = df.iloc[frame_idx][self.image_key]
            png_bytes = img_data["bytes"]
            img = Image.open(io.BytesIO(png_bytes)).convert("RGB")
            imgs.append(self.transform(img))
        frames = torch.stack(imgs)
        return frames[: self.fin], frames[self.fin :]


class SyntheticVideoDataset(Dataset):
    """
    Synthetic moving-dot dataset — no files needed.
    Useful for quick smoke-tests and overfitting checks.
    """

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

        # ── device ──────────────────────────────────────────────────────
        dev = cfg.train.device
        if dev and dev not in ("", "auto"):
            self.device = torch.device(dev)
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        log.info(f"Device: {self.device}")

        # ── performance flags ──────────────────────────────────────────
        if self.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32       = True
            torch.backends.cudnn.benchmark        = True
            log.info("Enabled TF32 + cuDNN benchmark")

        torch.manual_seed(cfg.seed)

        # ── model ───────────────────────────────────────────────────────
        mcfg = cfg.model

        # Base fields shared by both old and new model
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

        # New-model-only fields (read from config if present, else use dataclass defaults)
        if _NEW_MODEL:
            config_kwargs["qk_norm"]       = mcfg.get("qk_norm",       True)
            config_kwargs["parallel_attn"] = mcfg.get("parallel_attn", False)
            log.info("Using new Cosmos-style ARVideoPatchTransformer")
        else:
            log.info("Using original Emu3-style ARVideoPatchTransformer")

        self.model_cfg = ARPatchConfig(**config_kwargs)
        self.model = ARVideoPatchTransformer(self.model_cfg).to(self.device)
        n_params = sum(p.numel() for p in self.model.parameters()) / 1e6
        log.info(f"Parameters: {n_params:.2f} M")

        # ── torch.compile (H100 / SM90 benefits significantly) ─────────
        if cfg.train.get("compile", False) and hasattr(torch, "compile"):
            self.model = torch.compile(self.model)
            log.info("torch.compile enabled")

        # ── EMA ─────────────────────────────────────────────────────────
        ema_cfg = cfg.train.ema
        self.ema = EMA(self.model, decay=ema_cfg.decay) if ema_cfg.enabled else None
        if self.ema:
            log.info(f"EMA enabled  decay={ema_cfg.decay}  update_every={ema_cfg.update_every}")

        # ── optimizer ───────────────────────────────────────────────────
        self.optimizer = build_optimizer(self.model, cfg.train.optimizer)

        # ── AMP ─────────────────────────────────────────────────────────
        self.use_amp = cfg.train.amp and self.device.type == "cuda"
        self.scaler  = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        # ── data ────────────────────────────────────────────────────────
        train_ds, val_ds = build_datasets(cfg.data, cfg.model)
        n_workers = cfg.data.num_workers
        loader_kw = dict(
            num_workers        = n_workers,
            pin_memory         = cfg.data.pin_memory and self.device.type == "cuda",
            persistent_workers = cfg.data.get("persistent_workers", False) and n_workers > 0,
            prefetch_factor    = cfg.data.get("prefetch_factor", 2) if n_workers > 0 else None,
        )
        self.train_loader = DataLoader(
            train_ds, batch_size=cfg.train.batch_size,
            shuffle=True, drop_last=True, **loader_kw,
        )
        self.val_loader = DataLoader(
            val_ds, batch_size=cfg.train.batch_size * 2,
            shuffle=False, **loader_kw,
        )

        # ── scheduler ───────────────────────────────────────────────────
        total_steps = cfg.train.epochs * len(self.train_loader)
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr          = cfg.train.optimizer.lr,
            total_steps     = total_steps,
            pct_start       = cfg.train.scheduler.warmup_fraction,
            anneal_strategy = "cos",
        )

        # ── state ───────────────────────────────────────────────────────
        self.ckpt_dir      = Path(cfg.train.ckpt_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.start_epoch   = 0
        self.best_val_loss = float("inf")
        self.best_ckpt_path: Path | None = None
        self.global_step   = 0

        if cfg.train.resume:
            self._load_checkpoint(cfg.train.resume)

        # ── wandb ──────────────────────────────────────────────────────
        wandb_cfg = cfg.get("wandb", {})
        wandb.init(
            project = wandb_cfg.get("project", "TTT-WM"),
            name    = wandb_cfg.get("name", cfg.experiment_name),
            config  = OmegaConf.to_container(cfg, resolve=True),
            resume  = "allow" if cfg.train.resume else None,
        )
        # let epoch-level metrics use "epoch" as x-axis instead of global_step
        wandb.define_metric("epoch")
        wandb.define_metric("epoch/*", step_metric="epoch")

    # ── checkpoint ───────────────────────────────────────────────────────

    def _save_checkpoint(self, epoch: int, val_loss: float, tag: str = "last"):
        payload = {
            "epoch"      : epoch,
            "global_step": self.global_step,
            "model"      : self.model.state_dict(),
            "optimizer"  : self.optimizer.state_dict(),
            "scheduler"  : self.scheduler.state_dict(),
            "scaler"     : self.scaler.state_dict(),
            "val_loss"   : val_loss,
            "cfg"        : OmegaConf.to_container(self.cfg, resolve=True),
        }
        if self.ema is not None:
            payload["ema"] = self.ema.state_dict()

        path = self.ckpt_dir / f"{tag}.pt"
        torch.save(payload, path)
        log.info(f"Saved {path}  val_loss={val_loss:.6f}")

    def _load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.scheduler.load_state_dict(ckpt["scheduler"])
        self.scaler.load_state_dict(ckpt["scaler"])
        if self.ema is not None and "ema" in ckpt:
            self.ema.load_state_dict(ckpt["ema"])
        self.start_epoch   = ckpt["epoch"] + 1
        self.global_step   = ckpt.get("global_step", 0)
        self.best_val_loss = ckpt.get("val_loss", float("inf"))
        log.info(f"Resumed from {path}  (epoch {self.start_epoch})")

    # ── epoch ────────────────────────────────────────────────────────────

    def _run_epoch(self, loader: DataLoader, train: bool) -> float:
        self.model.train(train)
        total_loss, n_batches = 0.0, 0
        ema_cfg   = self.cfg.train.ema
        log_every = self.cfg.train.log_every

        pbar = tqdm(loader, desc="train" if train else "val", leave=False)
        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            for context, target in pbar:
                context = context.to(self.device, non_blocking=True)
                target  = target.to(self.device,  non_blocking=True)

                with torch.amp.autocast("cuda", enabled=self.use_amp):
                    _, loss = self.model(context, target)

                if train:
                    self.optimizer.zero_grad(set_to_none=True)
                    self.scaler.scale(loss).backward()

                    if self.cfg.train.grad_clip > 0:
                        self.scaler.unscale_(self.optimizer)
                        nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.cfg.train.grad_clip
                        )

                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.scheduler.step()
                    self.global_step += 1

                    # EMA update
                    if (self.ema is not None and
                            self.global_step % ema_cfg.update_every == 0):
                        self.ema.update(self.model)

                    # wandb log every gradient step
                    lr = self.scheduler.get_last_lr()[0]
                    wandb.log({
                        "train/loss": loss.item(),
                        "train/lr":   lr,
                        "global_step": self.global_step,
                    }, step=self.global_step)

                    # tqdm postfix
                    pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{lr:.2e}")

                    if self.global_step % log_every == 0:
                        log.info(
                            f"step {self.global_step:06d} | "
                            f"loss {loss.item():.6f} | lr {lr:.2e}"
                        )
                else:
                    pbar.set_postfix(loss=f"{loss.item():.4f}")

                total_loss += loss.item()
                n_batches  += 1

        return total_loss / max(n_batches, 1)

    def _val_loss(self) -> float:
        """Compute validation loss, using EMA weights if enabled."""
        if self.ema is not None:
            self.ema.apply(self.model)
        loss = self._run_epoch(self.val_loader, train=False)
        if self.ema is not None:
            self.ema.restore(self.model)
        return loss

    # ── wandb video helpers ──────────────────────────────────────────────

    @staticmethod
    def _frames_to_uint8(frames: torch.Tensor) -> np.ndarray:
        """
        Convert model frames from [-1, 1] to uint8 numpy array.
        frames: (T, C, H, W)  →  (T, H, W, C) uint8
        """
        x = (frames.clamp(-1, 1) * 0.5 + 0.5) * 255.0  # [0, 255]
        x = x.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)  # (T, H, W, C)
        return x

    @torch.no_grad()
    def _log_val_videos(self, n_samples: int = 4):
        """
        Run the model on a few val samples and log pred vs target media to wandb.
        - If total frames (context + target) <= 2: log side-by-side images
          (context | target | prediction) so the comparison is clear.
        - Otherwise: log videos as before.
        """
        self.model.eval()
        if self.ema is not None:
            self.ema.apply(self.model)

        # Random indices spread across val set → diverse scenes/episodes
        ds = self.val_loader.dataset
        indices = torch.randperm(len(ds))[:n_samples].tolist()
        ctx_list, tgt_list = zip(*[ds[i] for i in indices])
        context = torch.stack(ctx_list).to(self.device)
        target  = torch.stack(tgt_list).to(self.device)

        with torch.amp.autocast("cuda", enabled=self.use_amp):
            pred_frames, _ = self.model(context, target)

        if self.ema is not None:
            self.ema.restore(self.model)

        n = min(n_samples, context.shape[0])
        total_frames = context.shape[1] + target.shape[1]
        media = {}

        for i in range(n):
            ctx_np  = self._frames_to_uint8(context[i])        # (fin, H, W, C)
            tgt_np  = self._frames_to_uint8(target[i])         # (fout, H, W, C)
            pred_np = self._frames_to_uint8(pred_frames[i])    # (fout, H, W, C)

            if total_frames <= 2:
                # Few frames — log as a single side-by-side image:
                # [context | target | prediction]  with labels
                parts = []
                for f in range(ctx_np.shape[0]):
                    parts.append(ctx_np[f])
                for f in range(tgt_np.shape[0]):
                    parts.append(tgt_np[f])
                for f in range(pred_np.shape[0]):
                    parts.append(pred_np[f])
                composite = np.concatenate(parts, axis=1)  # (H, W*N, C)
                caption = "context | target | prediction"
                media[f"val/sample_{i}"] = wandb.Image(
                    composite, caption=caption,
                )
            else:
                # Enough frames — log as videos
                gt_video   = np.concatenate([ctx_np, tgt_np], axis=0)
                pred_video = np.concatenate([ctx_np, pred_np], axis=0)
                # wandb.Video expects (T, C, H, W)
                gt_video   = gt_video.transpose(0, 3, 1, 2)
                pred_video = pred_video.transpose(0, 3, 1, 2)
                media[f"val/sample_{i}_target"] = wandb.Video(
                    gt_video, fps=4, format="mp4"
                )
                media[f"val/sample_{i}_pred"] = wandb.Video(
                    pred_video, fps=4, format="mp4"
                )

        wandb.log(media, step=self.global_step)

    # ── main loop ────────────────────────────────────────────────────────

    def train(self):
        tcfg = self.cfg.train
        log.info(f"Training {tcfg.epochs} epochs | batch {tcfg.batch_size}")
        log.info(f"\n{OmegaConf.to_yaml(self.cfg)}")

        for epoch in range(self.start_epoch, tcfg.epochs):
            t0 = time.time()

            train_loss    = self._run_epoch(self.train_loader, train=True)
            val_loss      = self._val_loss()                          # EMA if enabled
            val_loss_raw  = self._run_epoch(self.val_loader, train=False)  # current weights

            elapsed = time.time() - t0
            ema_tag = " [EMA]" if self.ema else ""
            log.info(
                f"epoch {epoch:04d} | "
                f"train {train_loss:.6f} | "
                f"val{ema_tag} {val_loss:.6f} | "
                f"val[raw] {val_loss_raw:.6f} | "
                f"{elapsed:.1f}s"
            )

            # wandb epoch-level logging
            wandb.log({
                "epoch":              epoch,
                "epoch/train_loss":   train_loss,
                "epoch/val_loss":     val_loss,
                "epoch/val_loss_raw": val_loss_raw,
                "epoch/lr":        self.scheduler.get_last_lr()[0],
            }, step=self.global_step)

            # log prediction vs target videos
            self._log_val_videos(n_samples=4)

            self._save_checkpoint(epoch, val_loss, tag="last")

            if val_loss < self.best_val_loss:
                # delete previous best checkpoint
                if self.best_ckpt_path is not None and self.best_ckpt_path.exists():
                    self.best_ckpt_path.unlink()
                    log.info(f"Removed old best ckpt: {self.best_ckpt_path}")
                self.best_val_loss  = val_loss
                best_tag = f"best_epoch{epoch:04d}_loss{val_loss:.6f}"
                self._save_checkpoint(epoch, val_loss, tag=best_tag)
                self.best_ckpt_path = self.ckpt_dir / f"{best_tag}.pt"
                wandb.run.summary["best_val_loss"] = val_loss
                wandb.run.summary["best_epoch"]    = epoch

            if (epoch + 1) % tcfg.save_every == 0:
                self._save_checkpoint(epoch, val_loss, tag=f"epoch_{epoch:04d}")

        wandb.finish()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

@hydra.main(config_path="configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    Trainer(cfg).train()


if __name__ == "__main__":
    main()