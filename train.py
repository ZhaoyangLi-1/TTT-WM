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
import logging
import time
from pathlib import Path

import hydra
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from ar_video_patch_transformer import ARPatchConfig, ARVideoPatchTransformer

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
        self.decay  = decay
        self.shadow = copy.deepcopy(model)
        self.shadow.eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for s, m in zip(self.shadow.parameters(), model.parameters()):
            s.data.mul_(self.decay).add_(m.data, alpha=1.0 - self.decay)
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
    Real video dataset.

        data.root/
            clip_000/
                0000.png  0001.png  ...
            clip_001/
                ...

    Each sample: sliding window of frames_in + frames_out consecutive frames.
    Returns (context, target) in [-1, 1].
    """

    def __init__(self, data_cfg: DictConfig, model_cfg: DictConfig, split: str):
        self.T   = model_cfg.frames_in + model_cfg.frames_out
        self.fin = model_cfg.frames_in
        self.transform = transforms.Compose([
            transforms.Resize((model_cfg.resolution, model_cfg.resolution)),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ])

        root      = Path(data_cfg.root)
        exts      = list(data_cfg.extensions)
        clip_dirs = sorted(d for d in root.iterdir() if d.is_dir())
        n_val     = max(1, int(len(clip_dirs) * data_cfg.val_split))
        clip_dirs = clip_dirs[n_val:] if split == "train" else clip_dirs[:n_val]

        self.samples = []
        for clip_dir in clip_dirs:
            frames = []
            for ext in exts:
                frames += sorted(clip_dir.glob(f"*.{ext}"))
            frames.sort()
            for start in range(len(frames) - self.T + 1):
                self.samples.append((frames, start))

        log.info(f"[{split}] {len(self.samples)} windows / {len(clip_dirs)} clips")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        paths, start = self.samples[idx]
        imgs = [
            self.transform(Image.open(paths[start + t]).convert("RGB"))
            for t in range(self.T)
        ]
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
        if cfg.train.device:
            self.device = torch.device(cfg.train.device)
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        log.info(f"Device: {self.device}")

        torch.manual_seed(cfg.seed)

        # ── model ───────────────────────────────────────────────────────
        mcfg = cfg.model
        self.model_cfg = ARPatchConfig(
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
        self.model = ARVideoPatchTransformer(self.model_cfg).to(self.device)
        n_params = sum(p.numel() for p in self.model.parameters()) / 1e6
        log.info(f"Parameters: {n_params:.2f} M")

        # ── EMA ─────────────────────────────────────────────────────────
        ema_cfg = cfg.train.ema
        self.ema = EMA(self.model, decay=ema_cfg.decay) if ema_cfg.enabled else None
        if self.ema:
            log.info(f"EMA enabled  decay={ema_cfg.decay}  update_every={ema_cfg.update_every}")

        # ── optimizer ───────────────────────────────────────────────────
        self.optimizer = build_optimizer(self.model, cfg.train.optimizer)

        # ── AMP ─────────────────────────────────────────────────────────
        self.use_amp = cfg.train.amp and self.device.type == "cuda"
        self.scaler  = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        # ── data ────────────────────────────────────────────────────────
        train_ds, val_ds = build_datasets(cfg.data, cfg.model)
        loader_kw = dict(
            num_workers = cfg.data.num_workers,
            pin_memory  = cfg.data.pin_memory and self.device.type == "cuda",
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
        self.global_step   = 0

        if cfg.train.resume:
            self._load_checkpoint(cfg.train.resume)

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

        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            for context, target in loader:
                context = context.to(self.device, non_blocking=True)
                target  = target.to(self.device,  non_blocking=True)

                with torch.cuda.amp.autocast(enabled=self.use_amp):
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

                    if self.global_step % log_every == 0:
                        lr = self.scheduler.get_last_lr()[0]
                        log.info(
                            f"step {self.global_step:06d} | "
                            f"loss {loss.item():.6f} | lr {lr:.2e}"
                        )

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

    # ── main loop ────────────────────────────────────────────────────────

    def train(self):
        tcfg = self.cfg.train
        log.info(f"Training {tcfg.epochs} epochs | batch {tcfg.batch_size}")
        log.info(f"\n{OmegaConf.to_yaml(self.cfg)}")

        for epoch in range(self.start_epoch, tcfg.epochs):
            t0 = time.time()

            train_loss = self._run_epoch(self.train_loader, train=True)
            val_loss   = self._val_loss()

            elapsed = time.time() - t0
            ema_tag = " [EMA]" if self.ema else ""
            log.info(
                f"epoch {epoch:04d} | "
                f"train {train_loss:.6f} | "
                f"val{ema_tag} {val_loss:.6f} | "
                f"{elapsed:.1f}s"
            )

            self._save_checkpoint(epoch, val_loss, tag="last")

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self._save_checkpoint(epoch, val_loss, tag="best")

            if (epoch + 1) % tcfg.save_every == 0:
                self._save_checkpoint(epoch, val_loss, tag=f"epoch_{epoch:04d}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

@hydra.main(config_path="configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    Trainer(cfg).train()


if __name__ == "__main__":
    main()