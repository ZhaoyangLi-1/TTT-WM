from __future__ import annotations

import copy
import math
import os
import random
from contextlib import nullcontext
from pathlib import Path

import hydra
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import tqdm
from omegaconf import OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from dp.base_workspace import BaseWorkspace
from dp.common import (
    JsonLogger,
    ModelEMA,
    NullLogger,
    NullRun,
    TopKCheckpointManager,
    align_action_tensors,
    cleanup_distributed,
    dict_apply,
    distributed_mean,
    get_scheduler,
    is_main_process,
    optimizer_to,
    resolve_device,
    setup_distributed,
)


class TrainDiffusionWorkspace(BaseWorkspace):
    include_keys = ("global_step", "epoch", "optimizer_step")
    exclude_keys = ("ddp_model",)
    wandb_exclude_keys = frozenset({"global_step", "optimizer_step"})

    def __init__(self, cfg: OmegaConf, output_dir: str | None = None):
        super().__init__(cfg, output_dir=output_dir)

        self.rank, self.local_rank, self.world_size = setup_distributed()
        self.is_main = is_main_process()
        self.ddp_model: DDP | None = None
        self.train_sampler: DistributedSampler | None = None
        self.val_sampler: DistributedSampler | None = None

        seed = int(cfg.training.seed)
        seed = seed + self.rank
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        self.model = hydra.utils.instantiate(cfg.policy)
        self.ema_model = copy.deepcopy(self.model) if cfg.training.use_ema else None
        if self.ema_model is not None:
            self.ema_model.eval()
            for param in self.ema_model.parameters():
                param.requires_grad_(False)

        self.optimizer = hydra.utils.instantiate(cfg.optimizer, params=self.model.parameters())
        self.global_step = 0
        self.optimizer_step = 0
        self.epoch = 0
        self.device = torch.device("cpu")

    def _resolve_resume_path(self, cfg: OmegaConf) -> str | None:
        if not cfg.training.resume:
            return None

        resume_path = OmegaConf.select(cfg, "training.resume_path", default=None)
        if resume_path not in (None, "", "None"):
            return str(resume_path)

        latest_ckpt = self.get_checkpoint_path()
        if latest_ckpt.is_file():
            return str(latest_ckpt)
        return None

    def _build_dataloader(self, dataset, loader_cfg: OmegaConf) -> DataLoader:
        loader_kwargs = OmegaConf.to_container(loader_cfg, resolve=True)
        shuffle = bool(loader_kwargs.pop("shuffle", False))
        num_workers = int(loader_kwargs.get("num_workers", 0))
        if num_workers == 0:
            loader_kwargs["persistent_workers"] = False
            loader_kwargs.pop("prefetch_factor", None)
        if self.world_size > 1:
            sampler = DistributedSampler(
                dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=shuffle,
                drop_last=False,
            )
            loader_kwargs["sampler"] = sampler
            loader_kwargs["shuffle"] = False
            if shuffle:
                self.train_sampler = sampler
            else:
                self.val_sampler = sampler
        else:
            loader_kwargs["shuffle"] = shuffle
        return DataLoader(dataset, **loader_kwargs)

    def _init_wandb(self, cfg: OmegaConf):
        if not self.is_main:
            return NullRun()
        logging_cfg = OmegaConf.to_container(cfg.logging, resolve=True)
        if not logging_cfg.pop("enabled", True):
            return NullRun()

        try:
            import wandb
        except ImportError:
            return NullRun()

        run = wandb.init(
            dir=str(self.output_dir),
            config=OmegaConf.to_container(cfg, resolve=True),
            **logging_cfg,
        )
        return run

    def _wandb_log(self, wandb_run, payload: dict, *, step: int) -> None:
        filtered = {
            key: value
            for key, value in payload.items()
            if key not in self.wandb_exclude_keys
        }
        wandb_run.log(filtered, step=step)

    def _format_wandb_batch_log(self, loss_value: float, lr: float) -> dict:
        return {
            "train/loss": float(loss_value),
            "train/lr": float(lr),
            "train/epoch": int(self.epoch),
        }

    def _format_wandb_epoch_log(self, step_log: dict) -> dict:
        wandb_log = {
            "train/epoch": int(step_log["epoch"]),
            "train/lr": float(step_log["lr"]),
        }
        if "val_loss" in step_log:
            wandb_log["val/loss"] = float(step_log["val_loss"])
        if "train_action_mse_error" in step_log:
            wandb_log["train/action_mse_error"] = float(
                step_log["train_action_mse_error"]
            )
        return wandb_log

    def _refresh_best_symlink(self, topk_manager) -> None:
        """Point `<ckpt_dir>/best.ckpt` at the current best topk member.

        Best = min value when `topk.mode == "min"`, else max. The symlink is
        replaced atomically (rename) so concurrent readers see either the
        old or new target, never a missing file. Falls back to a copy on
        filesystems that don't support symlinks.
        """
        items = topk_manager.path_value_map
        if not items:
            return
        reverse = topk_manager.mode == "max"
        best_path = sorted(items.items(), key=lambda kv: kv[1], reverse=reverse)[0][0]
        save_dir = Path(topk_manager.save_dir)
        link_path = save_dir / "best.ckpt"
        tmp_path = save_dir / "best.ckpt.tmp"
        target = Path(best_path)
        if not target.is_file():
            return  # ckpt save may have failed; skip rather than dangle the link
        try:
            if tmp_path.exists() or tmp_path.is_symlink():
                tmp_path.unlink()
            try:
                tmp_path.symlink_to(target.name)
                os.replace(tmp_path, link_path)
            except OSError:
                # Filesystem doesn't allow symlinks — fall back to a copy.
                import shutil
                shutil.copy2(target, tmp_path)
                os.replace(tmp_path, link_path)
        except Exception as exc:
            print(f"[train_workspace] best.ckpt symlink update failed: {exc}")

    def _resolve_training_device(self, cfg: OmegaConf) -> torch.device:
        device_name = str(cfg.training.device)
        if self.world_size > 1:
            if torch.cuda.is_available():
                return torch.device(f"cuda:{self.local_rank}")
            return torch.device("cpu")
        return resolve_device(device_name)

    def run(self) -> None:
        cfg = copy.deepcopy(self.cfg)
        wandb_run = NullRun()
        run_failed = True

        try:
            if cfg.training.debug:
                cfg.training.num_epochs = min(int(cfg.training.num_epochs), 2)
                cfg.training.max_train_steps = 3
                cfg.training.max_val_steps = 3
                cfg.training.checkpoint_every = 1
                cfg.training.val_every = 1
                cfg.training.sample_every = 1

            resume_path = self._resolve_resume_path(cfg)
            if resume_path is not None:
                print(f"Resuming from checkpoint: {resume_path}")
                self.load_checkpoint(path=resume_path)

            dataset = hydra.utils.instantiate(cfg.task.dataset)
            train_dataloader = self._build_dataloader(dataset, cfg.dataloader)
            if len(train_dataloader) == 0:
                raise RuntimeError("Training dataloader is empty. Check dataset_root and split config.")

            val_dataset = dataset.get_validation_dataset()
            val_dataloader = self._build_dataloader(val_dataset, cfg.val_dataloader)

            # Try to load cached normalizer from disk. Key the cache path by
            # action_dim so 7D (delta) and 10D (abs+rot6d) caches don't collide
            # — this prevented a hard-to-debug `shape '[-1, 7]' is invalid for
            # input of size N` crash at compute_loss after toggling abs_action.
            action_dim = int(dataset.action_shape[0])
            meta_dir = Path(dataset.root) / "meta"
            meta_dir.mkdir(parents=True, exist_ok=True)
            normalizer_cache_path = meta_dir / f"normalizer_cache_act{action_dim}.pt"
            legacy_cache_path = meta_dir / "normalizer_cache.pt"
            normalizer = None
            if self.is_main or self.world_size == 1:
                cache_to_load: Path | None = None
                if normalizer_cache_path.is_file():
                    cache_to_load = normalizer_cache_path
                elif legacy_cache_path.is_file():
                    # Tolerate the legacy unkeyed path for back-compat, but
                    # only if its action scale matches the current action_dim.
                    try:
                        state = torch.load(legacy_cache_path, map_location="cpu")
                        legacy_scale = state.get("action", {}).get("scale", None)
                        if legacy_scale is not None and legacy_scale.numel() == action_dim:
                            cache_to_load = legacy_cache_path
                        else:
                            print(
                                f"Ignoring stale normalizer cache at {legacy_cache_path}: "
                                f"scale shape {tuple(legacy_scale.shape) if legacy_scale is not None else None} "
                                f"does not match current action_dim={action_dim}."
                            )
                    except Exception as exc:
                        print(f"Ignoring unreadable legacy normalizer cache {legacy_cache_path}: {exc}")

                if cache_to_load is not None:
                    print(f"Loading cached normalizer from {cache_to_load} ...")
                    # Peek at the cache's params_dict.<obs_key>.* layout — if
                    # the obs set differs from what the model expects, rebuild
                    # rather than crash inside compute_loss with a cryptic
                    # `'ParameterDict' object has no attribute '<key>'`.
                    cached_state = torch.load(cache_to_load, map_location="cpu")
                    cached_keys = {
                        k.split(".")[1]
                        for k in cached_state.keys()
                        if k.startswith("params_dict.") and "." in k.split("params_dict.", 1)[1]
                    }
                    expected_keys = set(dict(self.model.normalizer.params_dict).keys())
                    missing = expected_keys - cached_keys
                    extra = cached_keys - expected_keys
                    if missing or extra:
                        print(
                            f"Cached normalizer key set mismatch (missing={sorted(missing)}, "
                            f"extra={sorted(extra)}); rebuilding."
                        )
                        normalizer = None
                    else:
                        normalizer = copy.deepcopy(self.model.normalizer)
                        normalizer.load_state_dict(cached_state)
                        cached_scale = normalizer["action"].params_dict["scale"]
                        if cached_scale.numel() != action_dim:
                            print(
                                f"Cached normalizer action dim {cached_scale.numel()} != dataset action_dim "
                                f"{action_dim}; rebuilding."
                            )
                            normalizer = None
                        else:
                            print("Cached normalizer loaded.")

                if normalizer is None:
                    print("Building dataset normalizer (this reads all parquet files, may be slow)...")
                    normalizer = dataset.get_normalizer()
                    torch.save(normalizer.state_dict(), normalizer_cache_path)
                    print(f"Normalizer cached to {normalizer_cache_path}")

            if self.world_size > 1:
                normalizer_state_list = [normalizer.state_dict() if normalizer is not None else None]
                dist.broadcast_object_list(normalizer_state_list, src=0)
                if not self.is_main:
                    normalizer = copy.deepcopy(self.model.normalizer)
                    normalizer.load_state_dict(normalizer_state_list[0])
            if self.is_main:
                print("Dataset normalizer ready.")
            self.model.set_normalizer(normalizer)
            if self.ema_model is not None:
                self.ema_model.set_normalizer(normalizer)

            grad_accum = max(int(cfg.training.gradient_accumulate_every), 1)
            updates_per_epoch = max(1, math.ceil(len(train_dataloader) / grad_accum))
            lr_scheduler = get_scheduler(
                cfg.training.lr_scheduler,
                optimizer=self.optimizer,
                num_warmup_steps=cfg.training.lr_warmup_steps,
                num_training_steps=updates_per_epoch * int(cfg.training.num_epochs),
                last_epoch=self.optimizer_step - 1,
            )

            ema_helper = None
            if self.ema_model is not None:
                ema_helper = ModelEMA(
                    self.ema_model,
                    update_after_step=int(cfg.ema.update_after_step),
                    inv_gamma=float(cfg.ema.inv_gamma),
                    power=float(cfg.ema.power),
                    min_value=float(cfg.ema.min_value),
                    max_value=float(cfg.ema.max_value),
                )

            device = self._resolve_training_device(cfg)
            self.device = device
            if self.is_main:
                print(f"Moving models to device: {device}")
            self.model.to(device)
            if self.ema_model is not None:
                self.ema_model.to(device)
            optimizer_to(self.optimizer, device)
            if self.world_size > 1:
                # DDP only proxies forward(). The policy has no forward() but uses
                # compute_loss() for training, so alias it so DDP can intercept it.
                self.model.forward = self.model.compute_loss
                ddp_kwargs = {}
                if device.type == "cuda":
                    ddp_kwargs["device_ids"] = [self.local_rank]
                    ddp_kwargs["output_device"] = self.local_rank
                if self.is_main:
                    print("Wrapping model with DistributedDataParallel...")
                self.ddp_model = DDP(self.model, find_unused_parameters=True, broadcast_buffers=False, **ddp_kwargs)
            train_model = self.ddp_model if self.ddp_model is not None else self.model

            if self.is_main:
                print(
                    "Initializing experiment logging"
                    f" (enabled={bool(OmegaConf.select(cfg, 'logging.enabled', default=True))},"
                    f" mode={OmegaConf.select(cfg, 'logging.mode', default='online')})..."
                )
            wandb_run = self._init_wandb(cfg)
            topk_manager = (
                TopKCheckpointManager(
                    save_dir=os.path.join(self.output_dir, "checkpoints"),
                    **OmegaConf.to_container(cfg.checkpoint.topk, resolve=True),
                )
                if self.is_main
                else None
            )

            train_sampling_batch = None
            log_path = os.path.join(self.output_dir, "logs.jsonl")
            json_logger_cm = JsonLogger(log_path) if self.is_main else nullcontext(NullLogger())

            with json_logger_cm as json_logger:
                while self.epoch < int(cfg.training.num_epochs):
                    if self.train_sampler is not None:
                        self.train_sampler.set_epoch(self.epoch)
                    if self.val_sampler is not None:
                        self.val_sampler.set_epoch(self.epoch)

                    train_model.train()
                    self.optimizer.zero_grad(set_to_none=True)

                    train_losses: list[float] = []
                    with tqdm.tqdm(
                        train_dataloader,
                        desc=f"Training epoch {self.epoch}",
                        leave=False,
                        mininterval=float(cfg.training.tqdm_interval_sec),
                        disable=not self.is_main,
                    ) as tepoch:
                        for batch_idx, batch in enumerate(tepoch):
                            batch = dict_apply(
                                batch, lambda tensor: tensor.to(device, non_blocking=True)
                            )
                            if self.is_main and train_sampling_batch is None:
                                train_sampling_batch = dict_apply(
                                    batch, lambda tensor: tensor.detach().cpu()
                                )

                            is_last_batch = batch_idx == (len(train_dataloader) - 1)
                            should_step = ((batch_idx + 1) % grad_accum == 0) or is_last_batch
                            sync_ctx = (
                                nullcontext()
                                if (self.ddp_model is None or should_step)
                                else self.ddp_model.no_sync()
                            )
                            with sync_ctx:
                                # DDP model: forward() is aliased to compute_loss(),
                                # so calling the model directly goes through DDP's
                                # gradient sync machinery.
                                raw_loss = train_model(batch) if self.ddp_model is not None else train_model.compute_loss(batch)
                                loss = raw_loss / grad_accum
                                loss.backward()
                            if should_step:
                                grad_clip = OmegaConf.select(
                                    cfg, "training.grad_clip", default=None
                                )
                                if grad_clip not in (None, "None"):
                                    torch.nn.utils.clip_grad_norm_(
                                        self.model.parameters(), float(grad_clip)
                                    )
                                self.optimizer.step()
                                self.optimizer.zero_grad(set_to_none=True)
                                lr_scheduler.step()
                                self.optimizer_step += 1
                                if ema_helper is not None:
                                    ema_helper.step(self.model, self.optimizer_step)

                            raw_loss_value = distributed_mean(raw_loss, device=device)
                            train_losses.append(raw_loss_value)
                            tepoch.set_postfix(loss=raw_loss_value, refresh=False)

                            if self.is_main:
                                current_lr = float(lr_scheduler.get_last_lr()[0])
                                batch_log = {
                                    "epoch": self.epoch,
                                    "global_step": self.global_step,
                                    "optimizer_step": self.optimizer_step,
                                    "train_loss_step": raw_loss_value,
                                    "lr": current_lr,
                                }
                                self._wandb_log(
                                    wandb_run,
                                    self._format_wandb_batch_log(raw_loss_value, current_lr),
                                    step=self.global_step,
                                )
                                json_logger.log(batch_log)
                            self.global_step += 1

                            max_train_steps = OmegaConf.select(
                                cfg, "training.max_train_steps", default=None
                            )
                            if max_train_steps is not None and (batch_idx + 1) >= int(max_train_steps):
                                break

                    step_log: dict[str, float | int] = {
                        "epoch": self.epoch,
                        "global_step": self.global_step,
                        "optimizer_step": self.optimizer_step,
                        "lr": float(lr_scheduler.get_last_lr()[0]),
                        "train_loss": float(np.mean(train_losses)) if train_losses else float("nan"),
                    }

                    eval_policy = self.ema_model if self.ema_model is not None else self.model
                    eval_policy.eval()

                    if (self.epoch % int(cfg.training.val_every)) == 0 and len(val_dataloader) > 0:
                        val_losses: list[float] = []
                        with torch.no_grad():
                            with tqdm.tqdm(
                                val_dataloader,
                                desc=f"Validation epoch {self.epoch}",
                                leave=False,
                                mininterval=float(cfg.training.tqdm_interval_sec),
                                disable=not self.is_main,
                            ) as tepoch:
                                for batch_idx, batch in enumerate(tepoch):
                                    batch = dict_apply(
                                        batch, lambda tensor: tensor.to(device, non_blocking=True)
                                    )
                                    val_loss = self.model.compute_loss(batch)
                                    val_losses.append(distributed_mean(val_loss, device=device))

                                    max_val_steps = OmegaConf.select(
                                        cfg, "training.max_val_steps", default=None
                                    )
                                    if max_val_steps is not None and (batch_idx + 1) >= int(max_val_steps):
                                        break
                        if val_losses:
                            step_log["val_loss"] = float(np.mean(val_losses))

                    if (
                        train_sampling_batch is not None
                        and (self.epoch % int(cfg.training.sample_every)) == 0
                    ):
                        with torch.no_grad():
                            sample_batch = dict_apply(
                                train_sampling_batch,
                                lambda tensor: tensor.to(device, non_blocking=True),
                            )
                            result = eval_policy.predict_action(sample_batch["obs"])
                            pred_action = result.get("action_pred", result["action"])
                            gt_action = sample_batch["action"]
                            pred_action, gt_action = align_action_tensors(
                                pred_action,
                                gt_action,
                                int(cfg.n_obs_steps),
                            )
                            step_log["train_action_mse_error"] = float(
                                F.mse_loss(pred_action, gt_action).item()
                            )

                    if self.is_main and (self.epoch % int(cfg.training.checkpoint_every)) == 0:
                        if cfg.checkpoint.save_last_ckpt:
                            self.save_checkpoint()
                        if cfg.checkpoint.save_last_snapshot:
                            self.save_snapshot()

                        metric_dict = {
                            key.replace("/", "_"): value for key, value in step_log.items()
                        }
                        topk_ckpt_path = topk_manager.get_ckpt_path(metric_dict) if topk_manager is not None else None
                        if topk_ckpt_path is not None:
                            self.save_checkpoint(path=topk_ckpt_path)

                        # Maintain a convenience `best.ckpt` symlink pointing at
                        # the lowest-val_loss (or highest, depending on `mode`)
                        # member of the topk set, so downstream eval / rollout
                        # never has to grep filenames.
                        if topk_manager is not None and topk_manager.path_value_map:
                            self._refresh_best_symlink(topk_manager)

                    self.model.train()
                    if self.is_main:
                        self._wandb_log(
                            wandb_run,
                            self._format_wandb_epoch_log(step_log),
                            step=self.global_step,
                        )
                        json_logger.log(step_log)
                    self.epoch += 1

            run_failed = False
        finally:
            if self.is_main and hasattr(wandb_run, "finish"):
                try:
                    if run_failed:
                        wandb_run.finish(exit_code=1)
                    else:
                        wandb_run.finish()
                except TypeError:
                    wandb_run.finish()
            if self._saving_thread is not None:
                self._saving_thread.join()
            cleanup_distributed()
