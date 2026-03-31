from __future__ import annotations

import inspect
import json
import math
import numbers
import os
from datetime import timedelta
from pathlib import Path
from typing import Any, Callable

import torch
import torch.distributed as dist
from torch.optim.lr_scheduler import LambdaLR


def _set_dist_env_defaults() -> None:
    if not torch.cuda.is_available():
        return

    defaults = {
        "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
        "NCCL_IB_TIMEOUT": "23",
        "NCCL_IB_RETRY_CNT": "7",
    }
    for key, value in defaults.items():
        os.environ.setdefault(key, value)


def dict_apply(x: Any, func: Callable[[torch.Tensor], torch.Tensor]) -> Any:
    if isinstance(x, dict):
        return {key: dict_apply(value, func) for key, value in x.items()}
    return func(x)


def optimizer_to(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device=device)


def resolve_device(device: str) -> torch.device:
    device = str(device)
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if device.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device)


def is_dist_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    if is_dist_initialized():
        return dist.get_rank()
    return int(os.environ.get("RANK", "0"))


def get_world_size() -> int:
    if is_dist_initialized():
        return dist.get_world_size()
    return int(os.environ.get("WORLD_SIZE", "1"))


def is_main_process() -> bool:
    return get_rank() == 0


def setup_distributed() -> tuple[int, int, int]:
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        return 0, 0, 1

    if is_dist_initialized():
        return dist.get_rank(), int(os.environ.get("LOCAL_RANK", "0")), dist.get_world_size()

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ["WORLD_SIZE"])
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    _set_dist_env_defaults()
    dist.init_process_group(backend=backend, timeout=timedelta(minutes=120))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size


def cleanup_distributed() -> None:
    if not is_dist_initialized():
        return
    try:
        dist.destroy_process_group()
    except Exception:
        pass


def distributed_mean(value: float | torch.Tensor, device: torch.device) -> float:
    if isinstance(value, torch.Tensor):
        tensor = value.detach().to(device=device, dtype=torch.float32)
    else:
        tensor = torch.tensor(float(value), device=device, dtype=torch.float32)

    if is_dist_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor /= dist.get_world_size()
    return float(tensor.item())


def get_scheduler(
    name: str,
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    last_epoch: int = -1,
) -> LambdaLR:
    name = str(name).lower()
    num_warmup_steps = max(int(num_warmup_steps), 0)
    num_training_steps = max(int(num_training_steps), 1)

    if name not in {"cosine", "linear"}:
        raise ValueError(f"Unsupported scheduler: {name}")

    def lr_lambda(step: int) -> float:
        if num_warmup_steps > 0 and step < num_warmup_steps:
            return float(step + 1) / float(max(1, num_warmup_steps))

        progress_denom = max(1, num_training_steps - num_warmup_steps)
        progress = float(step - num_warmup_steps) / float(progress_denom)
        progress = min(max(progress, 0.0), 1.0)

        if name == "linear":
            return max(0.0, 1.0 - progress)

        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)


class ModelEMA:
    def __init__(
        self,
        ema_model: torch.nn.Module,
        *,
        update_after_step: int = 0,
        inv_gamma: float = 1.0,
        power: float = 0.75,
        min_value: float = 0.0,
        max_value: float = 0.9999,
    ):
        self.ema_model = ema_model
        self.update_after_step = int(update_after_step)
        self.inv_gamma = float(inv_gamma)
        self.power = float(power)
        self.min_value = float(min_value)
        self.max_value = float(max_value)

        self.ema_model.eval()
        for param in self.ema_model.parameters():
            param.requires_grad_(False)

    def _get_decay(self, optimization_step: int) -> float:
        step = max(int(optimization_step) - self.update_after_step - 1, 0)
        if step <= 0:
            return 0.0
        value = 1.0 - (1.0 + step / self.inv_gamma) ** (-self.power)
        return min(max(value, self.min_value), self.max_value)

    @torch.no_grad()
    def step(self, model: torch.nn.Module, optimization_step: int) -> None:
        decay = self._get_decay(optimization_step)

        ema_params = dict(self.ema_model.named_parameters())
        src_params = dict(model.named_parameters())
        for name, ema_param in ema_params.items():
            src_param = src_params[name]
            ema_param.lerp_(src_param.detach(), 1.0 - decay)

        ema_buffers = dict(self.ema_model.named_buffers())
        src_buffers = dict(model.named_buffers())
        for name, ema_buffer in ema_buffers.items():
            ema_buffer.copy_(src_buffers[name])


class TopKCheckpointManager:
    def __init__(
        self,
        save_dir: str,
        monitor_key: str,
        mode: str = "min",
        k: int = 1,
        format_str: str = "epoch={epoch:04d}-val_loss={val_loss:.4f}.ckpt",
    ):
        if mode not in {"min", "max"}:
            raise ValueError(f"Unsupported checkpoint mode: {mode}")
        self.save_dir = save_dir
        self.monitor_key = monitor_key
        self.mode = mode
        self.k = int(k)
        self.format_str = format_str
        self.path_value_map: dict[str, float] = {}

    def get_ckpt_path(self, metrics: dict[str, float]) -> str | None:
        if self.k <= 0 or self.monitor_key not in metrics:
            return None

        value = float(metrics[self.monitor_key])
        ckpt_path = str(Path(self.save_dir) / self.format_str.format(**metrics))

        if len(self.path_value_map) < self.k:
            self.path_value_map[ckpt_path] = value
            return ckpt_path

        sorted_items = sorted(self.path_value_map.items(), key=lambda item: item[1])
        min_path, min_value = sorted_items[0]
        max_path, max_value = sorted_items[-1]

        delete_path = None
        if self.mode == "max" and value > min_value:
            delete_path = min_path
        if self.mode == "min" and value < max_value:
            delete_path = max_path
        if delete_path is None:
            return None

        del self.path_value_map[delete_path]
        self.path_value_map[ckpt_path] = value
        delete_file = Path(delete_path)
        if delete_file.exists():
            delete_file.unlink()
        return ckpt_path


class JsonLogger:
    def __init__(
        self,
        path: str,
        filter_fn: Callable[[str, Any], bool] | None = None,
    ):
        self.path = Path(path)
        self.filter_fn = filter_fn or (
            lambda key, value: isinstance(value, numbers.Number)
        )
        self.file = None

    def __enter__(self) -> "JsonLogger":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = self.path.open("a", buffering=1)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self.file is not None:
            self.file.close()
            self.file = None

    def log(self, data: dict[str, Any]) -> None:
        if self.file is None:
            raise RuntimeError("JsonLogger is not opened.")

        filtered = {}
        for key, value in data.items():
            if not self.filter_fn(key, value):
                continue
            if isinstance(value, numbers.Integral):
                filtered[key] = int(value)
            elif isinstance(value, numbers.Number):
                filtered[key] = float(value)
        self.file.write(json.dumps(filtered) + "\n")


class NullRun:
    def log(self, *args, **kwargs) -> None:
        return None

    def finish(self) -> None:
        return None


class NullLogger:
    def log(self, *args, **kwargs) -> None:
        return None


def strip_state_dict_prefixes(state_dict: dict[str, Any]) -> dict[str, Any]:
    prefixes = ("module._orig_mod.", "_orig_mod.module.", "module.", "_orig_mod.")
    cleaned = dict(state_dict)
    for prefix in prefixes:
        if any(key.startswith(prefix) for key in cleaned):
            cleaned = {key.removeprefix(prefix): value for key, value in cleaned.items()}
    return cleaned


def load_state_dict_flexible(module: Any, state_dict: dict[str, Any]) -> None:
    load_fn = module.load_state_dict
    try:
        signature = inspect.signature(load_fn)
        if "strict" in signature.parameters:
            load_fn(state_dict, strict=False)
            return
    except (TypeError, ValueError):
        pass
    load_fn(state_dict)


def align_action_tensors(
    pred_action: torch.Tensor,
    gt_action: torch.Tensor,
    n_obs_steps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if pred_action.ndim == 3 and gt_action.ndim == 3:
        if pred_action.shape[-1] != gt_action.shape[-1]:
            raise ValueError(
                f"Action dim mismatch: pred={pred_action.shape[-1]} gt={gt_action.shape[-1]}"
            )
        if pred_action.shape[1] == gt_action.shape[1]:
            return pred_action, gt_action

        start = max(int(n_obs_steps) - 1, 0)
        end = start + pred_action.shape[1]
        if end <= gt_action.shape[1]:
            return pred_action, gt_action[:, start:end, :]

        t_common = min(pred_action.shape[1], gt_action.shape[1])
        return pred_action[:, :t_common, :], gt_action[:, :t_common, :]

    pred_flat = pred_action.reshape(pred_action.shape[0], -1)
    gt_flat = gt_action.reshape(gt_action.shape[0], -1)
    d_common = min(pred_flat.shape[1], gt_flat.shape[1])
    return pred_flat[:, :d_common], gt_flat[:, :d_common]


def resolve_checkpoint_path(path_or_dir: str) -> Path:
    path = Path(path_or_dir).expanduser()
    if path.is_file():
        return path
    if not path.is_dir():
        raise FileNotFoundError(f"Checkpoint path does not exist: {path_or_dir}")

    latest_ckpt = path / "checkpoints" / "latest.ckpt"
    if latest_ckpt.is_file():
        return latest_ckpt

    candidates = sorted((path / "checkpoints").glob("*.ckpt")) if (path / "checkpoints").is_dir() else []
    if not candidates:
        candidates = sorted(path.glob("*.ckpt"))
    candidates = [candidate for candidate in candidates if candidate.is_file()]
    if not candidates:
        raise FileNotFoundError(
            f"No checkpoint found in {path_or_dir} or {path_or_dir}/checkpoints"
        )
    return max(candidates, key=lambda item: item.stat().st_mtime)
