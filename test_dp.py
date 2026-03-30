from __future__ import annotations

import argparse
import copy
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import dill
import hydra
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from dp.common import (
    align_action_tensors,
    dict_apply,
    load_state_dict_flexible,
    resolve_checkpoint_path,
    resolve_device,
    strip_state_dict_prefixes,
)
from dp.runtime import configure_diffusion_policy_path, register_omegaconf_resolvers

register_omegaconf_resolvers()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline evaluation for TTT-WM diffusion-policy checkpoints."
    )
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--dataset-root", type=str, default=None)
    parser.add_argument("--split", type=str, choices=["train", "val"], default="val")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--num-batches", type=int, default=20)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--use-ema", dest="use_ema", action="store_true")
    parser.add_argument("--no-ema", dest="use_ema", action="store_false")
    parser.set_defaults(use_ema=True)
    parser.add_argument("--diffusion-policy-src", type=str, default=None)
    parser.add_argument("--output-json", type=str, default=None)
    parser.add_argument("--vis-dir", type=str, default=None)
    parser.add_argument("--max-mse", type=float, default=None)
    return parser.parse_args()


def load_cfg(args: argparse.Namespace, payload: dict[str, Any]) -> Any:
    if args.config is not None:
        cfg_path = Path(args.config).expanduser()
        if not cfg_path.is_file():
            raise FileNotFoundError(f"Config file does not exist: {cfg_path}")
        return OmegaConf.load(cfg_path)

    cfg = payload.get("cfg")
    if cfg is None:
        raise KeyError("Checkpoint does not contain an embedded cfg. Pass --config explicitly.")
    return copy.deepcopy(cfg)


def instantiate_policy(cfg: Any, payload: dict[str, Any], use_ema: bool):
    policy = hydra.utils.instantiate(cfg.policy)
    state_dicts = payload.get("state_dicts", {})

    candidates = []
    if use_ema and isinstance(state_dicts.get("ema_model"), dict):
        candidates.append(("ema_model", state_dicts["ema_model"]))
    if isinstance(state_dicts.get("model"), dict):
        candidates.append(("model", state_dicts["model"]))

    if not candidates:
        raise KeyError("Checkpoint does not contain `state_dicts['model']` or `state_dicts['ema_model']`.")

    last_error: Exception | None = None
    for source, state_dict in candidates:
        try:
            load_state_dict_flexible(policy, strip_state_dict_prefixes(state_dict))
            return policy, source
        except Exception as exc:
            last_error = exc

    raise RuntimeError("Failed to load policy weights from checkpoint.") from last_error


def build_dataset(cfg: Any, dataset_root: str | None, split: str):
    cfg = copy.deepcopy(cfg)
    if dataset_root is not None:
        cfg.dataset_root = dataset_root
        cfg.task.dataset.dataset_root = dataset_root
    dataset = hydra.utils.instantiate(cfg.task.dataset)
    if split == "val":
        dataset = dataset.get_validation_dataset()
    return dataset


def build_dataloader(
    cfg: Any,
    dataset,
    split: str,
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    loader_cfg = cfg.val_dataloader if split == "val" else cfg.dataloader
    loader_kwargs = OmegaConf.to_container(loader_cfg, resolve=True)
    loader_kwargs["batch_size"] = int(batch_size)
    loader_kwargs["num_workers"] = int(num_workers)
    loader_kwargs["shuffle"] = False
    if num_workers == 0:
        loader_kwargs["persistent_workers"] = False
        loader_kwargs.pop("prefetch_factor", None)
    return DataLoader(dataset, **loader_kwargs)


def save_visualization(
    pred_action: torch.Tensor,
    gt_action: torch.Tensor,
    vis_dir: Path,
) -> list[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[WARN] Skip visualization: {exc}")
        return []

    vis_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[str] = []

    pred = pred_action[0].detach().cpu().numpy()
    gt = gt_action[0].detach().cpu().numpy()
    action_dim = pred.shape[-1]

    fig, axes = plt.subplots(action_dim, 1, figsize=(10, max(3, 2 * action_dim)), sharex=True)
    if action_dim == 1:
        axes = [axes]
    for dim_idx, axis in enumerate(axes):
        axis.plot(gt[:, dim_idx], label="gt", linewidth=2)
        axis.plot(pred[:, dim_idx], label="pred", linewidth=2)
        axis.set_ylabel(f"a[{dim_idx}]")
        axis.grid(True, alpha=0.3)
    axes[0].legend(loc="upper right")
    axes[-1].set_xlabel("timestep")
    fig.tight_layout()

    plot_path = vis_dir / "action_compare.png"
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)
    saved_paths.append(str(plot_path))
    return saved_paths


def main() -> None:
    args = parse_args()

    ckpt_path = resolve_checkpoint_path(args.checkpoint)
    payload = torch.load(ckpt_path, map_location="cpu", pickle_module=dill)
    cfg = load_cfg(args, payload)

    diffusion_policy_src = args.diffusion_policy_src or OmegaConf.select(
        cfg, "runtime.diffusion_policy_src", default=None
    )
    configure_diffusion_policy_path(diffusion_policy_src)
    OmegaConf.resolve(cfg)

    device = resolve_device(args.device)
    dataset = build_dataset(cfg, args.dataset_root, args.split)
    dataloader = build_dataloader(
        cfg,
        dataset,
        split=args.split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    policy, weight_source = instantiate_policy(cfg, payload, args.use_ema)
    policy.to(device)
    policy.eval()

    metrics: list[dict[str, float]] = []
    first_pred: torch.Tensor | None = None
    first_gt: torch.Tensor | None = None

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            batch = dict_apply(batch, lambda tensor: tensor.to(device, non_blocking=True))

            loss = float(policy.compute_loss(batch).item())
            result = policy.predict_action(batch["obs"])
            pred_action = result.get("action_pred", result["action"])
            pred_action, gt_action = align_action_tensors(
                pred_action,
                batch["action"],
                int(cfg.n_obs_steps),
            )

            action_mse = float(F.mse_loss(pred_action, gt_action).item())
            action_l1 = float(F.l1_loss(pred_action, gt_action).item())
            finite_ratio = float(torch.isfinite(pred_action).float().mean().item())

            metrics.append(
                {
                    "batch_idx": float(batch_idx),
                    "loss": loss,
                    "action_mse": action_mse,
                    "action_l1": action_l1,
                    "finite_ratio": finite_ratio,
                }
            )

            if first_pred is None:
                first_pred = pred_action.detach().cpu()
                first_gt = gt_action.detach().cpu()

            if args.num_batches > 0 and (batch_idx + 1) >= args.num_batches:
                break

    if not metrics:
        raise RuntimeError("No batches were evaluated.")

    summary = {
        "checkpoint": str(ckpt_path),
        "weight_source": weight_source,
        "split": args.split,
        "num_batches": len(metrics),
        "timestamp_utc": datetime.utcnow().isoformat(timespec="seconds"),
        "mean_loss": float(sum(item["loss"] for item in metrics) / len(metrics)),
        "mean_action_mse": float(
            sum(item["action_mse"] for item in metrics) / len(metrics)
        ),
        "mean_action_l1": float(sum(item["action_l1"] for item in metrics) / len(metrics)),
        "mean_finite_ratio": float(
            sum(item["finite_ratio"] for item in metrics) / len(metrics)
        ),
        "batches": metrics,
        "visualizations": [],
    }

    if args.vis_dir is not None and first_pred is not None and first_gt is not None:
        summary["visualizations"] = save_visualization(
            first_pred,
            first_gt,
            Path(args.vis_dir).expanduser(),
        )

    if args.output_json is not None:
        output_path = Path(args.output_json).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2))

    print(json.dumps(summary, indent=2))

    if args.max_mse is not None and summary["mean_action_mse"] > float(args.max_mse):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

