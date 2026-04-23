#!/usr/bin/env python3
"""Pre-compute Stage-1 (or fine-tuned Stage-2.1) backbone predictions for
every training window of a single LIBERO test task, so Stage 2.2 training
can skip the in-line AR decode and just mmap the cached predictions.

Why
----
In Stage 2.2, the backbone is frozen (`freeze_stage1=True`) and its output
for a given (ctx, goal) is a deterministic function of (weights, ctx, goal).
Every epoch, training reruns the identical AR decode for each window. This
script runs the decode once and stores the result as uint8 tensors on disk.

Cache layout (must match what ``CachedStage1PredDataset`` expects)
-----------------------------------------------------------------
<output-dir>/<task_dirname>/
    cache_meta.json
    episode_{ep_idx:06d}.stage1pred.pt   # one per held-out-task episode
        payload = {
            "pred_frames_u8": uint8 Tensor (N_windows, fout, H, W, C),
            "window_starts":  int64 Tensor (N_windows,),
        }

Usage
-----
    python scripts/prebake_stage1_pred.py \\
        --backbone-ckpt /path/to/stage2_1_best.pt \\
        --dataset-root  /scr2/zhaoyang/libero_wm \\
        --task "KITCHEN_SCENE10: put the butter at the back in the top drawer of the cabinet and close it" \\
        --output-dir    /scr2/zhaoyang/stage1_pred_cache \\
        [--batch-size 32] [--device cuda] [--dtype bf16]

One script invocation produces the cache for **one** task. Run it 3 times
(once per held-out task) or wrap in a shell loop.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf

# Make sibling modules importable when running the script directly.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from cosmos_model import ARPatchConfig, ARVideoPatchTransformer  # noqa: E402
from train_stage1 import VideoFrameDataset, _clean_state_dict  # noqa: E402

log = logging.getLogger("prebake_stage1_pred")


# ---------------------------------------------------------------------------
# Args & config
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backbone-ckpt", required=True, type=str,
                    help="Path to a Stage-1 or Stage-2.1 checkpoint (.pt).")
    ap.add_argument("--dataset-root", default="/scr2/zhaoyang/libero_wm", type=str)
    ap.add_argument("--task", required=True, type=str,
                    help='Held-out task (must match meta/test_tasks.json), '
                         'e.g. "KITCHEN_SCENE10: put the butter ...".')
    ap.add_argument("--output-dir", required=True, type=str,
                    help="Cache root; per-task subdir will be created under it.")
    ap.add_argument("--batch-size", default=32, type=int)
    ap.add_argument("--device", default="cuda", type=str)
    ap.add_argument("--dtype", default="bf16", choices=("bf16", "fp16", "fp32"),
                    help="Forward dtype for the backbone.")
    ap.add_argument("--num-workers", default=4, type=int,
                    help="DataLoader workers feeding (ctx, goal).")
    ap.add_argument("--use-ema", action="store_true", default=True,
                    help="Prefer EMA shadow weights when the ckpt has them (default).")
    ap.add_argument("--no-use-ema", dest="use_ema", action="store_false")
    ap.add_argument("--overwrite", action="store_true",
                    help="Re-prebake even if the cache dir already has data.")
    return ap


def _resolve_dtype(name: str) -> torch.dtype:
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[name]


def _task_dirname(task: str) -> str:
    return task.replace(":", "").replace(" ", "_").strip("_")


# ---------------------------------------------------------------------------
# Backbone loading (mirrors train_stage2._extract_backbone_state_dict)
# ---------------------------------------------------------------------------


def _extract_state_dict(ckpt: dict, prefer_ema: bool) -> tuple[dict, str]:
    if prefer_ema and "ema" in ckpt:
        ema_state = ckpt["ema"]
        sd = ema_state["shadow"] if "shadow" in ema_state else ema_state
        return _clean_state_dict(sd), "EMA"
    return _clean_state_dict(ckpt["model"]), "live"


def _compute_backbone_sha1(state_dict: dict) -> str:
    buf = io.BytesIO()
    ordered = {k: state_dict[k] for k in sorted(state_dict.keys())}
    torch.save(ordered, buf)
    return hashlib.sha1(buf.getvalue()).hexdigest()


def _build_model_cfg_from_ckpt(ckpt: dict) -> ARPatchConfig:
    # Stage 1 / 2.1 always save `cfg` in the checkpoint (see Trainer._save_checkpoint).
    cfg = ckpt.get("cfg", None)
    if cfg is None:
        raise ValueError(
            "Checkpoint has no saved cfg; cannot reconstruct model shape. "
            "Use a ckpt produced by train_stage1.py / train_stage2.py."
        )
    mcfg = cfg["model"]
    return ARPatchConfig(
        resolution=int(mcfg["resolution"]),
        num_channels=int(mcfg["num_channels"]),
        patch_size=int(mcfg["patch_size"]),
        d_model=int(mcfg["d_model"]),
        n_heads=int(mcfg["n_heads"]),
        n_layers=int(mcfg["n_layers"]),
        mlp_ratio=float(mcfg["mlp_ratio"]),
        dropout=float(mcfg.get("dropout", 0.0)),
        frames_in=int(mcfg["frames_in"]),
        frames_out=int(mcfg["frames_out"]),
        action_dim=int(cfg["data"].get("action_dim", 7)),
        qk_norm=bool(mcfg.get("qk_norm", True)),
        parallel_attn=bool(mcfg.get("parallel_attn", False)),
    )


# ---------------------------------------------------------------------------
# Dataset wrapper — thin index-only view that feeds the backbone batches
# grouped by episode (so per-episode output files fall out naturally).
# ---------------------------------------------------------------------------


class _PerEpisodeWindowStream:
    """Iterate over (episode_idx, list of (sample_idx, start)) groups."""

    def __init__(self, ds: VideoFrameDataset):
        self.ds = ds
        self._groups: list[tuple[int, list[tuple[int, int]]]] = []
        by_ep: dict[int, list[tuple[int, int]]] = {}
        # VideoFrameDataset.samples entries are (ep_path_str, start, length).
        # Reverse-map ep_path -> ep_idx via episode_files.
        path_to_idx = {v: k for k, v in self.ds.episode_files.items()}
        for sample_idx, (ep_path, start, _length) in enumerate(ds.samples):
            ep_idx = path_to_idx.get(ep_path)
            if ep_idx is None:
                raise RuntimeError(f"Unknown episode path: {ep_path}")
            by_ep.setdefault(ep_idx, []).append((sample_idx, int(start)))
        for ep_idx in sorted(by_ep.keys()):
            by_ep[ep_idx].sort(key=lambda pair: pair[1])
            self._groups.append((ep_idx, by_ep[ep_idx]))

    def __iter__(self):
        return iter(self._groups)

    def __len__(self):
        return len(self._groups)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


@torch.no_grad()
def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(name)s][%(levelname)s] %(message)s",
    )

    args = _build_arg_parser().parse_args()
    device = torch.device(args.device)
    dtype = _resolve_dtype(args.dtype)

    # 1) Load checkpoint & rebuild the model.
    ckpt = torch.load(args.backbone_ckpt, map_location="cpu", weights_only=False)
    sd, src = _extract_state_dict(ckpt, prefer_ema=args.use_ema)
    sha1 = _compute_backbone_sha1(sd)
    model_cfg = _build_model_cfg_from_ckpt(ckpt)
    log.info(
        f"Backbone: ckpt={args.backbone_ckpt} src={src} sha1={sha1[:12]} "
        f"cfg=fin{model_cfg.frames_in}/fout{model_cfg.frames_out}/"
        f"res{model_cfg.resolution}/patch{model_cfg.patch_size}"
    )

    model = ARVideoPatchTransformer(model_cfg).to(device=device, dtype=dtype)
    model.load_state_dict(sd)
    model.eval()

    # 2) Build the dataset with the task filter set to the single held-out task.
    saved_cfg = ckpt["cfg"]
    data_cfg_raw = dict(saved_cfg["data"])
    # Override the held-out-task filter. Split="val" then returns episodes of
    # THIS task only (see VideoFrameDataset.__init__).
    data_cfg_raw["test_tasks"] = [args.task]
    data_cfg_raw["test_task_count"] = 1
    data_cfg_raw["root"] = args.dataset_root
    # Disable goal if the run doesn't use one (cache must mirror runtime).
    use_goal = bool(data_cfg_raw.get("use_goal", True))

    data_cfg = OmegaConf.create(data_cfg_raw)
    model_cfg_dc = OmegaConf.create(dict(saved_cfg["model"]))

    ds = VideoFrameDataset(data_cfg, model_cfg_dc, split="val", is_main=True)
    if len(ds) == 0:
        raise SystemExit(f"No windows found for task {args.task!r} under {args.dataset_root}.")

    # 3) Enumerate windows grouped by episode.
    stream = _PerEpisodeWindowStream(ds)
    log.info(
        f"Task {args.task!r}: {len(ds)} windows across {len(stream)} episodes."
    )

    # 4) Output setup.
    out_task_dir = Path(args.output_dir).expanduser() / _task_dirname(args.task)
    out_task_dir.mkdir(parents=True, exist_ok=True)

    meta_path = out_task_dir / "cache_meta.json"
    if meta_path.is_file() and not args.overwrite:
        existing = json.loads(meta_path.read_text())
        if existing.get("backbone_sha1") == sha1 and existing.get("task") == args.task:
            log.info(
                f"Cache at {out_task_dir} already matches this backbone+task; "
                f"use --overwrite to regenerate."
            )
            return
        log.warning(
            f"Cache at {out_task_dir} exists with different metadata; "
            f"will overwrite per --overwrite."
        )

    cfg_fingerprint = {
        "fin": model_cfg.frames_in,
        "fout": model_cfg.frames_out,
        "resolution": model_cfg.resolution,
        "num_channels": model_cfg.num_channels,
        "frame_gap": int(data_cfg.get("frame_gap", 1)),
        "use_goal": use_goal,
        "patch_size": model_cfg.patch_size,
    }

    # 5) Main loop: for each episode, batch its windows through stage1.generate.
    total_windows = 0
    t0 = time.time()
    for ep_idx, sample_pairs in stream:
        sample_indices = [s[0] for s in sample_pairs]
        starts = [s[1] for s in sample_pairs]

        preds_for_ep: list[torch.Tensor] = []
        batch_size = int(args.batch_size)
        for i in range(0, len(sample_indices), batch_size):
            chunk = sample_indices[i : i + batch_size]
            ctx_list = []
            goal_list = []
            for s_idx in chunk:
                # VideoFrameDataset.__getitem__ returns (ctx, tgt, actions, goal)
                ctx, _tgt, _actions, goal = ds[s_idx]
                ctx_list.append(ctx)
                goal_list.append(goal)
            ctx = torch.stack(ctx_list, dim=0).to(device=device, dtype=dtype)
            goal_in = (
                torch.stack(goal_list, dim=0).to(device=device, dtype=dtype)
                if use_goal
                else None
            )
            pred = model.generate(ctx, goal=goal_in)     # (B, fout, C, H, W) in [-1,1]
            # Quantize to uint8 with the same convention the dataset uses
            # for raw frames: (x+1) * 127.5 -> [0,255].
            pred_u8 = (
                pred.float().add(1.0).mul_(127.5).clamp_(0.0, 255.0).to(torch.uint8)
            )
            # Reorder to (B, fout, H, W, C) to match the dataset's payload shape.
            pred_u8 = pred_u8.permute(0, 1, 3, 4, 2).contiguous().cpu()
            preds_for_ep.append(pred_u8)

        pred_frames_u8 = torch.cat(preds_for_ep, dim=0)       # (N_ep, fout, H, W, C)
        window_starts = torch.tensor(starts, dtype=torch.int64)
        out_path = out_task_dir / f"episode_{ep_idx:06d}.stage1pred.pt"
        torch.save(
            {
                "pred_frames_u8": pred_frames_u8,
                "window_starts": window_starts,
            },
            out_path,
        )
        total_windows += pred_frames_u8.shape[0]
        log.info(
            f"  ep {ep_idx:06d}: {pred_frames_u8.shape[0]} windows -> {out_path.name}"
        )

    elapsed = time.time() - t0

    # 6) Write metadata.
    meta = {
        "backbone_ckpt": str(Path(args.backbone_ckpt).resolve()),
        "backbone_src": src,
        "backbone_sha1": sha1,
        "task": args.task,
        "num_episodes": len(stream),
        "num_windows": int(total_windows),
        "dtype": args.dtype,
        "dataset_root": str(Path(args.dataset_root).resolve()),
        "built_in_seconds": round(elapsed, 1),
        **cfg_fingerprint,
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    log.info(
        f"Done. {total_windows} windows / {len(stream)} episodes in {elapsed:.1f}s. "
        f"Cache at {out_task_dir}"
    )


if __name__ == "__main__":
    main()
