"""
eval.py — Load a pretrained ARVideoPatchTransformer and autoregressively
           generate future frames from initial context, saving as video.

Usage
-----
    # Basic: generate from episode 0, 50 AR steps, save mp4
    python eval.py --checkpoint /path/to/best.pt --dataset /scr2/zhaoyang/libero

    # Custom: pick episode, number of steps, output path
    python eval.py --checkpoint /path/to/best.pt --dataset /scr2/zhaoyang/libero \
        --episode 42 --n-steps 60 --output results/ep42.mp4 --use-ema

    # Use ground-truth context stride/gap from training config
    python eval.py --checkpoint /path/to/best.pt --dataset /scr2/zhaoyang/libero \
        --frame-stride 3 --frame-gap 8 --start-frame 0

    # Force Cosmos model (skip importing model.py entirely)
    python eval.py --checkpoint /path/to/best.pt --dataset /scr2/zhaoyang/libero \
        --force-cosmos

Checkpoint compatibility
------------------------
Handles checkpoints saved by:
  - Original single-GPU training  (no prefix)
  - torch.compile                 (_orig_mod. prefix)
  - DDP training                  (module. prefix)
  - DDP + compile                 (module._orig_mod. or _orig_mod.module. prefix)
All prefixes are stripped automatically before weight loading.
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms

# ---------------------------------------------------------------------------
# Lazy imports: only import what we need based on checkpoint arch
# ---------------------------------------------------------------------------

def _import_emu():
    from model import ARPatchConfig, ARVideoPatchTransformer
    return ARPatchConfig, ARVideoPatchTransformer


def _import_cosmos():
    from cosmos_model import ARPatchConfig, ARVideoPatchTransformer
    return ARPatchConfig, ARVideoPatchTransformer


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------

def _sniff_arch(state_dict: dict) -> str:
    """
    Infer model architecture from state_dict key names.

    Cosmos keys use:  RMSNorm '.scale', 'embed_norm', 'blocks.N.attn.out',
                      SwiGLU 'gate_proj'/'up_proj'/'down_proj', 'out_norm'
    Emu3 keys use:    LayerNorm '.weight'/'.bias', 'pos_emb', 'transformer.blocks',
                      standard MLP 'fc1'/'fc2', 'norm_out'

    Returns 'cosmos' or 'emu3'.
    """
    cosmos_signals = {"embed_norm.scale", "out_norm.scale"}
    emu3_signals   = {"pos_emb.frame_emb.weight", "norm_out.weight"}

    keys = set(state_dict.keys())
    if cosmos_signals & keys:
        return "cosmos"
    if emu3_signals & keys:
        return "emu3"

    # Fallback: check for characteristic sub-strings in any key
    all_keys = " ".join(keys)
    if "gate_proj" in all_keys or "q_norm.scale" in all_keys:
        return "cosmos"
    if "transformer.blocks" in all_keys or "pos_emb" in all_keys:
        return "emu3"

    return "unknown"   # caller will handle


def _clean_state_dict(state_dict: dict) -> dict:
    """
    Strip all wrapper prefixes that DDP and/or torch.compile may add.

    Handled prefixes (stripped in priority order so combined prefixes work):
      'module._orig_mod.'  — DDP wrapping a compiled model
      '_orig_mod.module.'  — compiled model wrapped with DDP (unusual but possible)
      'module.'            — DistributedDataParallel
      '_orig_mod.'         — torch.compile
    """
    prefixes = ["module._orig_mod.", "_orig_mod.module.", "module.", "_orig_mod."]
    for prefix in prefixes:
        if any(k.startswith(prefix) for k in state_dict):
            state_dict = {k.removeprefix(prefix): v for k, v in state_dict.items()}
    return state_dict


def _resolve_weight_state(
    ckpt: dict,
    use_ema: bool | None,
) -> tuple[dict, str]:
    has_ema = "ema" in ckpt

    if use_ema is None:
        if has_ema:
            ema_state = ckpt["ema"]
            raw_sd = ema_state["shadow"] if "shadow" in ema_state else ema_state
            print("Detected EMA weights in checkpoint; using EMA by default.")
            return raw_sd, "EMA"
        print("No EMA weights found in checkpoint; using live weights.")
        return ckpt["model"], "live"

    if use_ema:
        if has_ema:
            ema_state = ckpt["ema"]
            raw_sd = ema_state["shadow"] if "shadow" in ema_state else ema_state
            return raw_sd, "EMA"
        print("Warning: --use-ema requested but no EMA weights found; using live weights")

    return ckpt["model"], "live"


def load_model_from_checkpoint(
    ckpt_path: str,
    device: torch.device,
    use_ema: bool | None = False,
    force_cosmos: bool = False,
):
    """Load model from checkpoint, reconstructing config from saved cfg."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    raw_sd, weight_source = _resolve_weight_state(ckpt, use_ema)

    cfg  = ckpt["cfg"]
    mcfg = cfg["model"]

    # ── Determine architecture ────────────────────────────────────────────
    # Priority: --force-cosmos flag  >  cfg["model"]["arch"] field  >
    #           state_dict key sniffing  (handles checkpoints saved before
    #           "arch" was written into the config).
    if force_cosmos:
        use_cosmos  = True
        arch_source = "--force-cosmos flag"
    elif "arch" in mcfg:
        use_cosmos  = (mcfg["arch"] == "cosmos")
        arch_source = 'cfg["model"]["arch"] = ' + mcfg["arch"]
    else:
        # cfg has no "arch" key — sniff from the actual weights
        sniffed = _sniff_arch(_clean_state_dict(raw_sd))
        if sniffed == "unknown":
            print(
                "Warning: could not auto-detect architecture from state_dict keys. "
                "Defaulting to Cosmos. Pass --force-cosmos to suppress this warning."
            )
            sniffed = "cosmos"
        use_cosmos  = (sniffed == "cosmos")
        arch_source = f"state_dict key sniffing → {sniffed}"


    config_kwargs = dict(
        resolution   = mcfg["resolution"],
        num_channels = mcfg["num_channels"],
        patch_size   = mcfg["patch_size"],
        d_model      = mcfg["d_model"],
        n_heads      = mcfg["n_heads"],
        n_layers     = mcfg["n_layers"],
        mlp_ratio    = mcfg["mlp_ratio"],
        dropout      = mcfg.get("dropout", 0.0),
        frames_in    = mcfg["frames_in"],
        frames_out   = mcfg["frames_out"],
    )

    if use_cosmos:
        # Cosmos-specific config fields
        config_kwargs["qk_norm"]       = mcfg.get("qk_norm",       True)
        config_kwargs["parallel_attn"] = mcfg.get("parallel_attn", False)
        ARPatchConfig, ARVideoPatchTransformer = _import_cosmos()
        arch_label = "Cosmos"
    else:
        ARPatchConfig, ARVideoPatchTransformer = _import_emu()
        arch_label = "Emu3"

    print(f"Architecture: {arch_label} (source: {arch_source})")

    model_cfg = ARPatchConfig(**config_kwargs)
    model     = ARVideoPatchTransformer(model_cfg).to(device)

    model.load_state_dict(_clean_state_dict(raw_sd))
    print(f"Loaded {weight_source} weights from {ckpt_path}")

    epoch    = ckpt.get("epoch",    "?")
    val_loss = ckpt.get("val_loss", "?")
    print(f"  epoch={epoch}, val_loss={val_loss}")
    print(
        f"  resolution={model_cfg.resolution}, "
        f"frames_in={model_cfg.frames_in}, "
        f"frames_out={model_cfg.frames_out}, "
        f"patch_size={model_cfg.patch_size}"
    )
    if use_cosmos:
        print(
            f"  qk_norm={model_cfg.qk_norm}, "
            f"parallel_attn={model_cfg.parallel_attn}"
        )
    print(f"  parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    model.eval()
    return model, model_cfg, cfg


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_episode_frames(dataset_root: str, episode_idx: int, image_key: str = "image"):
    """Load all frames from a single LIBERO episode parquet file."""
    root = Path(dataset_root)
    with open(root / "meta" / "info.json") as f:
        info = json.load(f)
    chunks_size  = info["chunks_size"]
    chunk_idx    = episode_idx // chunks_size
    parquet_path = root / "data" / f"chunk-{chunk_idx:03d}" / f"episode_{episode_idx:06d}.parquet"

    if not parquet_path.exists():
        raise FileNotFoundError(f"Episode parquet not found: {parquet_path}")

    df = pd.read_parquet(parquet_path, columns=[image_key])
    print(f"Loaded episode {episode_idx}: {len(df)} frames from {parquet_path}")
    return df, image_key


def extract_context_frames(
    df: pd.DataFrame,
    image_key: str,
    n_context: int,
    start_frame: int,
    frame_stride: int,
    resolution: int,
) -> torch.Tensor:
    """Extract and preprocess context frames from a dataframe."""
    transform = transforms.Compose([
        transforms.Resize((resolution, resolution)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])

    frames = []
    for i in range(n_context):
        idx = start_frame + i * frame_stride
        if idx >= len(df):
            raise ValueError(
                f"Frame index {idx} out of range (episode has {len(df)} frames). "
                f"Reduce --start-frame or --frame-stride."
            )
        img_data  = df.iloc[idx][image_key]
        img       = Image.open(io.BytesIO(img_data["bytes"])).convert("RGB")
        frames.append(transform(img))

    return torch.stack(frames)   # (n_context, C, H, W)


# ---------------------------------------------------------------------------
# Video I/O
# ---------------------------------------------------------------------------

def frames_to_uint8(frames: torch.Tensor) -> np.ndarray:
    """(T, C, H, W) in [-1,1] → (T, H, W, C) uint8."""
    x = (frames.clamp(-1, 1) * 0.5 + 0.5) * 255.0
    return x.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)


def save_video(frames_uint8: np.ndarray, output_path: str, fps: int = 8):
    """
    Save uint8 frames (T, H, W, C) as mp4 (or gif if path ends with .gif).

    Tries imageio.v3 first, then imageio v2, then falls back to individual PNGs.
    For mp4 output the codec is libx264 with yuv420p for broad player compatibility.
    """
    output_path = str(output_path)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # ── imageio v3 ────────────────────────────────────────────────────────
    try:
        import imageio.v3 as iio
        ext = Path(output_path).suffix.lower()
        if ext == ".gif":
            iio.imwrite(output_path, frames_uint8, loop=0)
        else:
            iio.imwrite(
                output_path, frames_uint8,
                fps=fps,
                codec="libx264",
                output_params=["-pix_fmt", "yuv420p"],   # broad player compat
            )
        print(f"Saved video ({len(frames_uint8)} frames @ {fps} fps): {output_path}")
        return
    except ImportError:
        pass
    except Exception as e:
        print(f"imageio.v3 failed ({e}), trying imageio v2 …")

    # ── imageio v2 ────────────────────────────────────────────────────────
    try:
        import imageio
        writer = imageio.get_writer(
            output_path, fps=fps, codec="libx264",
            output_params=["-pix_fmt", "yuv420p"],
        )
        for frame in frames_uint8:
            writer.append_data(frame)
        writer.close()
        print(f"Saved video ({len(frames_uint8)} frames @ {fps} fps): {output_path}")
        return
    except ImportError:
        pass
    except Exception as e:
        print(f"imageio v2 also failed ({e}), saving PNGs instead …")

    # ── PNG fallback ──────────────────────────────────────────────────────
    out_dir = Path(output_path).with_suffix("")
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, frame in enumerate(frames_uint8):
        Image.fromarray(frame).save(out_dir / f"frame_{i:04d}.png")
    print(f"imageio not available. Saved {len(frames_uint8)} PNGs to {out_dir}/")


def save_comparison_video(
    context_uint8: np.ndarray,
    generated_uint8: np.ndarray,
    gt_uint8: np.ndarray | None,
    output_path: str,
    fps: int = 8,
):
    """
    Save a side-by-side video:
      Left : ground truth  (context + future GT frames)
      Right: model output  (context + generated frames)
    Falls back to single sequence video if no GT available.
    """
    if gt_uint8 is not None:
        gen_full  = np.concatenate([context_uint8, generated_uint8], axis=0)
        gt_full   = np.concatenate([context_uint8, gt_uint8[:generated_uint8.shape[0]]], axis=0)
        min_len   = min(len(gen_full), len(gt_full))
        H, W      = gen_full.shape[1], gen_full.shape[2]
        separator = np.ones((min_len, H, 4, 3), dtype=np.uint8) * 128
        combined  = np.concatenate([gt_full[:min_len], separator, gen_full[:min_len]], axis=2)
        save_video(combined, output_path, fps=fps)
    else:
        save_video(np.concatenate([context_uint8, generated_uint8], axis=0), output_path, fps=fps)


# ---------------------------------------------------------------------------
# AR generation with progress bar
# ---------------------------------------------------------------------------

def generate_with_progress(
    model,
    context: torch.Tensor,
    n_steps: int,
    device: torch.device,
    use_amp: bool = True,
) -> torch.Tensor:
    """
    Autoregressive generation with an optional tqdm progress bar.

    For Cosmos models the inner frame-by-frame loop is exposed so each AR step
    increments the bar.  Falls back to model.generate() if internals differ.
    """
    try:
        from tqdm import tqdm
        has_tqdm = True
    except ImportError:
        has_tqdm = False
        print("  (install tqdm for a progress bar)")

    cfg = model.cfg

    # import unpatchify once, outside the generation loop
    from cosmos_model import unpatchify as _unpatchify

    def _run_steps():
        """Core AR loop — identical to cosmos_model.generate()."""
        window        = context[:, -cfg.frames_in:].clone()
        all_generated = []

        iterator = range(n_steps)
        if has_tqdm:
            iterator = tqdm(
                iterator, desc="Generating", unit="step",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} steps "
                           "[{elapsed}<{remaining}, {rate_fmt}]",
            )

        for _ in iterator:
            step_frames = []
            cur_window  = window

            for _ in range(cfg.frames_out):
                T         = cur_window.shape[1]
                tokens_in = model._embed(cur_window)
                hidden    = model._run_transformer(tokens_in, T)

                last_hidden  = hidden[:, -cfg.n_patches:, :]
                pred_patches = model._decode(last_hidden)

                frame = _unpatchify(
                    pred_patches, cfg.patch_size,
                    cfg.resolution, cfg.num_channels
                ).clamp(-1, 1)                             # (B, 1, C, H, W)

                step_frames.append(frame)
                cur_window = torch.cat([cur_window, frame], dim=1)[:, -cfg.frames_in:]

            step_pred = torch.cat(step_frames, dim=1)      # (B, frames_out, C, H, W)
            all_generated.append(step_pred)
            window = torch.cat([window, step_pred], dim=1)[:, -cfg.frames_in:]

        return torch.cat(all_generated, dim=1)

    # Check whether model exposes the internal helpers needed for the step loop.
    # If not (e.g. an Emu3 model), fall back to model.generate().
    has_internals = (
        hasattr(model, "_embed")
        and hasattr(model, "_run_transformer")
        and hasattr(model, "_decode")
    )

    with torch.no_grad(), torch.amp.autocast(
        'cuda', enabled=(use_amp and device.type == "cuda")
    ):
        if has_internals:
            return _run_steps()
        else:
            print("  Model does not expose internal helpers; using model.generate() directly.")
            return model.generate(context, n_steps=n_steps)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="TTT-WM Eval: autoregressive video generation")
    parser.add_argument("--checkpoint",    type=str, required=True)
    parser.add_argument("--dataset",       type=str, default="/scr2/zhaoyang/libero")
    parser.add_argument("--episode",       type=int, default=0)
    parser.add_argument("--start-frame",   type=int, default=0)
    parser.add_argument("--n-steps",       type=int, default=50,
                        help="AR steps; each produces frames_out frames")
    parser.add_argument("--frame-stride",  type=int, default=1)
    parser.add_argument("--image-key",     type=str, default="image",
                        help="Parquet column: 'image' or 'wrist_image'")
    parser.add_argument("--output",        type=str, default=None,
                        help="Output path (.mp4 or .gif). Default: auto-named mp4.")
    parser.add_argument("--use-ema",       action="store_true")
    parser.add_argument("--fps",           type=int, default=8)
    parser.add_argument("--device",        type=str, default="cuda")
    parser.add_argument("--with-gt",       action="store_true",
                        help="Side-by-side comparison with ground truth")
    parser.add_argument("--resolution",    type=int, default=None,
                        help="Override eval resolution, e.g. 224 or 256 "
                             "(must be divisible by patch_size, usually 8). "
                             "Default: use training resolution from checkpoint.")
    parser.add_argument("--force-cosmos",  action="store_true",
                        help="Force loading as Cosmos model (skips model.py import)")
    parser.add_argument("--no-amp",        action="store_true",
                        help="Disable automatic mixed precision (AMP)")
    args = parser.parse_args()

    device = torch.device(
        args.device if (torch.cuda.is_available() or args.device != "cuda") else "cpu"
    )
    print(f"Device: {device}")

    # ── model ────────────────────────────────────────────────────────────
    model, model_cfg, train_cfg = load_model_from_checkpoint(
        args.checkpoint, device, args.use_ema, args.force_cosmos
    )

    if args.resolution is not None:
        P = model_cfg.patch_size
        if args.resolution % P != 0:
            lo = (args.resolution // P) * P
            hi = lo + P
            raise ValueError(
                f"--resolution {args.resolution} is not divisible by "
                f"patch_size={P}. Use {lo} or {hi} instead."
            )
        old_res              = model_cfg.resolution
        model_cfg.resolution = args.resolution
        old_np               = (old_res // P) ** 2
        new_np               = model_cfg.n_patches
        print(
            f"Resolution override: {old_res} → {args.resolution} "
            f"(patches/frame: {old_np} → {new_np}, seq_len ×{new_np/old_np:.1f})"
        )

    use_amp = not args.no_amp

    # ── data ─────────────────────────────────────────────────────────────
    df, image_key = load_episode_frames(args.dataset, args.episode, args.image_key)

    context = extract_context_frames(
        df, image_key, model_cfg.frames_in,
        start_frame  = args.start_frame,
        frame_stride = args.frame_stride,
        resolution   = model_cfg.resolution,
    ).unsqueeze(0).to(device)   # (1, frames_in, C, H, W)
    print(f"Context: {model_cfg.frames_in} frames, shape={context.shape}")

    # ── generate ─────────────────────────────────────────────────────────
    total_output_frames = args.n_steps * model_cfg.frames_out
    print(
        f"\nGenerating {args.n_steps} AR steps "
        f"× {model_cfg.frames_out} frames/step "
        f"= {total_output_frames} frames total …"
    )

    generated = generate_with_progress(
        model, context, args.n_steps, device, use_amp=use_amp
    )

    total_gen = generated.shape[1]
    print(f"\nGenerated {total_gen} frames, shape={generated.shape}")

    context_uint8   = frames_to_uint8(context[0])
    generated_uint8 = frames_to_uint8(generated[0])

    # ── optional GT comparison ────────────────────────────────────────────
    gt_uint8 = None
    if args.with_gt:
        transform = transforms.Compose([
            transforms.Resize((model_cfg.resolution, model_cfg.resolution)),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ])
        gt_frames = []
        gt_start  = args.start_frame + model_cfg.frames_in * args.frame_stride
        for i in range(total_gen):
            idx = gt_start + i
            if idx >= len(df):
                break
            img = Image.open(io.BytesIO(df.iloc[idx][image_key]["bytes"])).convert("RGB")
            gt_frames.append(transform(img))
        if gt_frames:
            gt_uint8 = frames_to_uint8(torch.stack(gt_frames))
            print(f"Ground truth: {len(gt_frames)} future frames loaded for comparison")
        else:
            print("Warning: no ground-truth frames available after context window.")

    # ── determine output path ─────────────────────────────────────────────
    if args.output is None:
        ckpt_stem   = Path(args.checkpoint).stem
        res_tag     = f"_res{model_cfg.resolution}" if args.resolution is not None else ""
        suffix      = "_gt" if (args.with_gt and gt_uint8 is not None) else ""
        args.output = (
            f"eval_ep{args.episode}_{ckpt_stem}_steps{args.n_steps}{res_tag}{suffix}.mp4"
        )

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    # ── save video ────────────────────────────────────────────────────────
    if args.with_gt and gt_uint8 is not None:
        save_comparison_video(
            context_uint8, generated_uint8, gt_uint8,
            args.output, fps=args.fps,
        )
    else:
        save_video(
            np.concatenate([context_uint8, generated_uint8], axis=0),
            args.output,
            fps=args.fps,
        )

    print("\nDone!")
    print(f"  Output  : {Path(args.output).resolve()}")
    print(f"  Frames  : {len(context_uint8)} context + {len(generated_uint8)} generated")
    print(
        f"  Duration: "
        f"{(len(context_uint8) + len(generated_uint8)) / args.fps:.1f}s "
        f"@ {args.fps} fps"
    )


if __name__ == "__main__":
    main()
