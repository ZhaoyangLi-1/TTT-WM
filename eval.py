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
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms

# Import model — try Cosmos-style first, fall back to original
try:
    from cosmos_model import ARPatchConfig, ARVideoPatchTransformer
    _NEW_MODEL = True
except ImportError:
    from model import ARPatchConfig, ARVideoPatchTransformer
    _NEW_MODEL = False


def load_model_from_checkpoint(ckpt_path: str, device: torch.device, use_ema: bool = False):
    """Load model from checkpoint, reconstructing config from saved cfg."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    # Reconstruct model config from checkpoint
    cfg = ckpt["cfg"]
    mcfg = cfg["model"]

    config_kwargs = dict(
        resolution=mcfg["resolution"],
        num_channels=mcfg["num_channels"],
        patch_size=mcfg["patch_size"],
        d_model=mcfg["d_model"],
        n_heads=mcfg["n_heads"],
        n_layers=mcfg["n_layers"],
        mlp_ratio=mcfg["mlp_ratio"],
        dropout=mcfg.get("dropout", 0.0),
        frames_in=mcfg["frames_in"],
        frames_out=mcfg["frames_out"],
    )

    if _NEW_MODEL:
        config_kwargs["qk_norm"] = mcfg.get("qk_norm", True)
        config_kwargs["parallel_attn"] = mcfg.get("parallel_attn", False)

    model_cfg = ARPatchConfig(**config_kwargs)
    model = ARVideoPatchTransformer(model_cfg).to(device)

    # Load weights (EMA or live)
    if use_ema and "ema" in ckpt:
        model.load_state_dict(ckpt["ema"])
        print(f"Loaded EMA weights from {ckpt_path}")
    else:
        model.load_state_dict(ckpt["model"])
        if use_ema:
            print(f"Warning: --use-ema requested but no EMA weights in checkpoint, using live weights")
        print(f"Loaded model weights from {ckpt_path}")

    epoch = ckpt.get("epoch", "?")
    val_loss = ckpt.get("val_loss", "?")
    print(f"  epoch={epoch}, val_loss={val_loss}")
    print(f"  resolution={model_cfg.resolution}, frames_in={model_cfg.frames_in}, "
          f"frames_out={model_cfg.frames_out}, patch_size={model_cfg.patch_size}")
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  parameters: {n_params:.2f}M")

    model.eval()
    return model, model_cfg, cfg


def load_episode_frames(dataset_root: str, episode_idx: int, image_key: str = "image"):
    """Load all frames from a single LIBERO episode parquet file."""
    root = Path(dataset_root)

    with open(root / "meta" / "info.json") as f:
        info = json.load(f)
    chunks_size = info["chunks_size"]

    chunk_idx = episode_idx // chunks_size
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
        transforms.Normalize([0.5] * 3, [0.5] * 3),  # [0,1] -> [-1,1]
    ])

    frames = []
    for i in range(n_context):
        idx = start_frame + i * frame_stride
        if idx >= len(df):
            raise ValueError(
                f"Frame index {idx} out of range (episode has {len(df)} frames). "
                f"Reduce start_frame or frame_stride."
            )
        img_data = df.iloc[idx][image_key]
        png_bytes = img_data["bytes"]
        img = Image.open(io.BytesIO(png_bytes)).convert("RGB")
        frames.append(transform(img))

    return torch.stack(frames)  # (n_context, C, H, W)


def frames_to_uint8(frames: torch.Tensor) -> np.ndarray:
    """Convert frames from [-1, 1] to uint8 numpy. (T, C, H, W) -> (T, H, W, C)."""
    x = (frames.clamp(-1, 1) * 0.5 + 0.5) * 255.0
    return x.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)


def save_video(frames_uint8: np.ndarray, output_path: str, fps: int = 8):
    """Save uint8 frames (T, H, W, C) as mp4 video."""
    try:
        import imageio.v3 as iio
        iio.imwrite(output_path, frames_uint8, fps=fps, codec="libx264")
        print(f"Saved video ({len(frames_uint8)} frames, {fps} fps): {output_path}")
    except ImportError:
        try:
            import imageio
            writer = imageio.get_writer(output_path, fps=fps, codec="libx264")
            for frame in frames_uint8:
                writer.append_data(frame)
            writer.close()
            print(f"Saved video ({len(frames_uint8)} frames, {fps} fps): {output_path}")
        except ImportError:
            # Fallback: save as individual PNGs
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
    Save a side-by-side comparison video:
      - Left: ground truth (context + future GT frames)
      - Right: model output (context + generated frames)
    If no GT available, just save the generated sequence.
    """
    if gt_uint8 is not None:
        # Pad to same length
        gen_full = np.concatenate([context_uint8, generated_uint8], axis=0)
        gt_full = np.concatenate([context_uint8, gt_uint8[:generated_uint8.shape[0]]], axis=0)
        min_len = min(len(gen_full), len(gt_full))
        gen_full = gen_full[:min_len]
        gt_full = gt_full[:min_len]

        # Add text labels via border coloring
        H, W = gen_full.shape[1], gen_full.shape[2]
        separator = np.ones((min_len, H, 4, 3), dtype=np.uint8) * 128
        combined = np.concatenate([gt_full, separator, gen_full], axis=2)
        save_video(combined, output_path, fps=fps)
    else:
        full_seq = np.concatenate([context_uint8, generated_uint8], axis=0)
        save_video(full_seq, output_path, fps=fps)


def main():
    parser = argparse.ArgumentParser(description="TTT-WM Eval: autoregressive video generation")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pt checkpoint")
    parser.add_argument("--dataset", type=str, default="/scr2/zhaoyang/libero", help="LIBERO dataset root")
    parser.add_argument("--episode", type=int, default=0, help="Episode index to use")
    parser.add_argument("--start-frame", type=int, default=0, help="Starting frame index in episode")
    parser.add_argument("--n-steps", type=int, default=50, help="Number of AR generation steps")
    parser.add_argument("--frame-stride", type=int, default=1,
                        help="Stride between context frames (default: 1)")
    parser.add_argument("--image-key", type=str, default="image",
                        help="Image column in parquet (image or wrist_image)")
    parser.add_argument("--output", type=str, default=None, help="Output video path (default: auto)")
    parser.add_argument("--use-ema", action="store_true", help="Use EMA weights if available")
    parser.add_argument("--fps", type=int, default=8, help="Output video FPS")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--with-gt", action="store_true",
                        help="Include ground-truth side-by-side comparison")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device != "cuda" else "cpu")
    print(f"Device: {device}")

    # Load model
    model, model_cfg, train_cfg = load_model_from_checkpoint(args.checkpoint, device, args.use_ema)

    # Load episode
    df, image_key = load_episode_frames(args.dataset, args.episode, args.image_key)

    # Extract context frames
    n_context = model_cfg.frames_in
    context = extract_context_frames(
        df, image_key, n_context,
        start_frame=args.start_frame,
        frame_stride=args.frame_stride,
        resolution=model_cfg.resolution,
    )
    context = context.unsqueeze(0).to(device)  # (1, frames_in, C, H, W)
    print(f"Context: {n_context} frames, shape={context.shape}")

    # Generate
    print(f"Generating {args.n_steps} AR steps (each produces {model_cfg.frames_out} frames)...")
    with torch.no_grad(), torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
        generated = model.generate(context, n_steps=args.n_steps)  # (1, n_steps*frames_out, C, H, W)

    total_gen_frames = generated.shape[1]
    print(f"Generated {total_gen_frames} frames, shape={generated.shape}")

    # Convert to uint8
    context_uint8 = frames_to_uint8(context[0])      # (frames_in, H, W, C)
    generated_uint8 = frames_to_uint8(generated[0])   # (total_gen, H, W, C)

    # Optionally load ground-truth future frames for comparison
    gt_uint8 = None
    if args.with_gt:
        transform = transforms.Compose([
            transforms.Resize((model_cfg.resolution, model_cfg.resolution)),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ])
        gt_frames = []
        # Frames after context
        gt_start = args.start_frame + n_context * args.frame_stride
        for i in range(total_gen_frames):
            idx = gt_start + i
            if idx >= len(df):
                break
            img_data = df.iloc[idx][image_key]
            png_bytes = img_data["bytes"]
            img = Image.open(io.BytesIO(png_bytes)).convert("RGB")
            gt_frames.append(transform(img))
        if gt_frames:
            gt_tensor = torch.stack(gt_frames)
            gt_uint8 = frames_to_uint8(gt_tensor)
            print(f"Ground truth: {len(gt_frames)} future frames loaded for comparison")

    # Output path
    if args.output is None:
        ckpt_name = Path(args.checkpoint).stem
        args.output = f"eval_ep{args.episode}_{ckpt_name}_steps{args.n_steps}.mp4"

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    if args.with_gt and gt_uint8 is not None:
        save_comparison_video(context_uint8, generated_uint8, gt_uint8, args.output, fps=args.fps)
    else:
        full_seq = np.concatenate([context_uint8, generated_uint8], axis=0)
        save_video(full_seq, args.output, fps=args.fps)

    print("Done!")


if __name__ == "__main__":
    main()
