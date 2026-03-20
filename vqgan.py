from typing import Optional, Tuple
from functools import cached_property
import pickle
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class VQGANConfig:
    resolution: int = 256
    num_channels: int = 3
    hidden_channels: int = 128
    channel_mult: Tuple[int, ...] = (1, 2, 2, 4, 6)
    num_res_blocks: int = 2
    attn_resolutions: Tuple[int, ...] = ()
    no_attn_mid_block: bool = True
    z_channels: int = 64
    num_embeddings: int = 8192
    quantized_embed_dim: int = 64
    dropout: float = 0.0
    resample_with_conv: bool = True
    commitment_cost: float = 0.25

    @property
    def num_resolutions(self):
        return len(self.channel_mult)


# ---------------------------------------------------------------------------
# Basic blocks
# ---------------------------------------------------------------------------

class ResnetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: Optional[int] = None,
                 use_conv_shortcut: bool = False, dropout_prob: float = 0.0):
        super().__init__()
        out_channels = out_channels or in_channels
        self.norm1 = nn.GroupNorm(32, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(32, out_channels)
        self.dropout = nn.Dropout(dropout_prob)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        self.shortcut = None
        if out_channels != in_channels:
            if use_conv_shortcut:
                self.shortcut = nn.Conv2d(in_channels, out_channels, 3, padding=1)
            else:
                self.shortcut = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm1(x)
        x = F.silu(x)
        x = self.conv1(x)
        x = self.norm2(x)
        x = F.silu(x)
        x = self.dropout(x)
        x = self.conv2(x)
        if self.shortcut is not None:
            residual = self.shortcut(residual)
        return x + residual


class AttnBlock(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.norm = nn.GroupNorm(32, in_channels)
        self.q = nn.Conv2d(in_channels, in_channels, 1)
        self.k = nn.Conv2d(in_channels, in_channels, 1)
        self.v = nn.Conv2d(in_channels, in_channels, 1)
        self.proj_out = nn.Conv2d(in_channels, in_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        B, C, H, W = x.shape
        q = self.q(x).reshape(B, C, -1)          # B C HW
        k = self.k(x).reshape(B, C, -1)
        v = self.v(x).reshape(B, C, -1)

        # attention: BQK
        attn = torch.einsum("bci,bcj->bij", q, k) * (C ** -0.5)
        attn = F.softmax(attn, dim=-1)
        out = torch.einsum("bij,bcj->bci", attn, v)  # B C HW
        out = out.reshape(B, C, H, W)
        out = self.proj_out(out)
        return out + residual


class Downsample(nn.Module):
    def __init__(self, in_channels: int, with_conv: bool):
        super().__init__()
        self.with_conv = with_conv
        if with_conv:
            # pad right/bottom by 1 then stride-2 conv (matches JAX asymmetric pad)
            self.conv = nn.Conv2d(in_channels, in_channels, 3, stride=2, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.with_conv:
            x = F.pad(x, (0, 1, 0, 1))   # left, right, top, bottom
            x = self.conv(x)
        else:
            x = F.avg_pool2d(x, 2, 2)
        return x


class Upsample(nn.Module):
    def __init__(self, in_channels: int, with_conv: bool):
        super().__init__()
        self.with_conv = with_conv
        if with_conv:
            self.conv = nn.Conv2d(in_channels, in_channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


# ---------------------------------------------------------------------------
# Encoder / Decoder blocks
# ---------------------------------------------------------------------------

class DownsamplingBlock(nn.Module):
    def __init__(self, config: VQGANConfig, block_idx: int, in_channels: int):
        super().__init__()
        block_out = config.hidden_channels * config.channel_mult[block_idx]
        layers = []
        attn_layers = []
        ch = in_channels
        for _ in range(config.num_res_blocks):
            layers.append(ResnetBlock(ch, block_out, dropout_prob=config.dropout))
            attn_layers.append(
                AttnBlock(block_out) if (block_out // config.hidden_channels) in config.attn_resolutions
                else nn.Identity()
            )
            ch = block_out
        self.resnets = nn.ModuleList(layers)
        self.attns = nn.ModuleList(attn_layers)

        self.downsample = None
        if block_idx != config.num_resolutions - 1:
            self.downsample = Downsample(ch, config.resample_with_conv)
        self.out_channels = ch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for resnet, attn in zip(self.resnets, self.attns):
            x = resnet(x)
            x = attn(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x


class UpsamplingBlock(nn.Module):
    def __init__(self, config: VQGANConfig, block_idx: int, in_channels: int):
        super().__init__()
        block_out = config.hidden_channels * config.channel_mult[block_idx]
        layers = []
        attn_layers = []
        ch = in_channels
        for _ in range(config.num_res_blocks + 1):
            layers.append(ResnetBlock(ch, block_out, dropout_prob=config.dropout))
            attn_layers.append(
                AttnBlock(block_out) if (block_out // config.hidden_channels) in config.attn_resolutions
                else nn.Identity()
            )
            ch = block_out
        self.resnets = nn.ModuleList(layers)
        self.attns = nn.ModuleList(attn_layers)

        self.upsample = None
        if block_idx != 0:
            self.upsample = Upsample(ch, config.resample_with_conv)
        self.out_channels = ch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for resnet, attn in zip(self.resnets, self.attns):
            x = resnet(x)
            x = attn(x)
        if self.upsample is not None:
            x = self.upsample(x)
        return x


class MidBlock(nn.Module):
    def __init__(self, in_channels: int, no_attn: bool, dropout: float):
        super().__init__()
        self.resnet1 = ResnetBlock(in_channels, dropout_prob=dropout)
        self.attn = nn.Identity() if no_attn else AttnBlock(in_channels)
        self.resnet2 = ResnetBlock(in_channels, dropout_prob=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.resnet1(x)
        x = self.attn(x)
        x = self.resnet2(x)
        return x


# ---------------------------------------------------------------------------
# Encoder & Decoder
# ---------------------------------------------------------------------------

class Encoder(nn.Module):
    def __init__(self, config: VQGANConfig):
        super().__init__()
        self.conv_in = nn.Conv2d(config.num_channels, config.hidden_channels, 3, padding=1)

        blocks = []
        ch = config.hidden_channels
        for i_level in range(config.num_resolutions):
            block = DownsamplingBlock(config, i_level, ch)
            blocks.append(block)
            ch = block.out_channels
        self.blocks = nn.ModuleList(blocks)

        self.mid = MidBlock(ch, config.no_attn_mid_block, config.dropout)
        self.norm_out = nn.GroupNorm(32, ch)
        self.conv_out = nn.Conv2d(ch, config.z_channels, 3, padding=1)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # pixel_values: (B, C, H, W)  — channels-first in PyTorch
        x = self.conv_in(pixel_values)
        for block in self.blocks:
            x = block(x)
        x = self.mid(x)
        x = self.norm_out(x)
        x = F.silu(x)
        x = self.conv_out(x)
        return x


class Decoder(nn.Module):
    def __init__(self, config: VQGANConfig):
        super().__init__()
        ch_last = config.hidden_channels * config.channel_mult[config.num_resolutions - 1]
        self.conv_in = nn.Conv2d(config.z_channels, ch_last, 3, padding=1)
        self.mid = MidBlock(ch_last, config.no_attn_mid_block, config.dropout)

        blocks = []
        ch = ch_last
        for i_level in reversed(range(config.num_resolutions)):
            block = UpsamplingBlock(config, i_level, ch)
            blocks.append(block)
            ch = block.out_channels
        self.blocks = nn.ModuleList(blocks)

        self.norm_out = nn.GroupNorm(32, ch)
        self.conv_out = nn.Conv2d(ch, config.num_channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_in(x)
        x = self.mid(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm_out(x)
        x = F.silu(x)
        x = self.conv_out(x)
        return x


# ---------------------------------------------------------------------------
# Vector Quantizer
# ---------------------------------------------------------------------------

class VectorQuantizer(nn.Module):
    def __init__(self, n_e: int, e_dim: int):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.embedding = nn.Embedding(n_e, e_dim)
        nn.init.uniform_(self.embedding.weight, -1.0 / n_e, 1.0 / n_e)

    def forward(self, z: Optional[torch.Tensor] = None,
                encoding_indices: Optional[torch.Tensor] = None) -> Tuple:
        """
        If encoding_indices is given, decode (lookup) only.
        Otherwise encode z -> (z_q, indices).
        """
        if encoding_indices is not None:
            return self.embedding(encoding_indices)

        # z: (B, e_dim, H, W)  -> flatten to (N, e_dim)
        B, C, H, W = z.shape
        z_flat = z.permute(0, 2, 3, 1).reshape(-1, C)  # (BHW, C)

        # Squared L2 distances
        d = (
            z_flat.pow(2).sum(1, keepdim=True)
            + self.embedding.weight.pow(2).sum(1)
            - 2 * z_flat @ self.embedding.weight.t()
        )  # (BHW, n_e)

        min_indices = d.argmin(dim=1)                         # (BHW,)
        z_q = self.embedding(min_indices)                     # (BHW, C)
        z_q = z_q.reshape(B, H, W, C).permute(0, 3, 1, 2)   # (B, C, H, W)

        # Straight-through estimator
        z_q = z + (z_q - z).detach()

        indices = min_indices.reshape(B, H, W)
        return z_q, indices


# ---------------------------------------------------------------------------
# Full VQGAN model
# ---------------------------------------------------------------------------

class VQGANModel(nn.Module):
    def __init__(self, config: VQGANConfig):
        super().__init__()
        self.config = config
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)
        self.quantize = VectorQuantizer(config.num_embeddings, config.quantized_embed_dim)
        self.quant_conv = nn.Conv2d(config.z_channels, config.quantized_embed_dim, 1)
        self.post_quant_conv = nn.Conv2d(config.quantized_embed_dim, config.z_channels, 1)

    # ------------------------------------------------------------------
    # encode / decode helpers that handle optional video (T) dimension
    # ------------------------------------------------------------------

    def encode(self, pixel_values: torch.Tensor):
        """
        pixel_values: (B, C, H, W) image  OR  (B, T, C, H, W) video
        Returns: (quantized_states, codebook_indices)
        """
        T = None
        if pixel_values.dim() == 5:                          # video
            T = pixel_values.shape[1]
            pixel_values = pixel_values.flatten(0, 1)        # (B*T, C, H, W)

        h = self.encoder(pixel_values)
        h = self.quant_conv(h)
        z_q, indices = self.quantize(h)

        if T is not None:
            z_q = z_q.unflatten(0, (-1, T))
            indices = indices.unflatten(0, (-1, T))
        return z_q, indices

    def decode(self, encoding: torch.Tensor, is_codebook_indices: bool = True):
        """
        encoding: codebook indices (B, H, W) or (B, T, H, W)
                  OR quantized latents (B, C, H, W) / (B, T, C, H, W)
        """
        if is_codebook_indices:
            encoding = self.quantize(encoding_indices=encoding)
            # quantize lookup returns (*, e_dim); fix channel dim
            # shape after lookup: same spatial dims, last dim = e_dim
            # need to permute to (B, C, H, W)
            if encoding.dim() == 4:          # (B, H, W, C)
                encoding = encoding.permute(0, 3, 1, 2)
            elif encoding.dim() == 5:        # (B, T, H, W, C)
                encoding = encoding.permute(0, 1, 4, 2, 3)

        T = None
        if encoding.dim() == 5:
            T = encoding.shape[1]
            encoding = encoding.flatten(0, 1)

        h = self.post_quant_conv(encoding)
        out = self.decoder(h)

        if T is not None:
            out = out.unflatten(0, (-1, T))
        return out.clamp(-1, 1)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        _, indices = self.encode(pixel_values)
        return self.decode(indices)


# ---------------------------------------------------------------------------
# High-level wrapper (mirrors the original VQGAN class)
# ---------------------------------------------------------------------------

class VQGAN:
    """
    Thin wrapper around VQGANModel that handles checkpoint loading
    and optional multi-GPU via DataParallel.
    """

    def __init__(self, vqgan_checkpoint: str, device: Optional[str] = None,
                 use_data_parallel: bool = False):
        assert vqgan_checkpoint != ''
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        config = VQGANConfig()
        self.model = VQGANModel(config)
        state_dict = torch.load(vqgan_checkpoint, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        if use_data_parallel and torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)
        self.model.eval()

    @torch.no_grad()
    def encode(self, pixel_values: torch.Tensor):
        pixel_values = pixel_values.to(self.device)
        return self.model.encode(pixel_values)

    @torch.no_grad()
    def decode(self, encoding: torch.Tensor, is_codebook_indices: bool = True):
        encoding = encoding.to(self.device)
        return self.model.decode(encoding, is_codebook_indices=is_codebook_indices)
