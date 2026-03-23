"""
AR Video Patch Transformer with KV-cache  (Emu3-style)
========================================================
Changes vs original
--------------------
  - gradient checkpointing in CausalTransformer (training only)
  - use_checkpoint flag; auto-disabled when kv_caches present or model.eval()
  - generate() is completely unaffected
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


@dataclass
class ARPatchConfig:
    resolution: int = 64
    num_channels: int = 3
    patch_size: int = 8
    d_model: int = 256
    n_heads: int = 4
    n_layers: int = 4
    mlp_ratio: float = 4.0
    dropout: float = 0.1
    frames_in: int = 2
    frames_out: int = 3

    @property
    def n_patches(self) -> int:
        return (self.resolution // self.patch_size) ** 2

    @property
    def patch_dim(self) -> int:
        return self.num_channels * self.patch_size * self.patch_size

    @property
    def max_seq_len(self) -> int:
        return (self.frames_in + self.frames_out) * self.n_patches


def patchify(frames: torch.Tensor, patch_size: int) -> torch.Tensor:
    B, T, C, H, W = frames.shape
    P = patch_size
    h, w = H // P, W // P
    x = frames.reshape(B * T, C, h, P, w, P)
    x = x.permute(0, 2, 4, 1, 3, 5)
    x = x.reshape(B, T * h * w, C * P * P)
    return x


def unpatchify(tokens: torch.Tensor, patch_size: int,
               resolution: int, num_channels: int) -> torch.Tensor:
    P = patch_size
    h = w = resolution // P
    N_p = h * w
    B, L, _ = tokens.shape
    T = L // N_p
    x = tokens.reshape(B * T, h, w, num_channels, P, P)
    x = x.permute(0, 3, 1, 4, 2, 5)
    x = x.reshape(B, T, num_channels, h * P, w * P)
    return x


class PatchPositionEmbedding(nn.Module):
    def __init__(self, cfg: ARPatchConfig):
        super().__init__()
        max_frames = cfg.frames_in + cfg.frames_out
        self.frame_emb = nn.Embedding(max_frames, cfg.d_model // 2)
        self.spatial_emb = nn.Embedding(cfg.n_patches, cfg.d_model // 2)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.cfg = cfg

    def forward(self, T: int, device: torch.device) -> torch.Tensor:
        N_p = self.cfg.n_patches
        frame_ids = torch.arange(T, device=device).repeat_interleave(N_p)
        spatial_ids = torch.arange(N_p, device=device).repeat(T)
        pos = torch.cat([self.frame_emb(frame_ids),
                         self.spatial_emb(spatial_ids)], dim=-1)
        return self.proj(pos)

    def forward_single(self, frame_id: int, spatial_id: int,
                       device: torch.device) -> torch.Tensor:
        fid = torch.tensor([frame_id], device=device)
        sid = torch.tensor([spatial_id], device=device)
        pos = torch.cat([self.frame_emb(fid), self.spatial_emb(sid)], dim=-1)
        return self.proj(pos)


class CausalSelfAttention(nn.Module):
    """
    Multi-head causal self-attention with FlashAttention via SDPA.

    Uses F.scaled_dot_product_attention which dispatches to FlashAttention-2
    on H100 / A100 with PyTorch >= 2.1.  No explicit O(B*H*S*S) attention
    tensor is ever materialised — this alone saves ~50 MB per block at
    B=32, S=320, on top of what gradient checkpointing saves.

    Two modes:
      Prefill (kv_cache=None, L>1) : is_causal=True  → FA-2 fused kernel
      Decode  (kv_cache given, L=1): is_causal=False → single-query kernel
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads  = n_heads
        self.head_dim = d_model // n_heads
        self.d_model  = d_model
        self.dropout  = dropout  # forwarded to SDPA during training only

        self.qkv      = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor,
                kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        B, L, D = x.shape
        qkv = self.qkv(x).reshape(B, L, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)  # (B, n_heads, L, head_dim)

        if kv_cache is not None:
            k = torch.cat([kv_cache[0], k], dim=2)
            v = torch.cat([kv_cache[1], v], dim=2)

        new_cache = (k, v)

        # Prefill with full sequence → causal mask handled inside FA-2 kernel.
        # Decode (L==1) → single query always attends to all cached keys, no mask.
        is_causal    = (L > 1) and (kv_cache is None)
        attn_dropout = self.dropout if self.training else 0.0

        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask = None,
            dropout_p = attn_dropout,
            is_causal = is_causal,
        )  # (B, n_heads, L, head_dim) — no S×S tensor ever allocated

        out = out.transpose(1, 2).reshape(B, L, D)
        return self.out_proj(out), new_cache


class MLP(nn.Module):
    def __init__(self, d_model: int, mlp_ratio: float, dropout: float = 0.0):
        super().__init__()
        inner = int(d_model * mlp_ratio)
        self.fc1 = nn.Linear(d_model, inner)
        self.fc2 = nn.Linear(inner, d_model)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(self.act(self.fc1(x))))


class TransformerBlock(nn.Module):
    def __init__(self, cfg: ARPatchConfig):
        super().__init__()
        self.norm1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg.d_model, cfg.n_heads, cfg.dropout)
        self.norm2 = nn.LayerNorm(cfg.d_model)
        self.mlp = MLP(cfg.d_model, cfg.mlp_ratio, cfg.dropout)

    def forward(self, x: torch.Tensor,
                kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        h, new_cache = self.attn(self.norm1(x), kv_cache=kv_cache)
        x = x + h
        x = x + self.mlp(self.norm2(x))
        return x, new_cache


class CausalTransformer(nn.Module):
    """
    Stack of transformer blocks.

    Gradient checkpointing
    ----------------------
    Active when: use_checkpoint=True AND self.training AND kv_caches is None.
    Recomputes each block's activations during backward instead of storing them.
    The O(B * S^2) attention maps are never kept in memory during training.
    """

    def __init__(self, cfg: ARPatchConfig):
        super().__init__()
        self.blocks = nn.ModuleList(
            [TransformerBlock(cfg) for _ in range(cfg.n_layers)]
        )

    def forward(
        self,
        x: torch.Tensor,
        kv_caches: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_checkpoint: bool = True,
    ) -> Tuple[torch.Tensor, List]:
        new_caches: List = []
        do_ckpt = use_checkpoint and self.training and (kv_caches is None)

        for i, block in enumerate(self.blocks):
            cache = kv_caches[i] if kv_caches is not None else None

            if do_ckpt:
                # Closure captures block; checkpoint only accepts Tensor args
                def _fwd(x_, b=block):
                    out, _ = b(x_, kv_cache=None)
                    return out
                x = checkpoint(_fwd, x, use_reentrant=False)
                new_caches.append(None)
            else:
                x, new_cache = block(x, kv_cache=cache)
                new_caches.append(new_cache)

        return x, new_caches


class ARVideoPatchTransformer(nn.Module):
    """
    Emu3-style autoregressive video predictor with KV-cache generation.

    Training  — teacher-forcing + gradient checkpointing:
        model.train()
        pred, loss = model(context, target)
        # activations are recomputed during backward, not stored

    Inference — token-level AR with KV-cache:
        model.eval()
        generated = model.generate(context, n_steps)
    """

    def __init__(self, cfg: ARPatchConfig):
        super().__init__()
        self.cfg = cfg
        self.patch_embed = nn.Linear(cfg.patch_dim, cfg.d_model)
        self.pos_emb = PatchPositionEmbedding(cfg)
        self.norm_in = nn.LayerNorm(cfg.d_model)
        self.transformer = CausalTransformer(cfg)
        self.head = nn.Linear(cfg.d_model, cfg.patch_dim)
        self.norm_out = nn.LayerNorm(cfg.d_model)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.trunc_normal_(m.weight, std=0.02)

    def _tokens_from_frames(self, frames: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg
        patches = patchify(frames, cfg.patch_size)
        tokens = self.patch_embed(patches)
        T = frames.shape[1]
        tokens = tokens + self.pos_emb(T, frames.device)
        return self.norm_in(tokens)

    def _embed_patch(self, patch: torch.Tensor,
                     frame_id: int, spatial_id: int,
                     device: torch.device) -> torch.Tensor:
        token = self.patch_embed(patch)
        pos = self.pos_emb.forward_single(frame_id, spatial_id, device)
        return self.norm_in(token + pos)

    def _decode_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.head(self.norm_out(tokens))

    def forward(
        self,
        input_frames: torch.Tensor,
        target_frames: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        cfg = self.cfg

        if target_frames is not None:
            all_frames = torch.cat([input_frames, target_frames], dim=1)
            tokens_in = self._tokens_from_frames(all_frames)
            tokens_out, _ = self.transformer(tokens_in, use_checkpoint=True)

            n_ctx = cfg.frames_in * cfg.n_patches
            n_tgt = cfg.frames_out * cfg.n_patches
            pred_tokens = tokens_out[:, n_ctx - 1: n_ctx + n_tgt - 1, :]
            pred_patches = self._decode_tokens(pred_tokens)
            tgt_patches = patchify(target_frames, cfg.patch_size)

            loss = F.mse_loss(pred_patches, tgt_patches)
            pred_frames = unpatchify(
                pred_patches.detach(), cfg.patch_size,
                cfg.resolution, cfg.num_channels,
            ).clamp(-1, 1)
        else:
            tokens_in = self._tokens_from_frames(input_frames)
            tokens_out, _ = self.transformer(tokens_in, use_checkpoint=False)
            last_tokens = tokens_out[:, -1:, :]
            pred_patches = self._decode_tokens(last_tokens)
            pred_frames = unpatchify(
                pred_patches, cfg.patch_size,
                cfg.resolution, cfg.num_channels,
            ).clamp(-1, 1)
            loss = None

        return pred_frames, loss

    @torch.no_grad()
    def generate(
        self,
        context_frames: torch.Tensor,
        n_steps: int = 1,
    ) -> torch.Tensor:
        cfg = self.cfg
        N_p = cfg.n_patches
        device = context_frames.device

        window = context_frames[:, -cfg.frames_in:].clone()
        all_generated = []

        for _ in range(n_steps):
            tokens_in = self._tokens_from_frames(window)
            hidden, kv_caches = self.transformer(tokens_in, use_checkpoint=False)

            pred_patch = self._decode_tokens(hidden[:, -1:, :])
            gen_patches = [pred_patch]

            n_gen = cfg.frames_out * N_p
            for i in range(1, n_gen):
                frame_id = cfg.frames_in + (i - 1) // N_p
                spatial_id = (i - 1) % N_p
                token = self._embed_patch(pred_patch, frame_id, spatial_id, device)
                hidden, kv_caches = self.transformer(
                    token, kv_caches, use_checkpoint=False
                )
                pred_patch = self._decode_tokens(hidden)
                gen_patches.append(pred_patch)

            all_patches = torch.cat(gen_patches, dim=1)
            gen_frames = unpatchify(
                all_patches, cfg.patch_size,
                cfg.resolution, cfg.num_channels,
            ).clamp(-1, 1)
            all_generated.append(gen_frames)
            window = torch.cat([window, gen_frames], dim=1)[:, -cfg.frames_in:]

        return torch.cat(all_generated, dim=1)


if __name__ == "__main__":
    cfg = ARPatchConfig(resolution=64, num_channels=3, patch_size=8,
                        d_model=256, n_heads=4, n_layers=4,
                        frames_in=2, frames_out=3)
    model = ARVideoPatchTransformer(cfg)
    B = 2
    ctx = torch.randn(B, cfg.frames_in, 3, 64, 64)
    tgt = torch.randn(B, cfg.frames_out, 3, 64, 64)
    model.train()
    pred, loss = model(ctx, tgt)
    loss.backward()
    print(f"train loss={loss.item():.4f}  pred={pred.shape}")
    model.eval()
    gen = model.generate(ctx, n_steps=2)
    print(f"generated={gen.shape}")
    print("model.py OK")