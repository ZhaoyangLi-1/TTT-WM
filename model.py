"""
AR Video Patch Transformer with KV-cache  (Emu3-style)
========================================================
Fully causal next-token prediction on continuous image patches.

Training:  teacher-forcing with causal mask (standard GPT-style).
Inference: token-by-token AR generation with KV-cache.

Pipeline
--------
  input  : (B, n, C, H, W)  — n context frames, pixel values in [-1, 1]
  patchify → (B, n*N_p, patch_dim)   N_p = (H/P)*(W/P), patch_dim = C*P*P
  + frame & spatial position embeddings
  causal Transformer
  linear head → patch_dim per token
  unpatchify → (B, m, C, H, W)  m predicted frames

AR generation with KV-cache
-----------------------------
  1. Prefill: forward all context tokens, cache K/V per layer
  2. Decode:  generate one patch at a time, update cache incrementally
  3. Each decode step is O(L·d) instead of O(L²·d)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class ARPatchConfig:
    # image
    resolution: int = 64
    num_channels: int = 3
    patch_size: int = 8

    # transformer
    d_model: int = 256
    n_heads: int = 4
    n_layers: int = 4
    mlp_ratio: float = 4.0
    dropout: float = 0.1

    # AR
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


# ---------------------------------------------------------------------------
# Patch utilities
# ---------------------------------------------------------------------------

def patchify(frames: torch.Tensor, patch_size: int) -> torch.Tensor:
    """(B, T, C, H, W) → (B, T*N_p, C*P*P)"""
    B, T, C, H, W = frames.shape
    P = patch_size
    h, w = H // P, W // P
    x = frames.reshape(B * T, C, h, P, w, P)
    x = x.permute(0, 2, 4, 1, 3, 5)
    x = x.reshape(B, T * h * w, C * P * P)
    return x


def unpatchify(tokens: torch.Tensor, patch_size: int,
               resolution: int, num_channels: int) -> torch.Tensor:
    """(B, T*N_p, C*P*P) → (B, T, C, H, W)"""
    P = patch_size
    h = w = resolution // P
    N_p = h * w
    B, L, _ = tokens.shape
    T = L // N_p
    x = tokens.reshape(B * T, h, w, num_channels, P, P)
    x = x.permute(0, 3, 1, 4, 2, 5)
    x = x.reshape(B, T, num_channels, h * P, w * P)
    return x


# ---------------------------------------------------------------------------
# Position embeddings
# ---------------------------------------------------------------------------

class PatchPositionEmbedding(nn.Module):
    """
    Learned 2-D factored position embedding:
        frame_pos  (0..T-1)   → d_model/2
        spatial_pos (0..N_p-1) → d_model/2
    concatenated → d_model, then projected.
    """

    def __init__(self, cfg: ARPatchConfig):
        super().__init__()
        max_frames = cfg.frames_in + cfg.frames_out
        self.frame_emb = nn.Embedding(max_frames, cfg.d_model // 2)
        self.spatial_emb = nn.Embedding(cfg.n_patches, cfg.d_model // 2)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.cfg = cfg

    def forward(self, T: int, device: torch.device) -> torch.Tensor:
        """Position embeddings for T frames. Returns (T*N_p, d_model)."""
        N_p = self.cfg.n_patches
        frame_ids = torch.arange(T, device=device).repeat_interleave(N_p)
        spatial_ids = torch.arange(N_p, device=device).repeat(T)
        pos = torch.cat([self.frame_emb(frame_ids),
                         self.spatial_emb(spatial_ids)], dim=-1)
        return self.proj(pos)

    def forward_single(self, frame_id: int, spatial_id: int,
                        device: torch.device) -> torch.Tensor:
        """Position embedding for one token. Returns (1, d_model)."""
        fid = torch.tensor([frame_id], device=device)
        sid = torch.tensor([spatial_id], device=device)
        pos = torch.cat([self.frame_emb(fid), self.spatial_emb(sid)], dim=-1)
        return self.proj(pos)


# ---------------------------------------------------------------------------
# Causal Self-Attention with KV-cache
# ---------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    """
    Multi-head causal self-attention.

    Supports two modes:
      - Prefill  (kv_cache=None):  process full sequence with causal mask
      - Decode   (kv_cache given): process 1 token, append to cache, no mask
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5
        self.d_model = d_model

        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        x        : (B, L, D)
        kv_cache : None or (k_prev, v_prev) each (B, n_heads, S, head_dim)
        Returns  : output (B, L, D), new_cache (k, v)
        """
        B, L, D = x.shape
        qkv = self.qkv(x).reshape(B, L, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)  # each (B, n_heads, L, head_dim)

        if kv_cache is not None:
            k = torch.cat([kv_cache[0], k], dim=2)
            v = torch.cat([kv_cache[1], v], dim=2)

        new_cache = (k, v)
        S = k.shape[2]  # total key length

        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, n_heads, L, S)

        # Causal mask only needed when L > 1 (prefill)
        if L > 1:
            row = torch.arange(L, device=x.device).unsqueeze(1)
            col = torch.arange(S, device=x.device).unsqueeze(0)
            mask = col > (row + S - L)  # (L, S)
            attn = attn.masked_fill(mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, L, D)
        return self.out_proj(out), new_cache


# ---------------------------------------------------------------------------
# Feed-forward
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Transformer block & stack
# ---------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    """Pre-LN transformer block with optional KV-cache."""

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
    """Stack of transformer blocks with per-layer KV-cache."""

    def __init__(self, cfg: ARPatchConfig):
        super().__init__()
        self.blocks = nn.ModuleList(
            [TransformerBlock(cfg) for _ in range(cfg.n_layers)]
        )

    def forward(
        self, x: torch.Tensor,
        kv_caches: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
    ) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
        new_caches = []
        for i, block in enumerate(self.blocks):
            cache = kv_caches[i] if kv_caches is not None else None
            x, new_cache = block(x, kv_cache=cache)
            new_caches.append(new_cache)
        return x, new_caches


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class ARVideoPatchTransformer(nn.Module):
    """
    Emu3-style autoregressive video predictor with KV-cache generation.

    Training  — teacher-forcing, standard next-token prediction:
        pred, loss = model(context, target)

    Inference — token-level AR with KV-cache:
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

    # ------------------------------------------------------------------
    # Embedding helpers
    # ------------------------------------------------------------------

    def _tokens_from_frames(self, frames: torch.Tensor) -> torch.Tensor:
        """(B, T, C, H, W) → (B, T*N_p, d_model)"""
        cfg = self.cfg
        patches = patchify(frames, cfg.patch_size)
        tokens = self.patch_embed(patches)
        T = frames.shape[1]
        tokens = tokens + self.pos_emb(T, frames.device)
        return self.norm_in(tokens)

    def _embed_patch(self, patch: torch.Tensor,
                     frame_id: int, spatial_id: int,
                     device: torch.device) -> torch.Tensor:
        """Embed a single predicted patch for incremental decode.
        patch: (B, 1, patch_dim) → (B, 1, d_model)
        """
        token = self.patch_embed(patch)
        pos = self.pos_emb.forward_single(frame_id, spatial_id, device)
        return self.norm_in(token + pos)  # (1, d_model) broadcasts over (B, 1, d_model)

    def _decode_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """(B, L, d_model) → (B, L, patch_dim)"""
        return self.head(self.norm_out(tokens))

    # ------------------------------------------------------------------
    # Forward (training)
    # ------------------------------------------------------------------

    def forward(
        self,
        input_frames: torch.Tensor,
        target_frames: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        cfg = self.cfg

        if target_frames is not None:
            all_frames = torch.cat([input_frames, target_frames], dim=1)
            tokens_in = self._tokens_from_frames(all_frames)
            tokens_out, _ = self.transformer(tokens_in)

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
            tokens_out, _ = self.transformer(tokens_in)

            last_tokens = tokens_out[:, -1:, :]   # only last position is meaningful
            pred_patches = self._decode_tokens(last_tokens)
            pred_frames = unpatchify(
                pred_patches, cfg.patch_size,
                cfg.resolution, cfg.num_channels,
            ).clamp(-1, 1)
            loss = None

        return pred_frames, loss

    # ------------------------------------------------------------------
    # AR generation with KV-cache
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        context_frames: torch.Tensor,
        n_steps: int = 1,
    ) -> torch.Tensor:
        """
        Token-level autoregressive generation with KV-cache.

        Each AR step:
          1. Prefill context window → cache KV for all context tokens
          2. Decode frames_out × N_p patches one-by-one:
             - last output predicts next patch
             - embed predicted patch, feed through transformer, update cache
          3. Slide context window, clear cache, repeat

        Position embedding constraint: frame_id ∈ [0, frames_in + frames_out),
        so each step generates at most frames_out frames before re-prefilling.

        Parameters
        ----------
        context_frames : (B, >= frames_in, C, H, W)
        n_steps        : number of AR steps, each produces frames_out frames

        Returns
        -------
        (B, n_steps * frames_out, C, H, W)
        """
        cfg = self.cfg
        N_p = cfg.n_patches
        device = context_frames.device

        window = context_frames[:, -cfg.frames_in:].clone()
        all_generated = []

        for _ in range(n_steps):
            # ── Prefill ──────────────────────────────────────────────
            tokens_in = self._tokens_from_frames(window)
            hidden, kv_caches = self.transformer(tokens_in)

            # First predicted patch (from last context token)
            pred_patch = self._decode_tokens(hidden[:, -1:, :])  # (B, 1, patch_dim)
            gen_patches = [pred_patch]

            # ── Decode remaining patches ─────────────────────────────
            n_gen = cfg.frames_out * N_p
            for i in range(1, n_gen):
                # Position of the token being fed back
                frame_id = cfg.frames_in + (i - 1) // N_p
                spatial_id = (i - 1) % N_p

                token = self._embed_patch(pred_patch, frame_id, spatial_id, device)
                hidden, kv_caches = self.transformer(token, kv_caches)

                pred_patch = self._decode_tokens(hidden)  # (B, 1, patch_dim)
                gen_patches.append(pred_patch)

            # ── Assemble frames ──────────────────────────────────────
            all_patches = torch.cat(gen_patches, dim=1)
            gen_frames = unpatchify(
                all_patches, cfg.patch_size,
                cfg.resolution, cfg.num_channels,
            ).clamp(-1, 1)
            all_generated.append(gen_frames)

            # Slide window (clear cache implicitly by re-prefilling)
            window = torch.cat([window, gen_frames], dim=1)[:, -cfg.frames_in:]

        return torch.cat(all_generated, dim=1)


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cfg = ARPatchConfig(
        resolution=64,
        num_channels=3,
        patch_size=8,
        d_model=256,
        n_heads=4,
        n_layers=4,
        frames_in=2,
        frames_out=3,
    )

    model = ARVideoPatchTransformer(cfg)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Patches/frame : {cfg.n_patches}")
    print(f"Sequence len  : {(cfg.frames_in + cfg.frames_out) * cfg.n_patches}")
    print(f"Parameters    : {n_params:.2f} M")

    B = 2
    ctx = torch.randn(B, cfg.frames_in, 3, cfg.resolution, cfg.resolution)
    target = torch.randn(B, cfg.frames_out, 3, cfg.resolution, cfg.resolution)

    # Teacher-forcing
    pred, loss = model(ctx, target)
    print(f"pred shape    : {pred.shape}")
    print(f"train loss    : {loss.item():.4f}")

    # AR generation with KV-cache
    model.eval()
    gen = model.generate(ctx, n_steps=2)
    print(f"generated     : {gen.shape}")    # (2, 6, 3, 64, 64)
    print("KV-cache generation OK!")
