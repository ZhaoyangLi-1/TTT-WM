"""
AR Video Patch Transformer  (Emu3-style, pixel-only, small)
=============================================================
Inspired by Emu3 (arxiv 2409.18869): next-token prediction on a flat
sequence of tokens.  Here tokens are IMAGE PATCHES (continuous), not
discrete codebook indices — no encoder / decoder conv stack needed.

Pipeline
--------
  input  : (B, n, C, H, W)  — n context frames, pixel values in [-1, 1]
  patchify → (B, n*N_p, patch_dim)   N_p = (H/P)*(W/P), patch_dim = C*P*P
  + frame & spatial position embeddings
  causal Transformer
  linear head → patch_dim per token
  unpatchify → (B, m, C, H, W)  m predicted frames

AR generation
-------------
  window  = last n frames
  for step in range(n_steps):
      pred_tokens = transformer(patchify(window))[-m*N_p:]
      pred_frames = unpatchify(pred_tokens)
      window      = cat([window, pred_frames])[-n:]

Everything lives in a single small Transformer.  No convolution stack.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class ARPatchConfig:
    # image
    resolution: int = 64        # H = W  (square for simplicity)
    num_channels: int = 3
    patch_size: int = 8         # P  — each patch is P×P pixels

    # transformer
    d_model: int = 256          # embedding dim
    n_heads: int = 4
    n_layers: int = 4
    mlp_ratio: float = 4.0
    dropout: float = 0.1

    # AR
    frames_in: int = 2          # n  context frames per step
    frames_out: int = 3         # m  frames to predict per step

    @property
    def n_patches(self) -> int:
        """Number of patches per frame."""
        return (self.resolution // self.patch_size) ** 2

    @property
    def patch_dim(self) -> int:
        """Flattened pixel dim of one patch."""
        return self.num_channels * self.patch_size * self.patch_size

    @property
    def max_seq_len(self) -> int:
        """Max tokens in one forward pass = (n + m) * patches_per_frame."""
        return (self.frames_in + self.frames_out) * self.n_patches


# ---------------------------------------------------------------------------
# Patch utilities
# ---------------------------------------------------------------------------

def patchify(frames: torch.Tensor, patch_size: int) -> torch.Tensor:
    """
    frames : (B, T, C, H, W)
    returns: (B, T * N_p, C * P * P)   N_p = (H/P)*(W/P)
    """
    B, T, C, H, W = frames.shape
    P = patch_size
    assert H % P == 0 and W % P == 0, "resolution must be divisible by patch_size"
    h, w = H // P, W // P                                # grid dims
    x = frames.reshape(B * T, C, h, P, w, P)
    x = x.permute(0, 2, 4, 1, 3, 5)                     # (B*T, h, w, C, P, P)
    x = x.reshape(B, T * h * w, C * P * P)              # (B, T*N_p, patch_dim)
    return x


def unpatchify(tokens: torch.Tensor, patch_size: int,
               resolution: int, num_channels: int) -> torch.Tensor:
    """
    tokens : (B, T * N_p, C * P * P)
    returns: (B, T, C, H, W)
    """
    P = patch_size
    h = w = resolution // P
    N_p = h * w
    B, L, patch_dim = tokens.shape
    T = L // N_p
    x = tokens.reshape(B * T, h, w, num_channels, P, P)
    x = x.permute(0, 3, 1, 4, 2, 5)                     # (B*T, C, h, P, w, P)
    x = x.reshape(B, T, num_channels, h * P, w * P)     # (B, T, C, H, W)
    return x


# ---------------------------------------------------------------------------
# Position embeddings
# ---------------------------------------------------------------------------

class PatchPositionEmbedding(nn.Module):
    """
    Learned 2-D factored position embedding:
        frame_pos  (which frame, 0..T-1)   → d_model/2
        spatial_pos (which patch in frame) → d_model/2
    concatenated → d_model, then projected.

    This mirrors Emu3's use of separate spatial and temporal position info.
    """

    def __init__(self, cfg: ARPatchConfig):
        super().__init__()
        max_frames = cfg.frames_in + cfg.frames_out
        self.frame_emb   = nn.Embedding(max_frames, cfg.d_model // 2)
        self.spatial_emb = nn.Embedding(cfg.n_patches, cfg.d_model // 2)
        self.proj        = nn.Linear(cfg.d_model, cfg.d_model)
        self.cfg = cfg

    def forward(self, T: int, device: torch.device) -> torch.Tensor:
        """
        Returns position bias (T * N_p, d_model) to add to token embeddings.
        """
        N_p = self.cfg.n_patches
        frame_ids   = torch.arange(T, device=device).repeat_interleave(N_p)   # (T*N_p,)
        spatial_ids = torch.arange(N_p, device=device).repeat(T)              # (T*N_p,)
        pos = torch.cat([self.frame_emb(frame_ids),
                         self.spatial_emb(spatial_ids)], dim=-1)              # (T*N_p, d_model)
        return self.proj(pos)


# ---------------------------------------------------------------------------
# Causal Transformer
# ---------------------------------------------------------------------------

class CausalTransformer(nn.Module):
    """
    Standard Pre-LN Transformer with causal (autoregressive) mask.
    Input/output shape: (B, L, d_model)
    """

    def __init__(self, cfg: ARPatchConfig):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=int(cfg.d_model * cfg.mlp_ratio),
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,     # Pre-LN
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=cfg.n_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, d_model)  →  (B, L, d_model)"""
        L = x.shape[1]
        mask = nn.Transformer.generate_square_subsequent_mask(L, device=x.device)
        return self.transformer(x, mask=mask, is_causal=True)


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class ARVideoPatchTransformer(nn.Module):
    """
    Emu3-style autoregressive video predictor operating on pixel patches.

    Training (teacher-forcing)
    --------------------------
    input_frames  : (B, frames_in,  C, H, W)   context
    target_frames : (B, frames_out, C, H, W)   next frames to predict

    We feed [context | target] (all frames) into the causal Transformer
    and supervise only the target positions (last frames_out * N_p tokens).
    This is exactly next-token prediction on the patch sequence.

    Inference (AR)
    --------------
    generate(context_frames, n_steps) — see below.
    """

    def __init__(self, cfg: ARPatchConfig):
        super().__init__()
        self.cfg = cfg

        # patch → token
        self.patch_embed = nn.Linear(cfg.patch_dim, cfg.d_model)
        self.pos_emb     = PatchPositionEmbedding(cfg)
        self.norm_in     = nn.LayerNorm(cfg.d_model)

        # transformer
        self.transformer = CausalTransformer(cfg)

        # token → patch pixels
        self.head     = nn.Linear(cfg.d_model, cfg.patch_dim)
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
    # core
    # ------------------------------------------------------------------

    def _tokens_from_frames(self, frames: torch.Tensor) -> torch.Tensor:
        """
        frames : (B, T, C, H, W)
        returns: (B, T*N_p, d_model)   embedded + position-encoded tokens
        """
        cfg = self.cfg
        B, T = frames.shape[:2]
        patches = patchify(frames, cfg.patch_size)           # (B, T*N_p, patch_dim)
        tokens  = self.patch_embed(patches)                  # (B, T*N_p, d_model)
        tokens  = tokens + self.pos_emb(T, frames.device)   # broadcast over B
        return self.norm_in(tokens)

    def _decode_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        tokens : (B, L, d_model)
        returns: (B, L, patch_dim)   raw patch pixel predictions
        """
        return self.head(self.norm_out(tokens))

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------

    def forward(self,
                input_frames:  torch.Tensor,
                target_frames: Optional[torch.Tensor] = None,
                ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Parameters
        ----------
        input_frames  : (B, frames_in,  C, H, W)
        target_frames : (B, frames_out, C, H, W)  optional; if given, compute loss

        Returns
        -------
        pred_frames : (B, frames_out, C, H, W)   predicted next frames
        loss        : MSE scalar or None
        """
        cfg = self.cfg
        B = input_frames.shape[0]

        if target_frames is not None:
            # teacher-forcing: feed all frames, supervise only target positions
            all_frames = torch.cat([input_frames, target_frames], dim=1)  # (B, n+m, C,H,W)
            tokens_in  = self._tokens_from_frames(all_frames)             # (B, (n+m)*N_p, d)
            tokens_out = self.transformer(tokens_in)                      # (B, (n+m)*N_p, d)

            # prediction at position t predicts patch t+1  (shift by 1 within sequence)
            # we supervise the last frames_out*N_p OUTPUT positions,
            # which correspond to predicting the target patches.
            n_ctx    = cfg.frames_in  * cfg.n_patches   # input token count
            n_tgt    = cfg.frames_out * cfg.n_patches   # target token count

            # output tokens [n_ctx-1 .. n_ctx+n_tgt-2] predict target patches [0..n_tgt-1]
            pred_tokens  = tokens_out[:, n_ctx - 1 : n_ctx + n_tgt - 1, :]  # (B, n_tgt, d)
            pred_patches = self._decode_tokens(pred_tokens)                  # (B, n_tgt, patch_dim)

            # ground-truth patches
            tgt_patches  = patchify(target_frames, cfg.patch_size)           # (B, n_tgt, patch_dim)

            loss        = F.mse_loss(pred_patches, tgt_patches)
            pred_frames = unpatchify(pred_patches.detach(), cfg.patch_size,
                                     cfg.resolution, cfg.num_channels).clamp(-1, 1)
        else:
            # inference: feed only input, take last n_patches outputs as pred
            tokens_in   = self._tokens_from_frames(input_frames)            # (B, n*N_p, d)
            tokens_out  = self.transformer(tokens_in)                       # (B, n*N_p, d)

            # last N_p output tokens predict first frame after context
            # (for multi-frame prediction we do iterative generation — see generate())
            last_tokens  = tokens_out[:, -cfg.n_patches:, :]                # (B, N_p, d)
            pred_patches = self._decode_tokens(last_tokens)                  # (B, N_p, patch_dim)
            pred_frames  = unpatchify(pred_patches, cfg.patch_size,
                                      cfg.resolution, cfg.num_channels).clamp(-1, 1)
            pred_frames  = pred_frames                                        # (B, 1, C, H, W)
            loss = None

        return pred_frames, loss

    # ------------------------------------------------------------------
    # AR generation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate(self,
                 context_frames: torch.Tensor,
                 n_steps: int = 1) -> torch.Tensor:
        """
        Autoregressively generate future frames.

        Each step:
          1. patchify window (n frames)
          2. run causal Transformer on all tokens
          3. take last N_p output tokens → first predicted frame
          4. append that frame to window, repeat for frames_out frames
          5. slide window: keep last frames_in frames

        Parameters
        ----------
        context_frames : (B, >= frames_in, C, H, W)
        n_steps        : number of AR steps; each produces frames_out frames

        Returns
        -------
        (B, n_steps * frames_out, C, H, W)
        """
        cfg    = self.cfg
        device = context_frames.device
        window = context_frames[:, -cfg.frames_in:].clone()   # (B, n, C, H, W)
        all_generated = []

        for _ in range(n_steps):
            step_frames = []

            # generate frames_out frames one at a time (strictly AR at frame level)
            cur_window = window
            for _ in range(cfg.frames_out):
                tokens_in  = self._tokens_from_frames(cur_window)
                tokens_out = self.transformer(tokens_in)
                last_tok   = tokens_out[:, -cfg.n_patches:, :]       # (B, N_p, d)
                patches    = self._decode_tokens(last_tok)            # (B, N_p, patch_dim)
                frame      = unpatchify(patches, cfg.patch_size,
                                        cfg.resolution,
                                        cfg.num_channels).clamp(-1, 1)  # (B,1,C,H,W)
                step_frames.append(frame)
                # append generated frame to window, drop oldest
                cur_window = torch.cat([cur_window, frame], dim=1)[:, -cfg.frames_in:]

            step_pred = torch.cat(step_frames, dim=1)   # (B, frames_out, C, H, W)
            all_generated.append(step_pred)

            # slide outer window
            window = torch.cat([window, step_pred], dim=1)[:, -cfg.frames_in:]

        return torch.cat(all_generated, dim=1)           # (B, n_steps*frames_out, C, H, W)


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cfg = ARPatchConfig(
        resolution=64,
        num_channels=3,
        patch_size=8,       # → 8×8 = 64 patches per frame
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
    ctx    = torch.randn(B, cfg.frames_in,  3, cfg.resolution, cfg.resolution)
    target = torch.randn(B, cfg.frames_out, 3, cfg.resolution, cfg.resolution)

    # teacher-forcing
    pred, loss = model(ctx, target)
    print(f"pred shape    : {pred.shape}")    # (2, 3, 3, 64, 64)
    print(f"train loss    : {loss.item():.4f}")

    # AR generation: 4 steps → 12 frames
    gen = model.generate(ctx, n_steps=4)
    print(f"generated     : {gen.shape}")     # (2, 12, 3, 64, 64)