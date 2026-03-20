"""
Cosmos-style AR Video Patch Transformer
========================================
Inspired by NVIDIA Cosmos (arxiv 2501.03575) autoregressive WFM design.

Key architectural changes vs. the original Emu3-style model
------------------------------------------------------------
1.  Block-Causal Attention
        Tokens within the SAME frame attend to each other freely (bidirectional).
        Tokens in LATER frames cannot attend to tokens in EARLIER frames
        (causal across frames, not across individual patches).
        This mirrors Cosmos' "block-causal" attention design.

2.  RoPE (Rotary Position Embedding)
        Replaces learned factored position embeddings.
        Applied per-head in the attention layer.
        Temporal RoPE: frame index.
        Spatial RoPE:  patch (row, col) — 2-D factored.

3.  RMSNorm  (instead of LayerNorm)
        Lighter, used throughout Cosmos / LLaMA family.

4.  SwiGLU MLP  (instead of standard GELU MLP)
        gate(x) = swish(W1·x) ⊙ (W2·x) followed by W3
        mlp_ratio controls the inner (pre-gate) dim.

5.  QK-Norm
        Independent RMSNorm on Q and K before computing attention scores.
        Stabilises training at scale (used in Cosmos and many recent LLMs).

6.  Parallel Attention Block  (optional, default off)
        Attention and MLP share the same pre-norm and their outputs are summed.
        When enabled it can improve throughput (fewer sequential deps).

Interface (unchanged — train.py works without modification)
-----------------------------------------------------------
    model(input_frames, target_frames)  →  pred_frames, loss
    model.generate(context_frames, n_steps)  →  generated frames
"""

from __future__ import annotations
import math
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
    resolution:   int   = 64
    num_channels: int   = 3
    patch_size:   int   = 8        # P — each patch is P×P pixels

    # transformer
    d_model:          int   = 512
    n_heads:          int   = 8
    n_layers:         int   = 8
    mlp_ratio:        float = 8/3  # SwiGLU convention: ~2.67 keeps param count similar
    dropout:          float = 0.0  # Cosmos uses 0 dropout
    qk_norm:          bool  = True
    parallel_attn:    bool  = False  # parallel attn+mlp block (Cosmos innovation)

    # AR
    frames_in:  int = 4   # context frames per step
    frames_out: int = 4   # frames to predict per step

    @property
    def n_patches(self) -> int:
        return (self.resolution // self.patch_size) ** 2

    @property
    def patch_dim(self) -> int:
        return self.num_channels * self.patch_size * self.patch_size

    @property
    def max_seq_len(self) -> int:
        return (self.frames_in + self.frames_out) * self.n_patches

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads


# ---------------------------------------------------------------------------
# Patch utilities  (unchanged from original)
# ---------------------------------------------------------------------------

def patchify(frames: torch.Tensor, patch_size: int) -> torch.Tensor:
    """(B, T, C, H, W)  →  (B, T*N_p, C*P*P)"""
    B, T, C, H, W = frames.shape
    P = patch_size
    h, w = H // P, W // P
    x = frames.reshape(B * T, C, h, P, w, P)
    x = x.permute(0, 2, 4, 1, 3, 5)
    x = x.reshape(B, T * h * w, C * P * P)
    return x


def unpatchify(tokens: torch.Tensor, patch_size: int,
               resolution: int, num_channels: int) -> torch.Tensor:
    """(B, T*N_p, C*P*P)  →  (B, T, C, H, W)"""
    P  = patch_size
    h  = w = resolution // P
    N_p = h * w
    B, L, _ = tokens.shape
    T = L // N_p
    x = tokens.reshape(B * T, h, w, num_channels, P, P)
    x = x.permute(0, 3, 1, 4, 2, 5)
    x = x.reshape(B, T, num_channels, h * P, w * P)
    return x


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps   = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * rms).to(x.dtype) * self.scale


# ---------------------------------------------------------------------------
# Rotary Position Embedding  (RoPE)
# ---------------------------------------------------------------------------

def _build_freqs(head_dim: int, max_len: int, base: float = 10000.0,
                 device: torch.device = None) -> torch.Tensor:
    """Returns (max_len, head_dim/2) complex-valued frequencies."""
    half = head_dim // 2
    inv_freq = 1.0 / (base ** (torch.arange(0, half, device=device).float() / half))
    t   = torch.arange(max_len, device=device).float()
    freqs = torch.outer(t, inv_freq)           # (max_len, half)
    return torch.polar(torch.ones_like(freqs), freqs)   # complex


def apply_rope(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """
    x      : (B, n_heads, L, head_dim)   real
    freqs  : (L, head_dim//2)             complex
    returns: (B, n_heads, L, head_dim)   real, rotated
    """
    xc = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))  # (...,L,hd/2)
    rotated = xc * freqs.unsqueeze(0).unsqueeze(0)
    return torch.view_as_real(rotated).reshape(x.shape).to(x.dtype)


class RoPEEmbedding(nn.Module):
    """
    2-D factored RoPE: half the head dim for temporal (frame index),
    half for spatial (linearised patch index within a frame).

    For a sequence of T frames each with N_p patches:
        token (t, p) gets rope from (frame_index=t, patch_index=p)
    """

    def __init__(self, cfg: ARPatchConfig):
        super().__init__()
        self.cfg      = cfg
        self.head_dim = cfg.head_dim
        # We split head_dim equally: hd//2 for time, hd//2 for space
        # Each half is further split for the complex rotation: hd//4 per component
        self.register_buffer("_dummy", torch.zeros(0), persistent=False)

    def _freqs(self, T: int, N_p: int, device: torch.device):
        hd = self.head_dim
        # temporal RoPE  (hd//2 dims)
        t_freqs = _build_freqs(hd // 2, T, device=device)       # (T, hd//4)  complex
        # spatial  RoPE  (hd//2 dims)
        s_freqs = _build_freqs(hd // 2, N_p, device=device)     # (N_p, hd//4) complex

        # Expand to full sequence (T*N_p,)
        t_idx = torch.arange(T,   device=device).repeat_interleave(N_p)  # (L,)
        s_idx = torch.arange(N_p, device=device).repeat(T)               # (L,)

        rope = torch.cat([t_freqs[t_idx], s_freqs[s_idx]], dim=-1)       # (L, hd//2) complex
        return rope

    def forward(self, q: torch.Tensor, k: torch.Tensor,
                T: int, N_p: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        q, k : (B, n_heads, L, head_dim)
        """
        device = q.device
        freqs  = self._freqs(T, N_p, device)   # (L, hd//2)  complex
        return apply_rope(q, freqs), apply_rope(k, freqs)


# ---------------------------------------------------------------------------
# Block-Causal Attention Mask
# ---------------------------------------------------------------------------

def block_causal_mask(T: int, N_p: int, device: torch.device) -> torch.Tensor:
    """
    Returns additive attention mask (L, L) where L = T * N_p.

    Rule:
      - token at (t, p) can attend to token at (t', p') if t' <= t
        (patch position within the same frame is unrestricted — bidirectional
         within a frame, causal across frames).

    This is Cosmos' block-causal design:
        - Within a frame:  full bidirectional attention (intra-block)
        - Across frames:   causal (no future frames)

    Returns 0 for allowed positions, -inf for masked positions.
    """
    L = T * N_p
    # frame index of each token
    frame_of = torch.arange(T, device=device).repeat_interleave(N_p)  # (L,)
    # mask[i, j] = True  →  token i CANNOT see token j
    mask = frame_of.unsqueeze(0) > frame_of.unsqueeze(1)   # (L, L) bool
    return mask.float().masked_fill(mask, float("-inf")).masked_fill(~mask, 0.0)


# ---------------------------------------------------------------------------
# SwiGLU MLP
# ---------------------------------------------------------------------------

class SwiGLUMLP(nn.Module):
    """
    SwiGLU: out = (swish(gate) ⊙ value) · W_out
    inner_dim = round(d_model * mlp_ratio / 2) * 2  (keeps it even)
    """

    def __init__(self, d_model: int, mlp_ratio: float, dropout: float = 0.0):
        super().__init__()
        inner = int(d_model * mlp_ratio)
        inner = (inner // 2) * 2          # make even for the gating split
        self.gate_proj = nn.Linear(d_model, inner, bias=False)
        self.up_proj   = nn.Linear(d_model, inner, bias=False)
        self.down_proj = nn.Linear(inner // 2, d_model, bias=False)
        self.drop      = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate  = F.silu(self.gate_proj(x))           # (B, L, inner)
        value = self.up_proj(x)                      # (B, L, inner)
        # gated: split gate into two halves
        g1, g2 = gate.chunk(2, dim=-1)
        v1, v2 = value.chunk(2, dim=-1)
        hidden = g1 * v1 + g2 * v2                  # (B, L, inner//2)
        return self.drop(self.down_proj(hidden))


# ---------------------------------------------------------------------------
# Attention with QK-Norm
# ---------------------------------------------------------------------------

class BlockCausalAttention(nn.Module):
    """
    Multi-head self-attention with:
      - Block-causal mask (Cosmos style)
      - RoPE positional encoding
      - Optional QK-Norm (RMSNorm on Q and K)
    """

    def __init__(self, cfg: ARPatchConfig):
        super().__init__()
        self.n_heads  = cfg.n_heads
        self.head_dim = cfg.head_dim
        self.scale    = self.head_dim ** -0.5

        self.qkv  = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.out  = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.drop = nn.Dropout(cfg.dropout)

        self.rope = RoPEEmbedding(cfg)

        self.q_norm = RMSNorm(cfg.head_dim) if cfg.qk_norm else nn.Identity()
        self.k_norm = RMSNorm(cfg.head_dim) if cfg.qk_norm else nn.Identity()

        # cached mask – rebuilt only when (T, N_p, device) changes
        self._mask_cache: Optional[Tuple] = None
        self._mask: Optional[torch.Tensor] = None

    def _get_mask(self, T: int, N_p: int, device: torch.device) -> torch.Tensor:
        key = (T, N_p, str(device))
        if self._mask_cache != key:
            self._mask       = block_causal_mask(T, N_p, device)
            self._mask_cache = key
        return self._mask

    def forward(self, x: torch.Tensor, T: int, N_p: int) -> torch.Tensor:
        B, L, D = x.shape
        qkv = self.qkv(x).reshape(B, L, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)                  # each (B, L, n_heads, hd)
        q = q.transpose(1, 2)                         # (B, n_heads, L, hd)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # QK-Norm
        q = self.q_norm(q)
        k = self.k_norm(k)

        # RoPE
        q, k = self.rope(q, k, T, N_p)

        # Block-causal mask
        mask = self._get_mask(T, N_p, x.device)

        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale + mask
        attn = F.softmax(attn, dim=-1)
        attn = self.drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, L, D)
        return self.out(out)


# ---------------------------------------------------------------------------
# Transformer Block  (standard or parallel)
# ---------------------------------------------------------------------------

class CosmosBlock(nn.Module):
    """
    Pre-RMSNorm Transformer block.

    Standard mode  (parallel_attn=False):
        y = x + attn(norm1(x))
        y = y + mlp(norm2(y))

    Parallel mode  (parallel_attn=True, Cosmos optimisation):
        h = norm(x)
        y = x + attn(h) + mlp(h)
    """

    def __init__(self, cfg: ARPatchConfig):
        super().__init__()
        self.parallel = cfg.parallel_attn
        self.norm1    = RMSNorm(cfg.d_model)
        self.attn     = BlockCausalAttention(cfg)
        self.norm2    = RMSNorm(cfg.d_model) if not cfg.parallel_attn else None
        self.mlp      = SwiGLUMLP(cfg.d_model, cfg.mlp_ratio, cfg.dropout)

    def forward(self, x: torch.Tensor, T: int, N_p: int) -> torch.Tensor:
        if self.parallel:
            h = self.norm1(x)
            return x + self.attn(h, T, N_p) + self.mlp(h)
        else:
            x = x + self.attn(self.norm1(x), T, N_p)
            x = x + self.mlp(self.norm2(x))
            return x


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class ARVideoPatchTransformer(nn.Module):
    """
    Cosmos-style Autoregressive Video Patch Transformer.

    Architecture
    ────────────
    • Patchify frames → linear embed → Cosmos Transformer stack → linear head → unpatchify
    • Block-causal attention: bidirectional within each frame, causal across frames
    • RoPE for spatio-temporal position encoding
    • RMSNorm + SwiGLU + QK-Norm  (Cosmos/LLaMA family conventions)

    Training (teacher-forcing, same interface as original)
    ───────────────────────────────────────────────────────
        pred_frames, loss = model(input_frames, target_frames)

        We concatenate [context | target] frames, run one forward pass,
        and supervise only the target token positions (next-token prediction).

    Inference (AR generation)
    ──────────────────────────
        generated = model.generate(context_frames, n_steps)

        Each step slides a window of frames_in frames and predicts
        frames_out new frames one at a time.
    """

    def __init__(self, cfg: ARPatchConfig):
        super().__init__()
        self.cfg = cfg

        # Input projection: patch_dim → d_model
        self.patch_embed = nn.Linear(cfg.patch_dim, cfg.d_model, bias=False)
        self.embed_norm  = RMSNorm(cfg.d_model)

        # Transformer blocks
        self.blocks  = nn.ModuleList([CosmosBlock(cfg) for _ in range(cfg.n_layers)])
        self.out_norm = RMSNorm(cfg.d_model)

        # Output head: d_model → patch_dim
        self.head = nn.Linear(cfg.d_model, cfg.patch_dim, bias=False)

        self._init_weights()

    # ------------------------------------------------------------------
    # Weight initialisation  (Cosmos / GPT-NeoX style)
    # ------------------------------------------------------------------

    def _init_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (RMSNorm,)):
                nn.init.ones_(m.scale)

        # Scale residual projections by 1/sqrt(2*n_layers)  (GPT-2 trick)
        scale = (2 * self.cfg.n_layers) ** -0.5
        for name, p in self.named_parameters():
            if "out.weight" in name or "down_proj.weight" in name:
                p.data.mul_(scale)

    # ------------------------------------------------------------------
    # Core helpers
    # ------------------------------------------------------------------

    def _embed(self, frames: torch.Tensor) -> torch.Tensor:
        """
        frames : (B, T, C, H, W)
        returns: (B, T*N_p, d_model)
        """
        patches = patchify(frames, self.cfg.patch_size)   # (B, T*N_p, patch_dim)
        tokens  = self.patch_embed(patches)               # (B, T*N_p, d_model)
        return self.embed_norm(tokens)

    def _run_transformer(self, tokens: torch.Tensor, T: int) -> torch.Tensor:
        """
        tokens : (B, T*N_p, d_model)
        T      : number of frames (needed for block-causal mask + RoPE)
        returns: (B, T*N_p, d_model)
        """
        N_p = self.cfg.n_patches
        x   = tokens
        for block in self.blocks:
            x = block(x, T, N_p)
        return self.out_norm(x)

    def _decode(self, hidden: torch.Tensor) -> torch.Tensor:
        """(B, L, d_model)  →  (B, L, patch_dim)"""
        return self.head(hidden)

    # ------------------------------------------------------------------
    # Forward  (teacher-forcing training)
    # ------------------------------------------------------------------

    def forward(self,
                input_frames:  torch.Tensor,
                target_frames: Optional[torch.Tensor] = None,
                ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Parameters
        ----------
        input_frames  : (B, frames_in,  C, H, W)   context, values in [-1, 1]
        target_frames : (B, frames_out, C, H, W)   targets (optional)

        Returns
        -------
        pred_frames : (B, frames_out, C, H, W)
        loss        : MSE scalar  or  None
        """
        cfg = self.cfg

        if target_frames is not None:
            # ── Teacher-forcing ──────────────────────────────────────────
            # Feed [context | target] as a single sequence through the model.
            # Block-causal mask ensures each token can only attend to tokens
            # in the same or earlier frames (no cheating on future frames).
            #
            # Supervision: output at position (t-1) predicts input at position t.
            # We supervise the last frames_out*N_p output positions.
            all_frames = torch.cat([input_frames, target_frames], dim=1)  # (B, n+m, C,H,W)
            T_total    = all_frames.shape[1]
            N_p        = cfg.n_patches
            n_ctx      = cfg.frames_in  * N_p
            n_tgt      = cfg.frames_out * N_p

            tokens_in  = self._embed(all_frames)                          # (B, (n+m)*N_p, d)
            hidden     = self._run_transformer(tokens_in, T_total)        # (B, (n+m)*N_p, d)

            # Output at position (n_ctx - 1) … (n_ctx + n_tgt - 2)
            # predicts target patches at positions 0 … n_tgt - 1.
            pred_hidden  = hidden[:, n_ctx - 1 : n_ctx + n_tgt - 1, :]   # (B, n_tgt, d)
            pred_patches = self._decode(pred_hidden)                      # (B, n_tgt, patch_dim)

            tgt_patches  = patchify(target_frames, cfg.patch_size)        # (B, n_tgt, patch_dim)
            loss         = F.mse_loss(pred_patches, tgt_patches)

            pred_frames  = unpatchify(
                pred_patches.detach(), cfg.patch_size,
                cfg.resolution, cfg.num_channels
            ).clamp(-1, 1)

        else:
            # ── Inference: predict first frame after context ─────────────
            T_ctx      = input_frames.shape[1]
            tokens_in  = self._embed(input_frames)
            hidden     = self._run_transformer(tokens_in, T_ctx)

            last_hidden  = hidden[:, -cfg.n_patches:, :]      # (B, N_p, d)
            pred_patches = self._decode(last_hidden)           # (B, N_p, patch_dim)
            pred_frames  = unpatchify(
                pred_patches, cfg.patch_size,
                cfg.resolution, cfg.num_channels
            ).clamp(-1, 1)
            loss = None

        return pred_frames, loss

    # ------------------------------------------------------------------
    # AR Generation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate(self,
                 context_frames: torch.Tensor,
                 n_steps: int = 1) -> torch.Tensor:
        """
        Autoregressive generation.

        Each step generates frames_out frames by appending one predicted
        frame at a time (strictly AR at frame level, block-causal within).

        Parameters
        ----------
        context_frames : (B, >= frames_in, C, H, W)
        n_steps        : how many groups of frames_out to generate

        Returns
        -------
        (B, n_steps * frames_out, C, H, W)
        """
        cfg    = self.cfg
        window = context_frames[:, -cfg.frames_in:].clone()
        all_generated = []

        for _ in range(n_steps):
            step_frames = []
            cur_window  = window

            for _ in range(cfg.frames_out):
                T         = cur_window.shape[1]
                tokens_in = self._embed(cur_window)
                hidden    = self._run_transformer(tokens_in, T)

                last_hidden  = hidden[:, -cfg.n_patches:, :]
                pred_patches = self._decode(last_hidden)
                frame        = unpatchify(
                    pred_patches, cfg.patch_size,
                    cfg.resolution, cfg.num_channels
                ).clamp(-1, 1)                                  # (B, 1, C, H, W)

                step_frames.append(frame)
                cur_window = torch.cat([cur_window, frame], dim=1)[:, -cfg.frames_in:]

            step_pred = torch.cat(step_frames, dim=1)          # (B, frames_out, C, H, W)
            all_generated.append(step_pred)
            window = torch.cat([window, step_pred], dim=1)[:, -cfg.frames_in:]

        return torch.cat(all_generated, dim=1)


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cfg = ARPatchConfig(
        resolution=64,
        num_channels=3,
        patch_size=8,        # → 64 patches/frame
        d_model=512,
        n_heads=8,
        n_layers=8,
        frames_in=4,
        frames_out=4,
        qk_norm=True,
        parallel_attn=False,
    )

    model = ARVideoPatchTransformer(cfg)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Patches/frame : {cfg.n_patches}")
    print(f"Seq len (train): {(cfg.frames_in + cfg.frames_out) * cfg.n_patches}")
    print(f"Parameters    : {n_params:.2f} M")

    B = 2
    ctx    = torch.randn(B, cfg.frames_in,  3, cfg.resolution, cfg.resolution)
    target = torch.randn(B, cfg.frames_out, 3, cfg.resolution, cfg.resolution)
    
    pred, loss = model(ctx, target)
    print(f"pred shape    : {pred.shape}")
    print(f"train loss    : {loss.item():.4f}")

    gen = model.generate(ctx, n_steps=2)
    print(f"generated     : {gen.shape}")    # (2, 8, 3, 64, 64)