"""
Cosmos-style AR Video Patch Transformer  (flash-attn only, with Goal)
======================================================================

Stage 1 — Video prediction via teacher forcing (no actions).

This version replaces FlexAttention with flash-attn for BOTH training and
inference. Attention is strictly token-causal (causal=True).

Sequence layout (training, Stage 1)
-------------------------------------
    [goal_patches | ctx_patches | tgt_patches]
     N_p tokens    fin*N_p        fout*N_p

    goal  = last frame of the episode (optional target condition)
    ctx   = frames_in  context frames
    tgt   = frames_out target frames

Attention
---------
    Strict token-causal (q_idx >= kv_idx) via flash_attn_func(causal=True)
    for training, and flash_attn_with_kvcache(causal=True) for inference.
    This preserves the fix that prevents shift-by-1 "echo" cheating
    (where block-causal same-frame bidirectional attention lets tgt[j+1]'s
    input embedding leak into hidden[prefix+j]).

Interface
----------
    Stage 1 training:   pred_frames, loss = model(ctx, tgt, goal)
    Stage 1 inference:  pred_frames       = model(ctx, goal=goal)

Implementation notes
--------------------
    * flash-attn K/V layout is (B, L, n_heads, head_dim) — the training
      path avoids an extra transpose vs FlexAttention's (B, H, L, D).
    * RMSNorm on head_dim operates on the last axis, so it works on
      either layout.  RoPE internally reshapes as needed.
    * KV-cache path mirrors the training numerics exactly:
        - prefill stores k AFTER k_norm and RoPE (so cached keys equal
          the rotated keys seen during training).
        - decode writes rope'd k into the cache via flash_attn_with_kvcache.
    * flash-attn supports only fp16 / bf16. fp32 will raise.

Sanity-check reference
----------------------
    If you train on teacher forcing and then call `generate`, the first
    decoded patch (pred[0]) should be bitwise-close (bf16 tolerance) to
    the training-path prediction for that position. Provided as a helper
    at the bottom of this file.
"""

from __future__ import annotations
import contextlib
from dataclasses import dataclass
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from flash_attn import flash_attn_func, flash_attn_with_kvcache

# F.rms_norm (PyTorch 2.4+) dispatches to a fused ATen kernel that uses
# fp32 accumulation internally and returns the input dtype — same numerics
# as the manual impl below but without the intermediate fp32 tensors.
_HAS_F_RMS_NORM = hasattr(F, "rms_norm")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class ARPatchConfig:
    # image
    resolution:   int = 64
    num_channels: int = 3
    patch_size:   int = 8

    # transformer
    d_model:       int   = 512
    n_heads:       int   = 8
    n_layers:      int   = 8
    mlp_ratio:     float = 8 / 3
    dropout:       float = 0.0
    qk_norm:       bool  = True
    parallel_attn: bool  = False

    # AR
    frames_in:  int = 4
    frames_out: int = 4

    # actions (used by Stage 2 inverse-dynamics models)
    action_dim: int = 7

    @property
    def n_patches(self) -> int:
        return (self.resolution // self.patch_size) ** 2

    @property
    def patch_dim(self) -> int:
        return self.num_channels * self.patch_size * self.patch_size

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    @property
    def max_temporal_positions(self) -> int:
        return self.frames_in + self.frames_out + 1


# ---------------------------------------------------------------------------
# Patch utilities
# ---------------------------------------------------------------------------

def patchify(frames: torch.Tensor, patch_size: int) -> torch.Tensor:
    """(B, T, C, H, W) -> (B, T*N_p, C*P*P).

    .contiguous() after permute makes the final reshape zero-copy.
    """
    B, T, C, H, W = frames.shape
    P = patch_size
    h, w = H // P, W // P
    x = frames.reshape(B * T, C, h, P, w, P)
    x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
    x = x.reshape(B, T * h * w, C * P * P)
    return x


def unpatchify(tokens: torch.Tensor, patch_size: int,
               resolution: int, num_channels: int) -> torch.Tensor:
    """(B, T*N_p, C*P*P) -> (B, T, C, H, W)."""
    P = patch_size
    h = w = resolution // P
    N_p = h * w
    B, L, _ = tokens.shape
    T = L // N_p
    x = tokens.reshape(B * T, h, w, num_channels, P, P)
    x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
    x = x.reshape(B, T, num_channels, h * P, w * P)
    return x


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.normalized_shape = (dim,)
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if _HAS_F_RMS_NORM:
            return F.rms_norm(x, self.normalized_shape, self.scale, self.eps)
        rms = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        y = (x.float() * rms) * self.scale.float()
        return y.to(x.dtype)


# ---------------------------------------------------------------------------
# Rotary Position Embedding  (2-D factored)
# ---------------------------------------------------------------------------

def _build_sin_cos(
    rotary_dim: int, max_len: int, base: float = 10000.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if rotary_dim % 2 != 0:
        raise ValueError(f"rotary_dim must be even, got {rotary_dim}")
    inv_freq = 1.0 / (
        base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim)
    )
    positions = torch.arange(max_len, dtype=torch.float32)
    angles = torch.outer(positions, inv_freq).repeat_interleave(2, dim=-1)
    return angles.cos(), angles.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(start_dim=-2)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: (B, L, H, D).  cos/sin: (L, D)."""
    cos = cos.to(dtype=x.dtype)[None, :, None, :]
    sin = sin.to(dtype=x.dtype)[None, :, None, :]
    return x * cos + rotate_half(x) * sin


class RoPEEmbedding(nn.Module):
    """Half of head_dim rotates on temporal index, other half on spatial."""

    def __init__(self, cfg: ARPatchConfig):
        super().__init__()
        self.head_dim = cfg.head_dim
        if self.head_dim % 4 != 0:
            raise ValueError(
                f"head_dim must be divisible by 4 for 2-D RoPE, got {self.head_dim}"
            )
        rotary_dim = self.head_dim // 2
        t_cos, t_sin = _build_sin_cos(rotary_dim, cfg.max_temporal_positions)
        s_cos, s_sin = _build_sin_cos(rotary_dim, cfg.n_patches)
        self.register_buffer("temporal_cos", t_cos, persistent=False)
        self.register_buffer("temporal_sin", t_sin, persistent=False)
        self.register_buffer("spatial_cos",  s_cos, persistent=False)
        self.register_buffer("spatial_sin",  s_sin, persistent=False)

    def forward(
        self, q: torch.Tensor, k: torch.Tensor,
        t_idx: torch.Tensor, s_idx: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """q, k: (B, L, n_heads, head_dim). t_idx, s_idx: (L,)."""
        q_t, q_s = q.chunk(2, dim=-1)
        k_t, k_s = k.chunk(2, dim=-1)
        t_cos = self.temporal_cos.index_select(0, t_idx)
        t_sin = self.temporal_sin.index_select(0, t_idx)
        s_cos = self.spatial_cos.index_select(0, s_idx)
        s_sin = self.spatial_sin.index_select(0, s_idx)
        q = torch.cat(
            [apply_rope(q_t, t_cos, t_sin), apply_rope(q_s, s_cos, s_sin)],
            dim=-1,
        )
        k = torch.cat(
            [apply_rope(k_t, t_cos, t_sin), apply_rope(k_s, s_cos, s_sin)],
            dim=-1,
        )
        return q, k


# ---------------------------------------------------------------------------
# SwiGLU MLP
# ---------------------------------------------------------------------------

class SwiGLUMLP(nn.Module):
    """Single gate_up_proj -> chunk in half. No redundant up_proj."""

    def __init__(self, d_model: int, mlp_ratio: float, dropout: float = 0.0):
        super().__init__()
        inner = int(d_model * mlp_ratio)
        inner = (inner // 2) * 2
        self.gate_up_proj = nn.Linear(d_model, inner, bias=False)
        self.down_proj    = nn.Linear(inner // 2, d_model, bias=False)
        self.drop         = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, value = self.gate_up_proj(x).chunk(2, dim=-1)
        return self.drop(self.down_proj(F.silu(gate) * value))


# ---------------------------------------------------------------------------
# Attention  (flash-attn for both training and inference)
# ---------------------------------------------------------------------------

class CausalAttention(nn.Module):
    """Strict token-causal attention using flash-attn.

    Training:  flash_attn_func(q, k, v, causal=True)
    Inference: flash_attn_with_kvcache(q, k, v, k_cache, v_cache, ...)
    Both expect (B, L, n_heads, head_dim) layout.
    """

    def __init__(self, cfg: ARPatchConfig):
        super().__init__()
        self.n_heads  = cfg.n_heads
        self.head_dim = cfg.head_dim
        self.d_model  = cfg.d_model

        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.out = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

        self.rope   = RoPEEmbedding(cfg)
        self.q_norm = RMSNorm(cfg.head_dim) if cfg.qk_norm else nn.Identity()
        self.k_norm = RMSNorm(cfg.head_dim) if cfg.qk_norm else nn.Identity()

    # ── training path ────────────────────────────────────────────────
    def forward(
        self, x: torch.Tensor,
        t_idx: torch.Tensor, s_idx: torch.Tensor,
    ) -> torch.Tensor:
        """x: (B, L, D) -> (B, L, D)."""
        B, L, D = x.shape
        qkv = self.qkv(x).reshape(B, L, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)  # each (B, L, n_heads, head_dim)

        # RMSNorm + RoPE both act on head_dim (last axis); keeping the
        # (B, L, H, D) layout that flash-attn already wants avoids two
        # transposes per branch and keeps tensors contiguous.
        q = self.q_norm(q)
        k = self.k_norm(k)
        q, k = self.rope(q, k, t_idx, s_idx)

        q = q.to(v.dtype)
        k = k.to(v.dtype)

        out = flash_attn_func(q, k, v, causal=True)  # (B, L, H, D)
        return self.out(out.reshape(B, L, D))

    # ── inference path (KV cache) ────────────────────────────────────
    def forward_with_kvcache(
        self,
        x: torch.Tensor,                  # (B, L_new, D)
        t_idx: torch.Tensor,              # (L_new,)
        s_idx: torch.Tensor,              # (L_new,)
        k_cache: torch.Tensor,            # (B, max_seqlen, H, D)
        v_cache: torch.Tensor,            # (B, max_seqlen, H, D)
        cache_seqlens: torch.Tensor,      # (B,) int32
    ) -> torch.Tensor:
        B, L_new, D = x.shape
        qkv = self.qkv(x).reshape(B, L_new, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)

        q = self.q_norm(q)
        k = self.k_norm(k)
        q, k = self.rope(q, k, t_idx, s_idx)
        q = q.to(v.dtype)
        k = k.to(v.dtype)

        # flash_attn_with_kvcache signature is (q, k_cache, v_cache, k=None,
        # v=None, ...) — NOT (q, k, v, k_cache, v_cache). Use keyword args
        # for k/v to avoid the position-argument trap. Given the new (k, v)
        # this function appends them at position `cache_seqlens` per batch
        # and computes causal attention of q against the populated cache.
        out = flash_attn_with_kvcache(
            q,
            k_cache, v_cache,
            k=k, v=v,
            cache_seqlens=cache_seqlens,
            causal=True,
        )
        return self.out(out.reshape(B, L_new, D))


# ---------------------------------------------------------------------------
# Transformer Block
# ---------------------------------------------------------------------------

class CosmosBlock(nn.Module):
    def __init__(self, cfg: ARPatchConfig):
        super().__init__()
        self.parallel = cfg.parallel_attn
        self.norm1 = RMSNorm(cfg.d_model)
        self.attn  = CausalAttention(cfg)
        self.norm2 = RMSNorm(cfg.d_model) if not cfg.parallel_attn else None
        self.mlp   = SwiGLUMLP(cfg.d_model, cfg.mlp_ratio, cfg.dropout)

    def forward(
        self, x: torch.Tensor,
        t_idx: torch.Tensor, s_idx: torch.Tensor,
    ) -> torch.Tensor:
        if self.parallel:
            h = self.norm1(x)
            return x + self.attn(h, t_idx, s_idx) + self.mlp(h)
        x = x + self.attn(self.norm1(x), t_idx, s_idx)
        x = x + self.mlp(self.norm2(x))
        return x

    def forward_kvcache_step(
        self,
        x: torch.Tensor,
        t_idx: torch.Tensor, s_idx: torch.Tensor,
        k_cache: torch.Tensor, v_cache: torch.Tensor,
        cache_seqlens: torch.Tensor,
    ) -> torch.Tensor:
        if self.parallel:
            h = self.norm1(x)
            return x + self.attn.forward_with_kvcache(
                h, t_idx, s_idx, k_cache, v_cache, cache_seqlens
            ) + self.mlp(h)
        x = x + self.attn.forward_with_kvcache(
            self.norm1(x), t_idx, s_idx, k_cache, v_cache, cache_seqlens
        )
        x = x + self.mlp(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class ARVideoPatchTransformer(nn.Module):
    def __init__(self, cfg: ARPatchConfig):
        super().__init__()
        self.cfg = cfg

        self.patch_embed = nn.Linear(cfg.patch_dim, cfg.d_model, bias=False)
        self.embed_norm  = RMSNorm(cfg.d_model)

        self.blocks   = nn.ModuleList([CosmosBlock(cfg) for _ in range(cfg.n_layers)])
        self.out_norm = RMSNorm(cfg.d_model)
        self.head     = nn.Linear(cfg.d_model, cfg.patch_dim, bias=False)

        self._init_weights()

        # Pre-build position-index buffers for all (has_goal, n_tgt) combos
        # used in training-forward (n_tgt = frames_out) and in generate()'s
        # prefill (n_tgt = 0). These are fixed given cfg, so caching avoids
        # reallocating + torch.cat'ing on every forward (which would cause
        # torch.compile mode="reduce-overhead" to miss its CUDA graph).
        # persistent=False → not serialized into checkpoints, stays .to()-ed
        # with the module.
        for has_goal in (True, False):
            for n_tgt in (cfg.frames_out, 0):
                t_idx, s_idx = self._compute_position_indices(
                    cfg.frames_in, n_tgt, torch.device("cpu"), has_goal,
                )
                tag = f"{int(has_goal)}_{cfg.frames_in}_{n_tgt}"
                self.register_buffer(f"_tidx_{tag}", t_idx, persistent=False)
                self.register_buffer(f"_sidx_{tag}", s_idx, persistent=False)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, RMSNorm):
                nn.init.ones_(m.scale)
        scale = (2 * self.cfg.n_layers) ** -0.5
        for name, p in self.named_parameters():
            if "out.weight" in name or "down_proj.weight" in name:
                p.data.mul_(scale)

    # ------------------------------------------------------------------
    # Position indices
    # ------------------------------------------------------------------

    def _compute_position_indices(
        self, n_ctx: int, n_tgt_frames: int,
        device: torch.device, has_goal: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Raw builder — used once per (has_goal, n_ctx, n_tgt) combo to
        populate the cached buffers, or as a fallback for unknown configs."""
        N_p = self.cfg.n_patches
        t_list: List[torch.Tensor] = []
        s_list: List[torch.Tensor] = []
        t_off = 0
        if has_goal:
            t_list.append(torch.zeros(N_p, dtype=torch.long, device=device))
            s_list.append(torch.arange(N_p, dtype=torch.long, device=device))
            t_off = 1
        for i in range(n_ctx):
            t_list.append(torch.full((N_p,), t_off + i, dtype=torch.long, device=device))
            s_list.append(torch.arange(N_p, dtype=torch.long, device=device))
        tgt_t = t_off + n_ctx
        for k in range(n_tgt_frames):
            t_list.append(torch.full((N_p,), tgt_t + k, dtype=torch.long, device=device))
            s_list.append(torch.arange(N_p, dtype=torch.long, device=device))
        return torch.cat(t_list), torch.cat(s_list)

    def _build_position_indices(
        self, n_ctx: int, n_tgt_frames: int,
        device: torch.device, has_goal: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (t_idx, s_idx) for the given sequence layout, reusing
        pre-registered buffers when possible so torch.compile doesn't see
        fresh tensor allocations every step."""
        tag = f"{int(has_goal)}_{n_ctx}_{n_tgt_frames}"
        t_name = f"_tidx_{tag}"
        s_name = f"_sidx_{tag}"
        if hasattr(self, t_name):
            return getattr(self, t_name), getattr(self, s_name)
        # Fallback for unexpected configs (e.g. a Stage-2 subclass calling
        # with a different n_ctx/n_tgt). Slow path — still correct.
        return self._compute_position_indices(n_ctx, n_tgt_frames, device, has_goal)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def prebuild_mask(self, device: torch.device, has_goal: bool = True) -> None:
        """No-op kept for backwards compatibility with trainers that call it.

        The old FlexAttention path needed to pre-compile BlockMask tensors
        for (fin, fout) and (fin, 0). flash_attn applies causal=True at the
        kernel level with no precomputation, so nothing to do here.
        """
        del device, has_goal

    def _embed_frames(self, frames: torch.Tensor) -> torch.Tensor:
        return self.embed_norm(self.patch_embed(patchify(frames, self.cfg.patch_size)))

    def _run_transformer(self, tokens, t_idx, s_idx):
        x = tokens
        for block in self.blocks:
            x = block(x, t_idx, s_idx)
        return self.out_norm(x)

    def _decode(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.head(hidden)

    # ------------------------------------------------------------------
    # Forward  (teacher-forcing)
    # ------------------------------------------------------------------

    def forward(
        self,
        input_frames:  torch.Tensor,
        target_frames: Optional[torch.Tensor] = None,
        goal:          Optional[torch.Tensor] = None,
        *,
        return_pred_frames: bool = True,
    ):
        cfg = self.cfg
        N_p = cfg.n_patches

        if target_frames is not None:
            fin      = cfg.frames_in
            fout     = cfg.frames_out
            has_goal = goal is not None

            if has_goal:
                all_frames = torch.cat([goal.unsqueeze(1), input_frames, target_frames], dim=1)
                prefix_len = (1 + fin) * N_p
            else:
                all_frames = torch.cat([input_frames, target_frames], dim=1)
                prefix_len = fin * N_p

            tokens       = self._embed_frames(all_frames)
            t_idx, s_idx = self._build_position_indices(fin, fout, tokens.device, has_goal)
            hidden       = self._run_transformer(tokens, t_idx, s_idx)

            t_start      = prefix_len - 1
            pred_patches = self._decode(hidden[:, t_start : t_start + fout * N_p])
            tgt_patches  = patchify(target_frames, cfg.patch_size)
            loss         = F.mse_loss(pred_patches, tgt_patches)
            # Training path discards pred_frames, so skip unpatchify/clamp
            # when the caller only wants the loss.
            if not return_pred_frames:
                return None, loss
            pred_frames  = unpatchify(
                pred_patches.detach(),
                cfg.patch_size, cfg.resolution, cfg.num_channels,
            ).clamp(-1, 1)
            return pred_frames, loss

        raise RuntimeError(
            "forward() without target_frames is not a valid inference path "
            "under the token-causal shift-by-1 training setup: only position "
            "prefix_len - 1 is supervised (to predict tgt[0]); the other "
            "N_p - 1 ctx positions decoded by the old `hidden[-N_p:]` trick "
            "have no training signal and produce garbage. "
            "Call model.generate(ctx, goal=goal) for autoregressive rollout."
        )

    # ------------------------------------------------------------------
    # AR Generation  (two-phase KV cache)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        context_frames: torch.Tensor,
        goal: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Patch-level AR generation. Mirrors training's shift-by-1 exactly.

        Phase 1 (prefill): run [goal, ctx] through the standard training
        forward. After each layer, we also compute k (post k_norm + RoPE)
        and v (raw) for that layer's input and write them to the cache,
        so that decode-time queries see exactly the rotated keys that
        flash_attn_func saw during training.

        pred[0] comes from the prefill's last hidden state (position
        prefix_len - 1, the last ctx patch — the same position supervised
        to predict tgt[0] in training).

        Phase 2 (decode): for j = 1 .. fout*N_p - 1, embed pred[j-1] as the
        query at absolute position prefix_len + j - 1 (t/s indices match
        tgt patch (j-1)'s training position), run one token through every
        layer via flash_attn_with_kvcache, decode its hidden as pred[j].

        Total: 1 prefill + (fout*N_p - 1) single-token decode steps.
        """
        cfg      = self.cfg
        N_p      = cfg.n_patches
        fin      = cfg.frames_in
        fout     = cfg.frames_out
        has_goal = goal is not None
        B        = context_frames.shape[0]
        device   = context_frames.device
        # NOTE: the KV-cache dtype is derived below from prefix_tokens (after
        # patch_embed/embed_norm), not from context_frames — autocast may
        # change the dtype between these two points.

        ctx = context_frames[:, -fin:]

        # ── Prefill tokens ────────────────────────────────────────────
        prefix_frames = (
            torch.cat([goal.unsqueeze(1), ctx], dim=1) if has_goal else ctx
        )
        prefix_tokens = self._embed_frames(prefix_frames)
        prefix_len    = prefix_tokens.shape[1]
        n_tgt_total   = fout * N_p
        assert n_tgt_total >= 1, "generate() requires frames_out*n_patches >= 1"

        # ── Allocate KV caches ────────────────────────────────────────
        # Last decoded patch has no successor, so only fout*N_p - 1 keys
        # beyond the prefix need to be appended. We allocate each layer's
        # cache LAZILY inside the prefill loop using v_pre.dtype — this is
        # the only reliable way to match flash_attn's dtype expectations,
        # because under torch.amp.autocast some ops (F.rms_norm / LayerNorm)
        # get promoted to fp32 for precision while Linear outputs are kept
        # at the autocast dtype, so prefix_tokens.dtype does NOT generally
        # equal the dtype of q/k/v inside the block. flash_attn requires
        # q.dtype == k_cache.dtype; the lazy allocation guarantees it.
        max_seqlen = prefix_len + n_tgt_total - 1
        k_caches: List[Optional[torch.Tensor]] = [None] * cfg.n_layers
        v_caches: List[Optional[torch.Tensor]] = [None] * cfg.n_layers

        # ── Phase 1: Prefill with RoPE-aware cache population ─────────
        t_idx_pre, s_idx_pre = self._build_position_indices(fin, 0, device, has_goal)

        x = prefix_tokens
        for layer_idx, block in enumerate(self.blocks):
            # Compute this layer's K/V from x, matching the training forward
            # up through q_norm / k_norm / RoPE (we only need k and v).
            h_norm = block.norm1(x)
            qkv = block.attn.qkv(h_norm).reshape(
                B, prefix_len, 3, cfg.n_heads, cfg.head_dim
            )
            _, k_pre, v_pre = qkv.unbind(dim=2)  # (B, L, H, D) each

            # Allocate cache in the true dtype that flash_attn will see.
            k_caches[layer_idx] = torch.zeros(
                B, max_seqlen, cfg.n_heads, cfg.head_dim,
                device=device, dtype=v_pre.dtype,
            )
            v_caches[layer_idx] = torch.zeros(
                B, max_seqlen, cfg.n_heads, cfg.head_dim,
                device=device, dtype=v_pre.dtype,
            )

            # k_norm / RoPE now operate directly on (B, L, H, D) — must
            # match the training path in CausalAttention.forward so the
            # cached keys equal the rotated keys seen during training.
            k_pre_normed = block.attn.k_norm(k_pre)
            _, k_rot = block.attn.rope(
                k_pre_normed, k_pre_normed, t_idx_pre, s_idx_pre,
            )
            k_pre_final = k_rot.to(v_pre.dtype)  # (B, L, H, D)

            k_caches[layer_idx][:, :prefix_len].copy_(k_pre_final)
            v_caches[layer_idx][:, :prefix_len].copy_(v_pre)

            # Normal forward for the next layer's input
            x = block(x, t_idx_pre, s_idx_pre)

        # Final norm and decode the last prefix hidden -> pred[0]
        x = self.out_norm(x)
        pred_patches: List[torch.Tensor] = [self._decode(x[:, -1:])]

        # ── Phase 2: Patch-level AR decode ────────────────────────────
        cache_seqlens = torch.full((B,), prefix_len, dtype=torch.int32, device=device)
        tgt_t_offset  = (1 if has_goal else 0) + fin

        for j in range(1, n_tgt_total):
            # Query is at absolute position prefix_len + j - 1, which
            # corresponds to tgt patch (j-1) in the training layout.
            tgt_idx   = j - 1
            frame_idx = tgt_idx // N_p
            patch_idx = tgt_idx %  N_p
            t_idx = torch.tensor([tgt_t_offset + frame_idx],
                                 dtype=torch.long, device=device)
            s_idx = torch.tensor([patch_idx],
                                 dtype=torch.long, device=device)

            # Input: embedding of the previously-predicted patch.
            token_in = self.embed_norm(self.patch_embed(pred_patches[-1]))  # (B,1,D)

            x = token_in
            for layer_idx, block in enumerate(self.blocks):
                x = block.forward_kvcache_step(
                    x, t_idx, s_idx,
                    k_caches[layer_idx], v_caches[layer_idx],
                    cache_seqlens,
                )

            x = self.out_norm(x)
            pred_patches.append(self._decode(x))
            cache_seqlens = cache_seqlens + 1

        all_patches = torch.cat(pred_patches, dim=1)  # (B, fout*N_p, patch_dim)
        return unpatchify(
            all_patches, cfg.patch_size, cfg.resolution, cfg.num_channels,
        ).clamp(-1, 1)


# ---------------------------------------------------------------------------
# Consistency helper  (compare training path vs KV-cache path for the first
# supervised patch — pred_patches[0], which maps to the first patch of tgt[0])
# ---------------------------------------------------------------------------

@torch.no_grad()
def check_generate_matches_training(
    model: ARVideoPatchTransformer,
    ctx: torch.Tensor,
    goal: Optional[torch.Tensor] = None,
) -> float:
    """Max abs diff between the two paths' prediction of the **first**
    target patch (i.e. patch 0 of frame 0 of the target region).

    Both paths compute this from the hidden state at absolute position
    ``prefix_len - 1`` (the last context patch), which under token-causal
    attention sees only [goal, ctx] — no target dependence. They must
    therefore agree up to bf16 rounding (~1e-2 typical).

    * training path: give any `target_frames` (content is ignored for
      this position under token-causal), take ``pred_patches[:, 0]``.
    * generate path: take the first decoded patch ``pred_patches[0]``,
      which is exactly the first patch of the unpatchified frame[:, 0].
    """
    model.eval()
    cfg = model.cfg

    # Training path needs SOME target to enter the if-branch. Token-causal
    # attention + prefix_len-1 decode position can't see the target block,
    # so the value of `dummy_target` is irrelevant for the first patch.
    B = ctx.shape[0]
    dummy_target = torch.zeros(
        B, cfg.frames_out, cfg.num_channels, cfg.resolution, cfg.resolution,
        device=ctx.device, dtype=ctx.dtype,
    )
    # flash-attn only supports fp16/bf16. If caller kept the model + inputs
    # in fp32, run both paths under bf16 autocast so the diagnostic still
    # works without requiring the caller to cast everything manually.
    need_autocast = (
        ctx.is_cuda
        and ctx.dtype == torch.float32
        and not torch.is_autocast_enabled()
    )
    amp_ctx = (
        torch.amp.autocast("cuda", dtype=torch.bfloat16)
        if need_autocast
        else contextlib.nullcontext()
    )
    with amp_ctx:
        pred_ref, _ = model(ctx, dummy_target, goal)   # (B, fout, C, H, W)
        pred_gen    = model.generate(ctx, goal=goal)   # (B, fout, C, H, W)

    # Compare the first patch of frame 0 in both outputs. Unpatchify lays
    # patches out in row-major, so the top-left patch is
    # pred[..., :patch_size, :patch_size].
    P = cfg.patch_size
    a = pred_ref[:, 0, :, :P, :P]
    b = pred_gen[:, 0, :, :P, :P]
    return (a - b).float().abs().max().item()


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cfg = ARPatchConfig(
        resolution=64, num_channels=3, patch_size=8,
        d_model=512, n_heads=8, n_layers=8,
        frames_in=1, frames_out=4,
        action_dim=7,
        qk_norm=True, parallel_attn=False,
    )

    device = torch.device("cuda")
    dtype  = torch.bfloat16

    model = ARVideoPatchTransformer(cfg).to(device=device, dtype=dtype)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Patches/frame : {cfg.n_patches}")
    print(f"Parameters    : {n_params:.2f} M")

    B      = 2
    ctx    = torch.randn(B, cfg.frames_in,  3, cfg.resolution, cfg.resolution,
                         device=device, dtype=dtype)
    target = torch.randn(B, cfg.frames_out, 3, cfg.resolution, cfg.resolution,
                         device=device, dtype=dtype)
    goal_f = torch.randn(B, 3, cfg.resolution, cfg.resolution,
                         device=device, dtype=dtype)

    pred, loss = model(ctx, target, goal_f)
    print(f"\n=== teacher-forcing (with goal) ===")
    print(f"pred : {pred.shape}   loss : {loss.item():.4f}")

    gen_f = model.generate(ctx, goal_f)
    print(f"\n=== generate (frames_out={cfg.frames_out}, KV-cache) ===")
    print(f"gen  : {gen_f.shape}")

    pred_ng, loss_ng = model(ctx, target)
    print(f"\n=== teacher-forcing (no goal) ===")
    print(f"pred : {pred_ng.shape}   loss : {loss_ng.item():.4f}")
    print(f"gen  : {model.generate(ctx).shape}")

    # Numerical consistency check
    diff = check_generate_matches_training(model, ctx, goal_f)
    print(f"\n=== pred[0] consistency (train vs generate) ===")
    print(f"max |diff| : {diff:.4e}   (bf16 typical ~1e-2)")