"""
Cosmos-style AR Video Patch Transformer  (FlexAttention + Goal)
================================================================

Stage 1 — Video prediction via teacher forcing (no actions).

Sequence layout (training, Stage 1)
-------------------------------------
    [goal_patches | ctx_patches | tgt_patches]
     N_p tokens    fin×N_p       fout×N_p

    goal  = last frame of the episode (optional target condition)
    ctx   = frames_in context frames
    tgt   = frames_out target frames

Attention mask  (FlexAttention BlockMask)
------------------------------------------
    Block-causal: bidirectional within each "block", causal across blocks.
    Blocks:
        block 0           : goal frame        (N_p tokens)  — optional
        blocks 1..fin     : context frames    (N_p tokens each)
        blocks fin+1..    : target frames     (N_p tokens each)

Interface
----------
    Stage 1 training:   pred_frames, loss = model(ctx, tgt, goal)
    Stage 1 inference:  pred_frames       = model(ctx, goal=goal)

Changes vs original
--------------------
    1. SwiGLUMLP fix: removed redundant up_proj. The original code had both
       gate_proj and up_proj each outputting `inner` dims, then chunked each
       in half — computing 2× the necessary GEMMs and carrying an extra
       d_model×inner parameter matrix. Fixed to a single gate_up_proj that
       outputs `inner` total, split evenly into gate and value halves.
       Saves ~25% of MLP compute and ~d_model×inner parameters per layer.

    2. patchify / unpatchify: added .contiguous() after permute. permute()
       returns a non-contiguous view; reshape() on a non-contiguous tensor
       falls back to an implicit memory copy. .contiguous() makes the copy
       explicit and lets reshape() proceed zero-copy.

    3. generate: replaced the O(T^2) full-recompute loop with a two-phase
       KV-cache approach using flash_attn_with_kvcache:
         Phase 1 (prefill)  — run goal + context through FlexAttention once,
                              populate per-layer KV caches.
         Phase 2 (decode)   — for each target frame run only the N_p new
                              query tokens; all history served from cache.
       Training forward is unchanged (still uses FlexAttention for the
       block-causal mask).
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention.flex_attention import (
    flex_attention,
    create_block_mask,
    BlockMask,
)

from flash_attn import flash_attn_with_kvcache


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class ARPatchConfig:
    # image
    resolution:   int   = 64
    num_channels: int   = 3
    patch_size:   int   = 8

    # transformer
    d_model:          int   = 512
    n_heads:          int   = 8
    n_layers:         int   = 8
    mlp_ratio:        float = 8/3
    dropout:          float = 0.0
    qk_norm:          bool  = True
    parallel_attn:    bool  = False

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
    """(B, T, C, H, W) → (B, T*N_p, C*P*P)

    Fix: .contiguous() after permute prevents the final reshape from
    triggering an implicit full-tensor memory copy.
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
    """(B, T*N_p, C*P*P) → (B, T, C, H, W)"""
    P   = patch_size
    h   = w = resolution // P
    N_p = h * w
    B, L, _ = tokens.shape
    T = L // N_p
    x = tokens.reshape(B * T, h, w, num_channels, P, P)
    x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
    x = x.reshape(B, T, num_channels, h * P, w * P)
    return x


# ---------------------------------------------------------------------------
# Sequence mask via FlexAttention
# ---------------------------------------------------------------------------

def make_sequence_mask(
    N_p: int,
    n_ctx_frames: int,
    n_actions: int,
    n_tgt_frames: int,
    device: torch.device,
    has_goal: bool = True,
) -> BlockMask:
    """Strict token-causal mask for the mixed sequence
        [(goal(N_p)) | ctx(fin*N_p) | actions(m) | tgt(fout*N_p)]

    Previous implementation was block-causal (bidirectional within each
    frame block). That let the model trivially echo tgt[j+1]'s input
    embedding into hidden[prefix_len+j] via attention inside the target
    block — exactly the path shift-by-1 loss was supposed to reward.
    Diagnostic `diagnose_cheating.py` confirmed this: the trained model
    reproduced any in-distribution target with ~zero residual, regardless
    of whether it matched the real future. Strict token-causal removes the
    echo path and matches the flash_attn(causal=True) inference mask.
    """
    goal_end = N_p if has_goal else 0
    ctx_end  = goal_end + n_ctx_frames * N_p
    act_end  = ctx_end + n_actions
    seq_len  = act_end + n_tgt_frames * N_p

    def mask_mod(b, h, q_idx, kv_idx):
        return q_idx >= kv_idx

    return create_block_mask(
        mask_mod, B=1, H=1, Q_LEN=seq_len, KV_LEN=seq_len, device=device,
    )


def make_frames_only_mask(N_p: int, T: int, device: torch.device) -> BlockMask:
    """Strict token-causal mask for a pure-frame sequence (no actions).

    Used for kvcache prefill — must match `make_sequence_mask`'s semantics
    on the prefix region so prefix hidden states at inference equal those
    seen during training.
    """
    seq_len = T * N_p

    def mask_mod(b, h, q_idx, kv_idx):
        return q_idx >= kv_idx

    return create_block_mask(
        mask_mod, B=1, H=1, Q_LEN=seq_len, KV_LEN=seq_len, device=device,
    )


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
        y = (x.float() * rms) * self.scale.float()
        return y.to(x.dtype)


# ---------------------------------------------------------------------------
# Rotary Position Embedding  (RoPE)
# ---------------------------------------------------------------------------

def _build_sin_cos(
    rotary_dim: int,
    max_len: int,
    base: float = 10000.0,
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
    cos = cos.to(dtype=x.dtype)[None, None, :, :]
    sin = sin.to(dtype=x.dtype)[None, None, :, :]
    return x * cos + rotate_half(x) * sin


class RoPEEmbedding(nn.Module):
    """2-D factored RoPE: half head_dim temporal, half spatial."""

    def __init__(self, cfg: ARPatchConfig):
        super().__init__()
        self.head_dim = cfg.head_dim
        if self.head_dim % 4 != 0:
            raise ValueError(
                f"head_dim must be divisible by 4 for 2-D RoPE, got {self.head_dim}"
            )
        rotary_dim = self.head_dim // 2
        temporal_cos, temporal_sin = _build_sin_cos(rotary_dim, cfg.max_temporal_positions)
        spatial_cos, spatial_sin   = _build_sin_cos(rotary_dim, cfg.n_patches)
        self.register_buffer("temporal_cos", temporal_cos, persistent=False)
        self.register_buffer("temporal_sin", temporal_sin, persistent=False)
        self.register_buffer("spatial_cos",  spatial_cos,  persistent=False)
        self.register_buffer("spatial_sin",  spatial_sin,  persistent=False)

    def forward(self, q: torch.Tensor, k: torch.Tensor,
                t_idx: torch.Tensor, s_idx: torch.Tensor,
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """q, k : (B, n_heads, L, head_dim)"""
        q_t, q_s = q.chunk(2, dim=-1)
        k_t, k_s = k.chunk(2, dim=-1)
        t_cos = self.temporal_cos.index_select(0, t_idx)
        t_sin = self.temporal_sin.index_select(0, t_idx)
        s_cos = self.spatial_cos.index_select(0, s_idx)
        s_sin = self.spatial_sin.index_select(0, s_idx)
        q = torch.cat([apply_rope(q_t, t_cos, t_sin), apply_rope(q_s, s_cos, s_sin)], dim=-1)
        k = torch.cat([apply_rope(k_t, t_cos, t_sin), apply_rope(k_s, s_cos, s_sin)], dim=-1)
        return q, k


# ---------------------------------------------------------------------------
# SwiGLU MLP  (fixed)
# ---------------------------------------------------------------------------

class SwiGLUMLP(nn.Module):
    """
    Fix: original had gate_proj + up_proj each outputting `inner` dims,
    then chunked each in half — 2× the GEMM work, half used.

    Now: single gate_up_proj outputs `inner` total; chunk gives gate and
    value each of size inner//2. down_proj maps inner//2 → d_model.
    """

    def __init__(self, d_model: int, mlp_ratio: float, dropout: float = 0.0):
        super().__init__()
        inner = int(d_model * mlp_ratio)
        inner = (inner // 2) * 2               # keep even for chunk
        self.gate_up_proj = nn.Linear(d_model, inner,      bias=False)
        self.down_proj    = nn.Linear(inner // 2, d_model, bias=False)
        self.drop         = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, value = self.gate_up_proj(x).chunk(2, dim=-1)
        return self.drop(self.down_proj(F.silu(gate) * value))


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------

class BlockCausalAttention(nn.Module):
    def __init__(self, cfg: ARPatchConfig):
        super().__init__()
        self.n_heads  = cfg.n_heads
        self.head_dim = cfg.head_dim
        self.d_model  = cfg.d_model

        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.out  = nn.Linear(cfg.d_model, cfg.d_model,     bias=False)

        self.rope   = RoPEEmbedding(cfg)
        self.q_norm = RMSNorm(cfg.head_dim) if cfg.qk_norm else nn.Identity()
        self.k_norm = RMSNorm(cfg.head_dim) if cfg.qk_norm else nn.Identity()

    # ── training path (FlexAttention) ────────────────────────────────
    def forward(self, x: torch.Tensor,
                t_idx: torch.Tensor, s_idx: torch.Tensor,
                block_mask: BlockMask = None) -> torch.Tensor:
        B, L, D = x.shape
        qkv = self.qkv(x).reshape(B, L, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        q = self.q_norm(q)
        k = self.k_norm(k)
        q, k = self.rope(q, k, t_idx, s_idx)
        q = q.to(v.dtype)
        k = k.to(v.dtype)
        out = flex_attention(q, k, v, block_mask=block_mask)
        return self.out(out.transpose(1, 2).reshape(B, L, D))

    # ── inference path (flash_attn KV cache) ─────────────────────────
    def forward_with_kvcache(
        self,
        x: torch.Tensor,                  # (B, L_new, D)
        t_idx: torch.Tensor,
        s_idx: torch.Tensor,
        k_cache: torch.Tensor,            # (B, max_seqlen, n_heads, head_dim)
        v_cache: torch.Tensor,            # (B, max_seqlen, n_heads, head_dim)
        cache_seqlens: torch.Tensor,      # (B,) int32
    ) -> torch.Tensor:
        B, L_new, D = x.shape
        qkv = self.qkv(x).reshape(B, L_new, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)      # each (B, L_new, n_heads, head_dim)

        # Norms operate on head_dim (last dim) — transpose to (B,H,L,D) and back
        q = self.q_norm(q.transpose(1, 2)).transpose(1, 2)
        k = self.k_norm(k.transpose(1, 2)).transpose(1, 2)

        # RoPE expects (B, n_heads, L, head_dim)
        q_r, k_r = self.rope(q.transpose(1, 2), k.transpose(1, 2), t_idx, s_idx)
        q = q_r.transpose(1, 2)          # back to (B, L_new, n_heads, head_dim)
        k = k_r.transpose(1, 2)

        # flash_attn_with_kvcache: writes k,v into cache at cache_seqlens positions
        # and returns attended output (B, L_new, n_heads, head_dim)
        out = flash_attn_with_kvcache(
            q, k, v,
            k_cache, v_cache,
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
        self.attn  = BlockCausalAttention(cfg)
        self.norm2 = RMSNorm(cfg.d_model) if not cfg.parallel_attn else None
        self.mlp   = SwiGLUMLP(cfg.d_model, cfg.mlp_ratio, cfg.dropout)

    # ── training forward ──────────────────────────────────────────────
    def forward(self, x: torch.Tensor,
                t_idx: torch.Tensor, s_idx: torch.Tensor,
                block_mask: BlockMask = None) -> torch.Tensor:
        if self.parallel:
            h = self.norm1(x)
            return x + self.attn(h, t_idx, s_idx, block_mask) + self.mlp(h)
        x = x + self.attn(self.norm1(x), t_idx, s_idx, block_mask)
        x = x + self.mlp(self.norm2(x))
        return x

    # ── inference forward (KV cache) ─────────────────────────────────
    def forward_kvcache_step(
        self,
        x: torch.Tensor,
        t_idx: torch.Tensor,
        s_idx: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
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

        self._mask_cache: dict = {}
        self._init_weights()

    # ------------------------------------------------------------------
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

    def _build_position_indices(
        self, n_ctx: int, n_tgt_frames: int,
        device: torch.device, has_goal: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        N_p = self.cfg.n_patches
        t_list, s_list = [], []
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

    def _frame_position_indices(
        self, frame_t: int, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        N_p = self.cfg.n_patches
        return (
            torch.full((N_p,), frame_t, dtype=torch.long, device=device),
            torch.arange(N_p, dtype=torch.long, device=device),
        )

    # ------------------------------------------------------------------
    # Mask management
    # ------------------------------------------------------------------

    def _ensure_mask(self, n_ctx: int, n_tgt: int,
                     device: torch.device, has_goal: bool = True) -> BlockMask:
        key = (n_ctx, n_tgt, device, has_goal)
        if key not in self._mask_cache:
            N_p = self.cfg.n_patches
            if n_tgt == 0:
                T = (1 + n_ctx) if has_goal else n_ctx
                self._mask_cache[key] = make_frames_only_mask(N_p, T, device)
            else:
                self._mask_cache[key] = make_sequence_mask(
                    N_p, n_ctx, 0, n_tgt, device, has_goal=has_goal,
                )
        return self._mask_cache[key]

    def prebuild_mask(self, device: torch.device, has_goal: bool = True) -> None:
        cfg = self.cfg
        self._ensure_mask(cfg.frames_in, cfg.frames_out, device, has_goal)
        self._ensure_mask(cfg.frames_in, 0, device, has_goal)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _embed_frames(self, frames: torch.Tensor) -> torch.Tensor:
        return self.embed_norm(self.patch_embed(patchify(frames, self.cfg.patch_size)))

    def _run_transformer(self, tokens, t_idx, s_idx, block_mask):
        x = tokens
        for block in self.blocks:
            x = block(x, t_idx, s_idx, block_mask)
        return self.out_norm(x)

    def _decode(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.head(hidden)

    # ------------------------------------------------------------------
    # Forward  (teacher-forcing, FlexAttention)
    # ------------------------------------------------------------------

    def forward(
        self,
        input_frames:  torch.Tensor,
        target_frames: Optional[torch.Tensor] = None,
        goal:          Optional[torch.Tensor] = None,
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

            tokens     = self._embed_frames(all_frames)
            t_idx, s_idx = self._build_position_indices(fin, fout, tokens.device, has_goal)
            block_mask = self._ensure_mask(fin, fout, tokens.device, has_goal)
            hidden     = self._run_transformer(tokens, t_idx, s_idx, block_mask)

            t_start      = prefix_len - 1
            pred_patches = self._decode(hidden[:, t_start : t_start + fout * N_p])
            tgt_patches  = patchify(target_frames, cfg.patch_size)
            loss         = F.mse_loss(pred_patches, tgt_patches)
            pred_frames  = unpatchify(
                pred_patches.detach(), cfg.patch_size, cfg.resolution, cfg.num_channels,
            ).clamp(-1, 1)
            return pred_frames, loss

        else:
            raise RuntimeError(
                "forward() without target_frames is not a valid inference path "
                "under the token-causal training setup. Call model.generate() "
                "for autoregressive rollout instead."
            )

    # ------------------------------------------------------------------
    # AR Generation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        context_frames: torch.Tensor,
        goal: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Two-phase KV-cache autoregressive generation (O(T) per step)."""
        return self._generate_kvcache(context_frames, goal)

    @torch.no_grad()
    def _generate_kvcache(
        self,
        context_frames: torch.Tensor,
        goal: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Patch-level autoregressive KV-cache generation.

        Mirrors the strict token-causal shift-by-1 setup used in training:

          * Prefill [goal, ctx] once. At each layer, extract K (post k_norm,
            post RoPE) and V (raw) and write to the per-layer cache so that
            decode-time queries see the same rotated keys as training.
          * `pred[0]` is decoded from the prefill's last-position hidden
            state (position `prefix_len - 1`, the last ctx patch — the same
            position supervised to predict tgt[0] during training).
          * For j = 1..fout*N_p - 1:
              - Input at absolute position `prefix_len + j - 1` is
                embed(pred[j-1]), with t/s indices of tgt patch (j-1).
              - Its hidden state is decoded as pred[j].

        Total cost: one multi-token prefill + `fout*N_p - 1` single-token
        decode steps. Single-token steps are cheap with
        `flash_attn_with_kvcache` — the sequence-length cost is only in
        the cache read, not recomputation.

        Cache layout: (B, max_seqlen, n_heads, head_dim) per layer,
        matching the flash_attn_with_kvcache signature. max_seqlen =
        prefix_len + fout*N_p - 1 (we only append `fout*N_p - 1` keys
        because the last decoded patch has no successor to condition on).
        """
        cfg      = self.cfg
        N_p      = cfg.n_patches
        fin      = cfg.frames_in
        fout     = cfg.frames_out
        has_goal = goal is not None
        B        = context_frames.shape[0]
        device   = context_frames.device
        dtype    = context_frames.dtype

        ctx = context_frames[:, -fin:]

        # ── Prefill tokens ────────────────────────────────────────────
        prefix_frames = torch.cat([goal.unsqueeze(1), ctx], dim=1) if has_goal else ctx
        prefix_tokens = self._embed_frames(prefix_frames)
        prefix_len    = prefix_tokens.shape[1]
        n_tgt_total   = fout * N_p
        assert n_tgt_total >= 1, "generate() requires frames_out*n_patches >= 1"

        # ── Allocate KV caches ────────────────────────────────────────
        max_seqlen = prefix_len + n_tgt_total - 1
        k_caches = [
            torch.zeros(B, max_seqlen, cfg.n_heads, cfg.head_dim, device=device, dtype=dtype)
            for _ in range(cfg.n_layers)
        ]
        v_caches = [
            torch.zeros(B, max_seqlen, cfg.n_heads, cfg.head_dim, device=device, dtype=dtype)
            for _ in range(cfg.n_layers)
        ]

        # ── Phase 1: Prefill with RoPE-aware cache population ─────────
        t_idx_pre, s_idx_pre = self._build_position_indices(fin, 0, device, has_goal)
        block_mask_pre = self._ensure_mask(fin, 0, device, has_goal)

        x = prefix_tokens
        for layer_idx, block in enumerate(self.blocks):
            h_norm = block.norm1(x)
            qkv = block.attn.qkv(h_norm).reshape(B, prefix_len, 3, cfg.n_heads, cfg.head_dim)
            _, k_pre, v_pre = qkv.unbind(dim=2)     # (B, L, n_heads, head_dim)

            # k_norm operates on (B, n_heads, L, head_dim)
            k_pre = block.attn.k_norm(k_pre.transpose(1, 2)).transpose(1, 2)

            # Apply RoPE to K so cached keys match the rotated K produced
            # during training's full forward. rope() rotates both inputs;
            # we only need the k output (duplicate as dummy q).
            k_rope_in = k_pre.transpose(1, 2)       # (B, n_heads, L, head_dim)
            _, k_pre_rope = block.attn.rope(
                k_rope_in, k_rope_in, t_idx_pre, s_idx_pre,
            )
            k_pre_rope = k_pre_rope.transpose(1, 2)  # (B, L, n_heads, head_dim)

            k_caches[layer_idx][:, :prefix_len].copy_(k_pre_rope)
            v_caches[layer_idx][:, :prefix_len].copy_(v_pre)

            x = block(x, t_idx_pre, s_idx_pre, block_mask_pre)

        # pred[0] comes from the last prefill hidden state (last ctx patch).
        pred_patches: list[torch.Tensor] = [self._decode(self.out_norm(x[:, -1:]))]

        # ── Phase 2: Patch-level AR decode ────────────────────────────
        cache_seqlens = torch.full((B,), prefix_len, dtype=torch.int32, device=device)
        tgt_t_offset  = (1 if has_goal else 0) + fin

        for j in range(1, n_tgt_total):
            # At this step we feed embed(pred[j-1]) as the query token at
            # absolute position prefix_len + j - 1. In training, that
            # position held tgt[j-1]; its RoPE indices are:
            tgt_idx   = j - 1
            frame_idx = tgt_idx // N_p
            patch_idx = tgt_idx % N_p
            t_idx = torch.tensor([tgt_t_offset + frame_idx], dtype=torch.long, device=device)
            s_idx = torch.tensor([patch_idx],                 dtype=torch.long, device=device)

            token_in = self.embed_norm(self.patch_embed(pred_patches[-1]))  # (B, 1, D)

            x = token_in
            for layer_idx, block in enumerate(self.blocks):
                x = block.forward_kvcache_step(
                    x, t_idx, s_idx,
                    k_caches[layer_idx], v_caches[layer_idx],
                    cache_seqlens,
                )

            # Hidden state at position prefix_len + j - 1 → pred[j]
            pred_patches.append(self._decode(self.out_norm(x)))
            cache_seqlens = cache_seqlens + 1

        # ── Stack and unpatchify ──────────────────────────────────────
        all_patches = torch.cat(pred_patches, dim=1)  # (B, n_tgt_total, patch_dim)
        return unpatchify(
            all_patches, cfg.patch_size, cfg.resolution, cfg.num_channels,
        ).clamp(-1, 1)


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
    print(f"Patches/frame  : {cfg.n_patches}")
    print(f"Parameters     : {n_params:.2f} M")

    B      = 2
    ctx    = torch.randn(B, cfg.frames_in,  3, cfg.resolution, cfg.resolution, device=device, dtype=dtype)
    target = torch.randn(B, cfg.frames_out, 3, cfg.resolution, cfg.resolution, device=device, dtype=dtype)
    goal_f = torch.randn(B, 3, cfg.resolution, cfg.resolution, device=device, dtype=dtype)

    model.prebuild_mask(device, has_goal=True)
    pred, loss = model(ctx, target, goal_f)
    print(f"\n=== teacher-forcing (with goal) ===")
    print(f"pred : {pred.shape}  loss : {loss.item():.4f}")

    gen_f = model.generate(ctx, goal_f)
    print(f"\n=== generate (frames_out={cfg.frames_out}, KV-cache) ===")
    print(f"gen  : {gen_f.shape}")

    model.prebuild_mask(device, has_goal=False)
    pred_ng, loss_ng = model(ctx, target)
    print(f"\n=== teacher-forcing (no goal) ===")
    print(f"pred : {pred_ng.shape}  loss : {loss_ng.item():.4f}")
    print(f"gen  : {model.generate(ctx).shape}")