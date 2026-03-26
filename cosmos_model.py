"""
Cosmos-style AR Video Patch Transformer  (FlexAttention + Actions + Goal)
==========================================================================
Predicts both **target frames** and **intermediate actions** via teacher forcing.

Sequence layout (training)
---------------------------
    [goal_patches | ctx_patches | action_tokens | tgt_patches]
     N_p tokens    fin×N_p       m tokens        fout×N_p

    goal  = last frame of the episode (target condition)
    ctx   = frames_in context frames
    actions = m action vectors between last ctx frame and first target frame
    tgt   = frames_out target frames

Attention mask  (FlexAttention BlockMask)
------------------------------------------
    Block-causal: bidirectional within each "block", causal across blocks.
    Blocks:
        block 0           : goal frame        (N_p tokens)
        blocks 1..fin     : context frames    (N_p tokens each)
        blocks fin+1..fin+m : action tokens   (1 token each)
        blocks fin+m+1..  : target frames     (N_p tokens each)

Prediction heads
-----------------
    action_head : d_model → action_dim   (predict actions)
    head        : d_model → patch_dim    (predict frame patches)

Interface
----------
    Training:   pred_frames, pred_actions, loss = model(ctx, tgt, actions, goal)
    Generation: gen_frames, gen_actions = model.generate(ctx, goal, n_actions)
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

    # actions
    action_dim: int = 7
    max_actions: int = 0

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
        return self.frames_in + self.frames_out + self.max_actions + 1


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
    """
    Block-causal mask for the mixed sequence:
        [(goal(N_p)) | ctx(fin*N_p) | actions(m) | tgt(fout*N_p)]

    Blocks are: (optional goal frame,) each ctx frame, each action (size 1),
    each tgt frame.  Causal across blocks, bidirectional within.
    """
    goal_end = N_p if has_goal else 0
    ctx_end  = goal_end + n_ctx_frames * N_p
    act_end  = ctx_end + n_actions
    seq_len  = act_end + n_tgt_frames * N_p

    _fin = n_ctx_frames
    _m   = n_actions
    _Np  = N_p

    def mask_mod(b, h, q_idx, kv_idx):
        def _block(idx):
            return torch.where(
                idx < goal_end,
                torch.zeros_like(idx),
                torch.where(
                    idx < ctx_end,
                    1 + (idx - goal_end) // _Np,
                    torch.where(
                        idx < act_end,
                        1 + _fin + (idx - ctx_end),
                        1 + _fin + _m + (idx - act_end) // _Np,
                    ),
                ),
            )
        return _block(q_idx) >= _block(kv_idx)

    return create_block_mask(
        mask_mod, B=1, H=1, Q_LEN=seq_len, KV_LEN=seq_len, device=device,
    )


def make_frames_only_mask(N_p: int, T: int, device: torch.device) -> BlockMask:
    """Simple block-causal mask for a pure-frame sequence (no actions)."""
    seq_len = T * N_p

    def mask_mod(b, h, q_idx, kv_idx):
        return q_idx // N_p >= kv_idx // N_p

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
    """
    x   : (B, n_heads, L, rotary_dim)
    cos : (L, rotary_dim)
    sin : (L, rotary_dim)
    """
    cos = cos.to(dtype=x.dtype)[None, None, :, :]
    sin = sin.to(dtype=x.dtype)[None, None, :, :]
    return x * cos + rotate_half(x) * sin


class RoPEEmbedding(nn.Module):
    """
    2-D factored RoPE accepting explicit (t_idx, s_idx) position arrays.

    half head_dim for temporal, half for spatial.
    """

    def __init__(self, cfg: ARPatchConfig):
        super().__init__()
        self.head_dim = cfg.head_dim
        if self.head_dim % 4 != 0:
            raise ValueError(
                f"head_dim must be divisible by 4 for 2-D RoPE, got {self.head_dim}"
            )

        rotary_dim = self.head_dim // 2
        temporal_cos, temporal_sin = _build_sin_cos(
            rotary_dim, cfg.max_temporal_positions
        )
        spatial_cos, spatial_sin = _build_sin_cos(rotary_dim, cfg.n_patches)
        self.register_buffer("temporal_cos", temporal_cos, persistent=False)
        self.register_buffer("temporal_sin", temporal_sin, persistent=False)
        self.register_buffer("spatial_cos", spatial_cos, persistent=False)
        self.register_buffer("spatial_sin", spatial_sin, persistent=False)

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
# SwiGLU MLP
# ---------------------------------------------------------------------------

class SwiGLUMLP(nn.Module):
    def __init__(self, d_model: int, mlp_ratio: float, dropout: float = 0.0):
        super().__init__()
        inner = int(d_model * mlp_ratio)
        inner = (inner // 2) * 2
        self.gate_proj = nn.Linear(d_model, inner, bias=False)
        self.up_proj   = nn.Linear(d_model, inner, bias=False)
        self.down_proj = nn.Linear(inner // 2, d_model, bias=False)
        self.drop      = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate  = F.silu(self.gate_proj(x))
        value = self.up_proj(x)
        g1, g2 = gate.chunk(2, dim=-1)
        v1, v2 = value.chunk(2, dim=-1)
        hidden = g1 * v1 + g2 * v2
        return self.drop(self.down_proj(hidden))


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
        self.out = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

        self.rope   = RoPEEmbedding(cfg)
        self.q_norm = RMSNorm(cfg.head_dim) if cfg.qk_norm else nn.Identity()
        self.k_norm = RMSNorm(cfg.head_dim) if cfg.qk_norm else nn.Identity()

    def forward(self, x: torch.Tensor,
                t_idx: torch.Tensor, s_idx: torch.Tensor,
                block_mask: BlockMask = None) -> torch.Tensor:
        B, L, D = x.shape
        qkv = self.qkv(x).reshape(B, L, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        q = self.q_norm(q)
        k = self.k_norm(k)
        q, k = self.rope(q, k, t_idx, s_idx)
        if q.dtype != v.dtype:
            q = q.to(v.dtype)
        if k.dtype != v.dtype:
            k = k.to(v.dtype)
        out = flex_attention(q, k, v, block_mask=block_mask)
        out = out.transpose(1, 2).reshape(B, L, D)
        return self.out(out)


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

    def forward(self, x: torch.Tensor,
                t_idx: torch.Tensor, s_idx: torch.Tensor,
                block_mask: BlockMask = None) -> torch.Tensor:
        if self.parallel:
            h = self.norm1(x)
            return x + self.attn(h, t_idx, s_idx, block_mask) + self.mlp(h)
        else:
            x = x + self.attn(self.norm1(x), t_idx, s_idx, block_mask)
            x = x + self.mlp(self.norm2(x))
            return x


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class ARVideoPatchTransformer(nn.Module):
    def __init__(self, cfg: ARPatchConfig):
        super().__init__()
        self.cfg = cfg

        # Frame embeddings
        self.patch_embed = nn.Linear(cfg.patch_dim, cfg.d_model, bias=False)
        self.embed_norm  = RMSNorm(cfg.d_model)

        # Action embeddings
        self.action_embed = nn.Linear(cfg.action_dim, cfg.d_model, bias=False)
        self.action_norm  = RMSNorm(cfg.d_model)

        # Transformer
        self.blocks   = nn.ModuleList([CosmosBlock(cfg) for _ in range(cfg.n_layers)])
        self.out_norm = RMSNorm(cfg.d_model)

        # Output heads
        self.head        = nn.Linear(cfg.d_model, cfg.patch_dim, bias=False)   # frames
        self.action_head = nn.Linear(cfg.d_model, cfg.action_dim, bias=False)  # actions

        # Mask cache
        self._mask_cache: dict = {}

        self._init_weights()

    # ------------------------------------------------------------------
    # Init
    # ------------------------------------------------------------------

    def _init_weights(self):
        for name, m in self.named_modules():
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
        self, n_ctx: int, n_actions: int, n_tgt_frames: int,
        device: torch.device, has_goal: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build (t_idx, s_idx) for [(goal) | ctx | actions | tgt]."""
        N_p = self.cfg.n_patches
        t_list, s_list = [], []
        t_off = 0

        # Goal frame (optional): t=0
        if has_goal:
            t_list.append(torch.zeros(N_p, dtype=torch.long, device=device))
            s_list.append(torch.arange(N_p, dtype=torch.long, device=device))
            t_off = 1

        # Context frames
        for i in range(n_ctx):
            t_list.append(torch.full((N_p,), t_off + i, dtype=torch.long, device=device))
            s_list.append(torch.arange(N_p, dtype=torch.long, device=device))

        # Actions: s=0
        act_t = t_off + n_ctx
        if n_actions > 0:
            t_list.append(torch.arange(act_t, act_t + n_actions,
                                       dtype=torch.long, device=device))
            s_list.append(torch.zeros(n_actions, dtype=torch.long, device=device))

        # Target frames
        tgt_t = act_t + n_actions
        for k in range(n_tgt_frames):
            t_list.append(torch.full((N_p,), tgt_t + k, dtype=torch.long, device=device))
            s_list.append(torch.arange(N_p, dtype=torch.long, device=device))

        return torch.cat(t_list), torch.cat(s_list)

    # ------------------------------------------------------------------
    # Mask management
    # ------------------------------------------------------------------

    def _ensure_mask(self, n_ctx: int, n_actions: int, n_tgt: int,
                     device: torch.device, has_goal: bool = True) -> BlockMask:
        key = (n_ctx, n_actions, n_tgt, device, has_goal)
        if key not in self._mask_cache:
            N_p = self.cfg.n_patches
            if n_actions == 0 and n_tgt == 0:
                # Frames only ((goal +) ctx)
                T = (1 + n_ctx) if has_goal else n_ctx
                self._mask_cache[key] = make_frames_only_mask(N_p, T, device)
            else:
                self._mask_cache[key] = make_sequence_mask(
                    N_p, n_ctx, n_actions, n_tgt, device, has_goal=has_goal,
                )
        return self._mask_cache[key]

    def prebuild_mask(self, device: torch.device, n_actions: int = 0,
                      has_goal: bool = True) -> None:
        cfg = self.cfg
        self._ensure_mask(cfg.frames_in, n_actions, cfg.frames_out, device, has_goal)
        self._ensure_mask(cfg.frames_in, 0, 0, device, has_goal)

    # ------------------------------------------------------------------
    # Core helpers
    # ------------------------------------------------------------------

    def _embed_frames(self, frames: torch.Tensor) -> torch.Tensor:
        patches = patchify(frames, self.cfg.patch_size)
        return self.embed_norm(self.patch_embed(patches))

    def _embed_actions(self, actions: torch.Tensor) -> torch.Tensor:
        return self.action_norm(self.action_embed(actions))

    def _run_transformer(self, tokens: torch.Tensor,
                         t_idx: torch.Tensor, s_idx: torch.Tensor,
                         block_mask: BlockMask) -> torch.Tensor:
        x = tokens
        for block in self.blocks:
            x = block(x, t_idx, s_idx, block_mask)
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
        actions:       Optional[torch.Tensor] = None,
        goal:          Optional[torch.Tensor] = None,
    ):
        """
        Parameters
        ----------
        input_frames  : (B, frames_in,  C, H, W)
        target_frames : (B, frames_out, C, H, W)   optional
        actions       : (B, m, action_dim)          optional
        goal          : (B, C, H, W)                optional

        Returns
        -------
        pred_frames   : (B, frames_out, C, H, W)
        pred_actions  : (B, m, action_dim) or None
        loss          : scalar or None
        """
        cfg = self.cfg
        N_p = cfg.n_patches

        if target_frames is not None:
            # ── Teacher-forcing ────────────────────────────────────────
            fin  = cfg.frames_in
            fout = cfg.frames_out
            m    = actions.shape[1]
            has_goal = goal is not None

            # 1) Embed frames: [(goal) | ctx | tgt]
            if has_goal:
                all_frames = torch.cat([goal.unsqueeze(1), input_frames, target_frames], dim=1)
                prefix_len = (1 + fin) * N_p
            else:
                all_frames = torch.cat([input_frames, target_frames], dim=1)
                prefix_len = fin * N_p

            frame_tokens  = self._embed_frames(all_frames)
            action_tokens = self._embed_actions(actions)

            # Assemble: [(goal_patches) | ctx_patches | action_tokens | tgt_patches]
            tokens = torch.cat([
                frame_tokens[:, :prefix_len],     # (goal +) ctx
                action_tokens,                     # actions
                frame_tokens[:, prefix_len:],      # tgt
            ], dim=1)

            # Position indices & mask
            t_idx, s_idx = self._build_position_indices(
                fin, m, fout, tokens.device, has_goal=has_goal,
            )
            block_mask = self._ensure_mask(
                fin, m, fout, tokens.device, has_goal=has_goal,
            )

            # 5) Transformer
            hidden = self._run_transformer(tokens, t_idx, s_idx, block_mask)

            # 6) Predict actions — hidden right before each action token
            #    positions: [prefix_len-1 .. prefix_len+m-2]  →  m predictions
            a_start = prefix_len - 1
            pred_actions = self.action_head(hidden[:, a_start : a_start + m])

            # 7) Predict target patches — hidden right before each target patch
            #    positions: [prefix_len+m-1 .. prefix_len+m+fout*N_p-2]
            n_tgt   = fout * N_p
            t_start = prefix_len + m - 1
            pred_patches = self._decode(hidden[:, t_start : t_start + n_tgt])

            # 8) Loss
            tgt_patches  = patchify(target_frames, cfg.patch_size)
            frame_loss   = F.mse_loss(pred_patches, tgt_patches)
            action_loss  = F.mse_loss(pred_actions, actions)
            loss = frame_loss + action_loss

            pred_frames = unpatchify(
                pred_patches.detach(), cfg.patch_size,
                cfg.resolution, cfg.num_channels,
            ).clamp(-1, 1)

            return pred_frames, pred_actions.detach(), loss

        else:
            # ── Inference (context + optional goal, no actions) ────────
            has_goal = goal is not None
            if has_goal:
                all_frames = torch.cat([goal.unsqueeze(1), input_frames], dim=1)
            else:
                all_frames = input_frames
            n_ctx  = input_frames.shape[1]
            tokens = self._embed_frames(all_frames)
            t_idx, s_idx = self._build_position_indices(
                n_ctx, 0, 0, tokens.device, has_goal=has_goal,
            )
            block_mask = self._ensure_mask(
                n_ctx, 0, 0, tokens.device, has_goal=has_goal,
            )
            hidden = self._run_transformer(tokens, t_idx, s_idx, block_mask)
            pred_patches = self._decode(hidden[:, -N_p:, :])
            pred_frames  = unpatchify(
                pred_patches, cfg.patch_size, cfg.resolution, cfg.num_channels,
            ).clamp(-1, 1)
            return pred_frames, None, None

    # ------------------------------------------------------------------
    # AR Generation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        context_frames: torch.Tensor,
        goal: Optional[torch.Tensor] = None,
        n_actions: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate actions then target frames autoregressively.

        Returns (gen_frames, gen_actions).
        Each step does a full forward pass (no KV-cache).
        """
        cfg = self.cfg
        N_p = cfg.n_patches
        fin = cfg.frames_in
        has_goal = goal is not None

        # Initial: [(goal) | ctx]
        ctx = context_frames[:, -fin:]
        if has_goal:
            tokens = self._embed_frames(torch.cat([goal.unsqueeze(1), ctx], dim=1))
        else:
            tokens = self._embed_frames(ctx)

        gen_actions = []
        n_gen_act = 0

        # Phase 1: generate action tokens
        for _ in range(n_actions):
            t_idx, s_idx = self._build_position_indices(
                fin, n_gen_act, 0, tokens.device, has_goal=has_goal,
            )
            mask = self._ensure_mask(fin, n_gen_act, 0, tokens.device, has_goal=has_goal)
            hidden = self._run_transformer(tokens, t_idx, s_idx, mask)
            pred_a = self.action_head(hidden[:, -1:])
            gen_actions.append(pred_a)
            tokens = torch.cat([tokens, self._embed_actions(pred_a)], dim=1)
            n_gen_act += 1

        gen_actions = torch.cat(gen_actions, dim=1) if gen_actions else \
            tokens.new_zeros(tokens.shape[0], 0, cfg.action_dim)

        # Phase 2: generate target frames one frame at a time
        gen_patches = []
        for k in range(cfg.frames_out):
            n_tgt_so_far = k
            t_idx, s_idx = self._build_position_indices(
                fin, n_actions, n_tgt_so_far, tokens.device, has_goal=has_goal,
            )
            mask = self._ensure_mask(
                fin, n_actions, n_tgt_so_far, tokens.device, has_goal=has_goal,
            )
            hidden = self._run_transformer(tokens, t_idx, s_idx, mask)

            pred_p = self._decode(hidden[:, -N_p:])
            gen_patches.append(pred_p)

            tokens = torch.cat([
                tokens, self.embed_norm(self.patch_embed(pred_p))
            ], dim=1)

        gen_patches = torch.cat(gen_patches, dim=1)
        gen_frames  = unpatchify(
            gen_patches, cfg.patch_size, cfg.resolution, cfg.num_channels,
        ).clamp(-1, 1)

        return gen_frames, gen_actions


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cfg = ARPatchConfig(
        resolution=64, num_channels=3, patch_size=8,
        d_model=512, n_heads=8, n_layers=8,
        frames_in=4, frames_out=4,
        action_dim=7, max_actions=6,
        qk_norm=True, parallel_attn=False,
    )

    device = torch.device("cuda")
    dtype  = torch.bfloat16

    model = ARVideoPatchTransformer(cfg).to(device=device, dtype=dtype)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Patches/frame  : {cfg.n_patches}")
    print(f"Parameters     : {n_params:.2f} M")
    print(f"Attention      : FlexAttention block-causal (mixed sequence)")

    B, m = 2, 6
    ctx     = torch.randn(B, cfg.frames_in,  3, cfg.resolution, cfg.resolution,
                           device=device, dtype=dtype)
    target  = torch.randn(B, cfg.frames_out, 3, cfg.resolution, cfg.resolution,
                           device=device, dtype=dtype)
    actions = torch.randn(B, m, cfg.action_dim, device=device, dtype=dtype)
    goal_f  = torch.randn(B, 3, cfg.resolution, cfg.resolution,
                           device=device, dtype=dtype)

    model.prebuild_mask(device, n_actions=m)

    # Training
    pred, pred_a, loss = model(ctx, target, actions, goal_f)
    print(f"\npred shape     : {pred.shape}")
    print(f"pred_actions   : {pred_a.shape}")
    print(f"train loss     : {loss.item():.4f}")

    # Generation
    gen_f, gen_a = model.generate(ctx, goal_f, n_actions=m)
    print(f"generated      : {gen_f.shape}")
    print(f"gen actions    : {gen_a.shape}")
