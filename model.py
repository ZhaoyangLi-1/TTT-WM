"""
AR Video Patch Transformer with KV-cache  (Emu3-style)
========================================================
Predicts both **target frames** and **intermediate actions** via teacher forcing.

Sequence layout (training)
---------------------------
    [goal_patches | ctx_patches | action_tokens | tgt_patches]
     N_p tokens    fin×N_p       m tokens        fout×N_p

Attention: strictly causal (flash_attn causal=True)

Interface
----------
    Training:   pred_frames, pred_actions, loss = model(ctx, tgt, actions, goal)
    Generation: gen_frames, gen_actions = model.generate(ctx, goal, n_actions)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from flash_attn import flash_attn_func, flash_attn_with_kvcache

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
    action_dim: int = 7

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
    """Learned 2D position embedding: frame_id + spatial_id.

    Accepts explicit (frame_ids, spatial_ids) tensors for mixed sequences.
    Spatial index ``n_patches`` is reserved for action tokens.
    """

    def __init__(self, cfg: ARPatchConfig, max_frames: int = 256):
        super().__init__()
        self.frame_emb = nn.Embedding(max_frames, cfg.d_model // 2)
        # +1 spatial slot for action tokens (index = n_patches)
        self.spatial_emb = nn.Embedding(cfg.n_patches + 1, cfg.d_model // 2)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.cfg = cfg

    def forward(self, frame_ids: torch.Tensor,
                spatial_ids: torch.Tensor) -> torch.Tensor:
        """frame_ids, spatial_ids: (L,) long tensors."""
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
    Multi-head causal self-attention using flash_attn 2.8.

    flash_attn tensor layout : (B, S, n_heads, head_dim)  ← note: NOT (B, n_heads, S, head_dim)

    Two call paths:
      Prefill  (kv_cache=None, L>1) : flash_attn_func        — causal=True
      Decode   (kv_cache given, L=1): flash_attn_with_kvcache — updates cache in-place

    KV-cache layout change vs SDPA version:
      SDPA stored  : (B, n_heads, S, head_dim)
      flash_attn   : (B, S, n_heads, head_dim)
    The cache stored in generate() is in flash_attn layout throughout.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads  = n_heads
        self.head_dim = d_model // n_heads
        self.d_model  = d_model
        self.dropout  = dropout

        self.qkv      = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor,
                kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        x        : (B, L, D)
        kv_cache : None  or  (k_cache, v_cache) each (B, S_prev, n_heads, head_dim)
        Returns  : (B, L, D), new_cache
        """

        B, L, D = x.shape
        # Reshape to flash_attn layout: (B, L, 3, n_heads, head_dim)
        qkv = self.qkv(x).reshape(B, L, 3, self.n_heads, self.head_dim)
        q = qkv[:, :, 0]  # (B, L, n_heads, head_dim)
        k = qkv[:, :, 1]
        v = qkv[:, :, 2]

        attn_dropout = self.dropout if self.training else 0.0

        if kv_cache is None:
            # ── Prefill: process full sequence with causal mask ──────────
            out = flash_attn_func(
                q, k, v,
                dropout_p = attn_dropout,
                causal    = True,
            )
            new_cache = (k, v)
        else:
            # ── Decode: single new token, update cache in-place ──────────
            k_cache, v_cache = kv_cache          # (B, S_prev, n_heads, head_dim)
            S_prev = k_cache.shape[1]

            new_k_cache = torch.cat([k_cache, k], dim=1)
            new_v_cache = torch.cat([v_cache, v], dim=1)

            cache_seqlens = torch.full(
                (B,), S_prev, device=x.device, dtype=torch.int32
            )

            out = flash_attn_with_kvcache(
                q,
                new_k_cache,
                new_v_cache,
                k             = k,
                v             = v,
                cache_seqlens = cache_seqlens,
                causal        = False,   # single query, no causal mask needed
            )
            new_cache = (new_k_cache, new_v_cache)

        out = out.reshape(B, L, D)
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
    Stack of transformer blocks with optional gradient checkpointing.

    Gradient checkpointing is active when:
    use_checkpoint=True AND self.training AND kv_caches is None.
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
        pred_frames, pred_actions, loss = model(ctx, tgt, actions, goal)

    Inference — token-level AR with KV-cache:
        gen_frames, gen_actions = model.generate(ctx, goal, n_actions)
    """

    def __init__(self, cfg: ARPatchConfig):
        super().__init__()
        self.cfg = cfg

        # Frame embedding
        self.patch_embed = nn.Linear(cfg.patch_dim, cfg.d_model)
        self.pos_emb = PatchPositionEmbedding(cfg)
        self.norm_in = nn.LayerNorm(cfg.d_model)

        # Transformer
        self.transformer = CausalTransformer(cfg)

        # Frame output head
        self.norm_out = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.patch_dim)

        # Action embedding & head
        self.action_embed = nn.Linear(cfg.action_dim, cfg.d_model)
        self.action_norm = nn.LayerNorm(cfg.d_model)
        self.action_head = nn.Linear(cfg.d_model, cfg.action_dim)

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
    # Position indices
    # ------------------------------------------------------------------

    def _build_position_indices(
        self, n_ctx: int, n_actions: int, n_tgt_frames: int,
        device: torch.device, has_goal: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build (frame_ids, spatial_ids) for [(goal) | ctx | actions | tgt]."""
        N_p = self.cfg.n_patches
        ACT_SPATIAL = N_p                   # special spatial slot for actions
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

        # Action tokens
        act_t = t_off + n_ctx
        if n_actions > 0:
            t_list.append(torch.arange(act_t, act_t + n_actions,
                                       dtype=torch.long, device=device))
            s_list.append(torch.full((n_actions,), ACT_SPATIAL,
                                     dtype=torch.long, device=device))

        # Target frames
        tgt_t = act_t + n_actions
        for k in range(n_tgt_frames):
            t_list.append(torch.full((N_p,), tgt_t + k, dtype=torch.long, device=device))
            s_list.append(torch.arange(N_p, dtype=torch.long, device=device))

        return torch.cat(t_list), torch.cat(s_list)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _embed_frames(self, frames: torch.Tensor) -> torch.Tensor:
        """Patchify + linear embed. Returns (B, T*N_p, d_model)."""
        patches = patchify(frames, self.cfg.patch_size)
        return self.patch_embed(patches)

    def _embed_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """Embed action vectors. Returns (B, m, d_model)."""
        return self.action_norm(self.action_embed(actions))

    def _embed_patch_single(self, patch: torch.Tensor,
                            frame_id: int, spatial_id: int,
                            device: torch.device) -> torch.Tensor:
        """Embed a single predicted patch for AR generation."""
        token = self.patch_embed(patch)
        pos = self.pos_emb.forward_single(frame_id, spatial_id, device)
        return self.norm_in(token + pos)

    def _embed_action_single(self, action: torch.Tensor,
                             frame_id: int,
                             device: torch.device) -> torch.Tensor:
        """Embed a single predicted action for AR generation."""
        ACT_SPATIAL = self.cfg.n_patches
        token = self.action_norm(self.action_embed(action))
        pos = self.pos_emb.forward_single(frame_id, ACT_SPATIAL, device)
        return self.norm_in(token + pos)

    def _decode_patches(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.head(self.norm_out(tokens))

    def _decode_actions(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.action_head(self.norm_out(tokens))

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

            # 1) Embed all frames: [(goal) | ctx | tgt]
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

            # Position embeddings
            t_idx, s_idx = self._build_position_indices(
                fin, m, fout, tokens.device, has_goal=has_goal,
            )
            tokens = self.norm_in(tokens + self.pos_emb(t_idx, s_idx))

            # 5) Transformer
            tokens_out, _ = self.transformer(tokens, use_checkpoint=True)

            # 6) Predict actions — position i predicts token i+1
            #    positions: [prefix_len-1 .. prefix_len+m-2]  →  m predictions
            a_start = prefix_len - 1
            pred_actions = self._decode_actions(tokens_out[:, a_start : a_start + m])

            # 7) Predict target patches
            #    positions: [prefix_len+m-1 .. prefix_len+m+fout*N_p-2]
            n_tgt   = fout * N_p
            t_start = prefix_len + m - 1
            pred_patches = self._decode_patches(tokens_out[:, t_start : t_start + n_tgt])

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
            n_ctx = input_frames.shape[1]

            tokens = self._embed_frames(all_frames)
            t_idx, s_idx = self._build_position_indices(
                n_ctx, 0, 0, tokens.device, has_goal=has_goal,
            )
            tokens = self.norm_in(tokens + self.pos_emb(t_idx, s_idx))
            tokens_out, _ = self.transformer(tokens, use_checkpoint=False)
            pred_patches = self._decode_patches(tokens_out[:, -N_p:])
            pred_frames = unpatchify(
                pred_patches, cfg.patch_size,
                cfg.resolution, cfg.num_channels,
            ).clamp(-1, 1)
            return pred_frames, None, None

    # ------------------------------------------------------------------
    # AR Generation with KV-cache
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        context_frames: torch.Tensor,
        goal: Optional[torch.Tensor] = None,
        n_actions: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate actions then target frames autoregressively with KV-cache.

        Returns (gen_frames, gen_actions).
        """
        cfg = self.cfg
        N_p = cfg.n_patches
        fin = cfg.frames_in
        device = context_frames.device
        has_goal = goal is not None
        t_base = 1 if has_goal else 0  # temporal offset for ctx

        # ── Phase 0: Prefill [(goal) | ctx] ────────────────────────────
        ctx = context_frames[:, -fin:]
        if has_goal:
            all_frames = torch.cat([goal.unsqueeze(1), ctx], dim=1)
        else:
            all_frames = ctx
        tokens = self._embed_frames(all_frames)
        t_idx, s_idx = self._build_position_indices(fin, 0, 0, device, has_goal=has_goal)
        tokens = self.norm_in(tokens + self.pos_emb(t_idx, s_idx))
        hidden, kv_caches = self.transformer(tokens, use_checkpoint=False)

        # ── Phase 1: Generate actions one by one ───────────────────────
        gen_actions = []
        for i in range(n_actions):
            pred_a = self._decode_actions(hidden[:, -1:])
            gen_actions.append(pred_a)
            frame_id = t_base + fin + i
            token = self._embed_action_single(pred_a, frame_id, device)
            hidden, kv_caches = self.transformer(token, kv_caches, use_checkpoint=False)

        gen_actions = torch.cat(gen_actions, dim=1) if gen_actions else \
            tokens.new_zeros(tokens.shape[0], 0, cfg.action_dim)

        # ── Phase 2: Generate target frame patches one by one ──────────
        gen_patches = []
        t_off = t_base + fin + n_actions

        for k in range(cfg.frames_out):
            for s in range(N_p):
                pred_patch = self._decode_patches(hidden[:, -1:])
                gen_patches.append(pred_patch)
                if not (k == cfg.frames_out - 1 and s == N_p - 1):
                    token = self._embed_patch_single(pred_patch, t_off + k, s, device)
                    hidden, kv_caches = self.transformer(
                        token, kv_caches, use_checkpoint=False,
                    )

        all_patches = torch.cat(gen_patches, dim=1)
        gen_frames = unpatchify(
            all_patches, cfg.patch_size,
            cfg.resolution, cfg.num_channels,
        ).clamp(-1, 1)

        return gen_frames, gen_actions


if __name__ == "__main__":
    device = torch.device("cuda")
    dtype  = torch.bfloat16

    cfg = ARPatchConfig(
        resolution=64, num_channels=3, patch_size=8,
        d_model=256, n_heads=4, n_layers=4,
        frames_in=2, frames_out=3, action_dim=7,
    )
    model = ARVideoPatchTransformer(cfg).to(device=device, dtype=dtype)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Parameters: {n_params:.2f} M")

    B, m = 2, 6
    ctx     = torch.randn(B, cfg.frames_in,  3, 64, 64, device=device, dtype=dtype)
    tgt     = torch.randn(B, cfg.frames_out, 3, 64, 64, device=device, dtype=dtype)
    actions = torch.randn(B, m, cfg.action_dim, device=device, dtype=dtype)
    goal_f  = torch.randn(B, 3, 64, 64, device=device, dtype=dtype)

    # Training
    model.train()
    pred, pred_a, loss = model(ctx, tgt, actions, goal_f)
    loss.backward()
    print(f"train loss={loss.item():.4f}  pred={pred.shape}  pred_actions={pred_a.shape}")

    # Generation
    model.eval()
    gen_f, gen_a = model.generate(ctx, goal_f, n_actions=m)
    print(f"generated={gen_f.shape}  gen_actions={gen_a.shape}")
    print("model.py OK")
