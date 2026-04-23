"""
Pure inverse-dynamics models for TTT-WM.

Fully self-contained: no dependency on the Stage 1 world model
(``cosmos_model``). Each variant consumes the ground-truth
``(current_frame, next_frame)`` pair directly and predicts the
intermediate actions between the two observations.

Two flavors:

* ``PureInverseDynamicsModel``
    Bidirectional transformer encoder over the 2-frame patch sequence,
    mean-pool per frame, concat, then an MLP action head.

* ``PureInverseDynamicsModelDP``
    Uses the external ``diffusion_policy`` image architecture directly on
    the (current, next) image pair. No transformer encoder inside this
    file; the DP package handles its own feature extraction.

Because pure IDM is a pure encoder (no AR generation, no teacher-forced
shift-by-1 loss), attention is **bidirectional** (``causal=False``):
every patch in the 2-frame sequence can attend to every other patch.
There is no "target leakage" risk — the model does not predict frames,
only actions, and ground-truth actions cannot be echoed through frame
attention.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from dataclasses import dataclass
from typing import Any, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from flash_attn import flash_attn_func


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class PureIDMConfig:
    """Minimum config fields needed by the pure IDM encoder.

    Duck-type compatible with the Stage-1 ``ARPatchConfig`` so the
    existing trainer (which builds an ``ARPatchConfig``) can pass that
    object in unchanged.
    """
    # image
    resolution:   int = 128
    num_channels: int = 3
    patch_size:   int = 8

    # transformer
    d_model:   int   = 512
    n_heads:   int   = 8
    n_layers:  int   = 8
    mlp_ratio: float = 8 / 3
    dropout:   float = 0.0
    qk_norm:   bool  = True

    # actions
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


# ---------------------------------------------------------------------------
# Patchify
# ---------------------------------------------------------------------------

def _patchify(frames: torch.Tensor, patch_size: int) -> torch.Tensor:
    """(B, T, C, H, W) -> (B, T*N_p, C*P*P)."""
    B, T, C, H, W = frames.shape
    P = patch_size
    h, w = H // P, W // P
    x = frames.reshape(B * T, C, h, P, w, P)
    x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
    return x.reshape(B, T * h * w, C * P * P)


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------

class _RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps   = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        y = (x.float() * rms) * self.scale.float()
        return y.to(x.dtype)


# ---------------------------------------------------------------------------
# Rotary Position Embedding (2-D factored: temporal + spatial)
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


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(start_dim=-2)


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    cos = cos.to(dtype=x.dtype)[None, None, :, :]
    sin = sin.to(dtype=x.dtype)[None, None, :, :]
    return x * cos + _rotate_half(x) * sin


class _RoPE2D(nn.Module):
    """Half of head_dim rotates on temporal index, other half on spatial."""

    def __init__(self, head_dim: int, n_patches: int, max_temporal: int = 4):
        super().__init__()
        if head_dim % 4 != 0:
            raise ValueError(
                f"head_dim must be divisible by 4 for 2-D RoPE, got {head_dim}"
            )
        r = head_dim // 2
        t_cos, t_sin = _build_sin_cos(r, max_temporal)
        s_cos, s_sin = _build_sin_cos(r, n_patches)
        self.register_buffer("t_cos", t_cos, persistent=False)
        self.register_buffer("t_sin", t_sin, persistent=False)
        self.register_buffer("s_cos", s_cos, persistent=False)
        self.register_buffer("s_sin", s_sin, persistent=False)

    def forward(
        self, q: torch.Tensor, k: torch.Tensor,
        t_idx: torch.Tensor, s_idx: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        q_t, q_s = q.chunk(2, dim=-1)
        k_t, k_s = k.chunk(2, dim=-1)
        t_cos = self.t_cos.index_select(0, t_idx)
        t_sin = self.t_sin.index_select(0, t_idx)
        s_cos = self.s_cos.index_select(0, s_idx)
        s_sin = self.s_sin.index_select(0, s_idx)
        q = torch.cat(
            [_apply_rope(q_t, t_cos, t_sin), _apply_rope(q_s, s_cos, s_sin)],
            dim=-1,
        )
        k = torch.cat(
            [_apply_rope(k_t, t_cos, t_sin), _apply_rope(k_s, s_cos, s_sin)],
            dim=-1,
        )
        return q, k


# ---------------------------------------------------------------------------
# SwiGLU MLP
# ---------------------------------------------------------------------------

class _SwiGLUMLP(nn.Module):
    def __init__(self, d_model: int, mlp_ratio: float, dropout: float = 0.0):
        super().__init__()
        inner = int(d_model * mlp_ratio)
        inner = (inner // 2) * 2
        self.gate_up_proj = nn.Linear(d_model, inner,      bias=False)
        self.down_proj    = nn.Linear(inner // 2, d_model, bias=False)
        self.drop         = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, value = self.gate_up_proj(x).chunk(2, dim=-1)
        return self.drop(self.down_proj(F.silu(gate) * value))


# ---------------------------------------------------------------------------
# Bidirectional attention block
# ---------------------------------------------------------------------------

class _BidirectionalAttention(nn.Module):
    """Full bidirectional self-attention via ``flash_attn_func(causal=False)``."""

    def __init__(self, d_model: int, n_heads: int, head_dim: int, n_patches: int,
                 qk_norm: bool = True):
        super().__init__()
        self.n_heads  = n_heads
        self.head_dim = head_dim
        self.d_model  = d_model

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model,     bias=False)

        self.rope   = _RoPE2D(head_dim, n_patches)
        self.q_norm = _RMSNorm(head_dim) if qk_norm else nn.Identity()
        self.k_norm = _RMSNorm(head_dim) if qk_norm else nn.Identity()

    def forward(
        self, x: torch.Tensor,
        t_idx: torch.Tensor, s_idx: torch.Tensor,
    ) -> torch.Tensor:
        B, L, D = x.shape
        qkv = self.qkv(x).reshape(B, L, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = self.q_norm(q.transpose(1, 2))
        k = self.k_norm(k.transpose(1, 2))
        q, k = self.rope(q, k, t_idx, s_idx)
        q = q.transpose(1, 2).to(v.dtype)
        k = k.transpose(1, 2).to(v.dtype)

        out = flash_attn_func(q, k, v, causal=False)  # bidirectional
        return self.out(out.reshape(B, L, D))


class _EncoderBlock(nn.Module):
    def __init__(self, cfg: Any):
        super().__init__()
        self.norm1 = _RMSNorm(cfg.d_model)
        self.attn  = _BidirectionalAttention(
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            head_dim=cfg.head_dim,
            n_patches=cfg.n_patches,
            qk_norm=getattr(cfg, "qk_norm", True),
        )
        self.norm2 = _RMSNorm(cfg.d_model)
        self.mlp   = _SwiGLUMLP(cfg.d_model, cfg.mlp_ratio, cfg.dropout)

    def forward(
        self, x: torch.Tensor,
        t_idx: torch.Tensor, s_idx: torch.Tensor,
    ) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), t_idx, s_idx)
        x = x + self.mlp(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# Diffusion-policy import path (inlined from idm_model.py to avoid pulling
# cosmos_model in transitively).
# ---------------------------------------------------------------------------

def _configure_diffusion_policy_import_path() -> None:
    if importlib.util.find_spec("diffusion_policy") is not None:
        return
    env_keys = ("DIFFUSION_POLICY_SRC", "TTT_WM_DIFFUSION_POLICY_SRC")
    for env_key in env_keys:
        value = os.environ.get(env_key)
        if not value:
            continue
        path = os.path.expanduser(value)
        if os.path.isdir(path) and path not in sys.path:
            sys.path.insert(0, path)
            break


# ---------------------------------------------------------------------------
# Frame-pair selector
# ---------------------------------------------------------------------------

def _select_frame_pair(
    input_frames: torch.Tensor,
    target_frames: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    if target_frames is None:
        raise ValueError(
            "Pure inverse-dynamics models require ground-truth target_frames."
        )
    if input_frames.ndim != 5 or target_frames.ndim != 5:
        raise ValueError(
            "Expected input_frames and target_frames with shape (B, T, C, H, W)."
        )
    if input_frames.shape[1] < 1 or target_frames.shape[1] < 1:
        raise ValueError("Need at least one context frame and one target frame.")
    return input_frames[:, -1:], target_frames[:, :1]


# ---------------------------------------------------------------------------
# MLP-head variant
# ---------------------------------------------------------------------------

class PureInverseDynamicsModel(nn.Module):
    """Predict actions from ``(current_frame, gt_next_frame)`` via a small
    bidirectional transformer encoder + MLP head.

    Action head matches the Stage-2 IDM for easy comparability:
        (2 * d_model) -> d_model -> (n_actions * action_dim)
    """

    def __init__(self, cfg: Any, n_actions: int):
        super().__init__()
        self.cfg = cfg
        self.n_actions = int(n_actions)

        self.patch_embed = nn.Linear(cfg.patch_dim, cfg.d_model, bias=False)
        self.embed_norm  = _RMSNorm(cfg.d_model)
        self.blocks      = nn.ModuleList([_EncoderBlock(cfg) for _ in range(cfg.n_layers)])
        self.out_norm    = _RMSNorm(cfg.d_model)

        self.action_head = nn.Sequential(
            nn.Linear(2 * cfg.d_model, cfg.d_model),
            nn.SiLU(),
            nn.Linear(cfg.d_model, self.n_actions * cfg.action_dim),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, _RMSNorm):
                nn.init.ones_(module.scale)

        scale = (2 * self.cfg.n_layers) ** -0.5
        for name, param in self.named_parameters():
            if "out.weight" in name or "down_proj.weight" in name:
                param.data.mul_(scale)

    def prebuild_mask(self, device: torch.device, has_goal: bool = False) -> None:
        """No-op. flash_attn handles the (trivial bidirectional) mask."""
        del device, has_goal

    def _embed(self, frames: torch.Tensor) -> torch.Tensor:
        return self.embed_norm(self.patch_embed(_patchify(frames, self.cfg.patch_size)))

    def _build_indices(self, n_frames: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        n_patches = self.cfg.n_patches
        t_idx = torch.repeat_interleave(
            torch.arange(n_frames, device=device, dtype=torch.long), n_patches,
        )
        s_idx = torch.arange(n_patches, device=device, dtype=torch.long).repeat(n_frames)
        return t_idx, s_idx

    def forward(
        self,
        input_frames: torch.Tensor,
        target_frames: Optional[torch.Tensor] = None,
        actions: Optional[torch.Tensor] = None,
        goal: Optional[torch.Tensor] = None,
    ):
        del goal  # pure IDM is not goal-conditioned

        current_frame, next_frame = _select_frame_pair(input_frames, target_frames)
        frame_pair = torch.cat([current_frame, next_frame], dim=1)

        tokens = self._embed(frame_pair)
        t_idx, s_idx = self._build_indices(2, tokens.device)

        x = tokens
        for block in self.blocks:
            x = block(x, t_idx, s_idx)
        hidden = self.out_norm(x)

        n_patches = self.cfg.n_patches
        current_feat = hidden[:, :n_patches].mean(dim=1)
        next_feat    = hidden[:, n_patches : 2 * n_patches].mean(dim=1)
        combined     = torch.cat([current_feat, next_feat], dim=-1)

        pred_actions = self.action_head(combined).reshape(
            frame_pair.shape[0], self.n_actions, self.cfg.action_dim,
        )

        loss = None
        if actions is not None:
            if actions.shape[1] != self.n_actions:
                raise ValueError(
                    f"Expected {self.n_actions} action steps, got {actions.shape[1]}."
                )
            if actions.shape[2] != self.cfg.action_dim:
                raise ValueError(
                    f"Expected action dim {self.cfg.action_dim}, got {actions.shape[2]}."
                )
            loss = F.mse_loss(pred_actions, actions)

        return next_frame, pred_actions, loss

    @torch.no_grad()
    def generate(
        self,
        input_frames: torch.Tensor,
        target_frames: torch.Tensor,
        goal: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        next_frame, pred_actions, _ = self.forward(
            input_frames,
            target_frames=target_frames,
            actions=None,
            goal=goal,
        )
        return next_frame, pred_actions


# ---------------------------------------------------------------------------
# Diffusion-policy head variant
# ---------------------------------------------------------------------------

class PureInverseDynamicsModelDP(nn.Module):
    """Diffusion-policy inverse dynamics from ``(current_frame, gt_next_frame)``.

    Mirrors ``InverseDynamicsModelDP``'s DP head but without a Stage-1 world
    model — the "predicted image" observation is the ground-truth next frame.
    """

    def __init__(
        self,
        cfg: Any,
        n_actions: int,
        *,
        horizon: Optional[int] = None,
        n_action_steps: Optional[int] = None,
        n_obs_steps: int = 1,
        num_train_timesteps: int = 100,
        num_inference_steps: Optional[int] = None,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = "squaredcos_cap_v2",
        variance_type: str = "fixed_small",
        clip_sample: bool = True,
        prediction_type: str = "epsilon",
        diffusion_step_embed_dim: int = 128,
        down_dims: Tuple[int, ...] = (256, 512, 1024),
        kernel_size: int = 5,
        n_groups: int = 8,
        cond_predict_scale: bool = True,
        obs_as_global_cond: bool = True,
        crop_shape: Tuple[int, int] = (84, 84),
        obs_encoder_group_norm: bool = True,
        eval_fixed_crop: bool = True,
    ):
        super().__init__()
        _configure_diffusion_policy_import_path()

        try:
            from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
            from diffusion_policy.common.normalize_util import (
                get_image_range_normalizer,
            )
            from diffusion_policy.model.common.normalizer import (
                LinearNormalizer,
                SingleFieldLinearNormalizer,
            )
            from dp.policy import TTTWMDiffusionPolicy
        except ImportError as exc:
            raise ImportError(
                "PureInverseDynamicsModelDP requires `diffusers` and `diffusion_policy`. "
                "Install the external diffusion_policy package and expose its source via "
                "`PYTHONPATH` or `DIFFUSION_POLICY_SRC`."
            ) from exc

        self.cfg = cfg
        self.n_actions = int(n_actions)
        self.horizon = int(horizon) if horizon is not None else self.n_actions
        self.n_action_steps = (
            int(n_action_steps) if n_action_steps is not None else self.n_actions
        )
        self.n_obs_steps = int(n_obs_steps)

        self.action_dim = cfg.action_dim
        self.obs_keys = ("image", "predicted_image")

        shape_meta = {
            "action": {"shape": [self.action_dim]},
            "obs": {
                "image": {
                    "shape": [cfg.num_channels, cfg.resolution, cfg.resolution],
                    "type": "rgb",
                },
                "predicted_image": {
                    "shape": [cfg.num_channels, cfg.resolution, cfg.resolution],
                    "type": "rgb",
                },
            },
        }
        noise_scheduler = DDPMScheduler(
            num_train_timesteps=num_train_timesteps,
            beta_start=beta_start,
            beta_end=beta_end,
            beta_schedule=beta_schedule,
            variance_type=variance_type,
            clip_sample=clip_sample,
            prediction_type=prediction_type,
        )
        self.policy = TTTWMDiffusionPolicy(
            shape_meta=shape_meta,
            noise_scheduler=noise_scheduler,
            horizon=self.horizon,
            n_action_steps=self.n_action_steps,
            n_obs_steps=self.n_obs_steps,
            num_inference_steps=num_inference_steps,
            obs_as_global_cond=obs_as_global_cond,
            crop_shape=tuple(crop_shape),
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=tuple(down_dims),
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale,
            obs_encoder_group_norm=obs_encoder_group_norm,
            eval_fixed_crop=eval_fixed_crop,
        )
        self.num_inference_steps = (
            int(num_inference_steps)
            if num_inference_steps is not None
            else int(num_train_timesteps)
        )
        self._linear_normalizer_cls       = LinearNormalizer
        self._single_field_normalizer_cls = SingleFieldLinearNormalizer
        self._image_normalizer_factory    = get_image_range_normalizer
        for name, param in self.policy.named_parameters():
            if name.endswith("_dummy_variable"):
                param.requires_grad_(False)

    def prebuild_mask(self, device: torch.device, has_goal: bool = False) -> None:
        del device, has_goal

    def set_action_stats(
        self,
        action_stats: dict[str, np.ndarray | torch.Tensor],
    ) -> None:
        action_min  = torch.as_tensor(action_stats["min"],  dtype=torch.float32)
        action_max  = torch.as_tensor(action_stats["max"],  dtype=torch.float32)
        action_mean = torch.as_tensor(action_stats["mean"], dtype=torch.float32)
        action_std  = torch.as_tensor(action_stats["std"],  dtype=torch.float32)
        if action_min.numel() != self.action_dim or action_max.numel() != self.action_dim:
            raise ValueError(
                f"Action stats must have dim {self.action_dim}, got "
                f"{action_min.numel()} and {action_max.numel()}."
            )

        output_max = 1.0
        output_min = -1.0
        range_eps = 1e-4

        input_range = action_max - action_min
        ignore_dim = input_range < range_eps
        input_range = input_range.clone()
        input_range[ignore_dim] = output_max - output_min

        scale = (output_max - output_min) / input_range
        offset = output_min - scale * action_min
        offset[ignore_dim] = (
            (output_max + output_min) / 2.0 - action_min[ignore_dim]
        )

        normalizer = self._linear_normalizer_cls()
        normalizer["action"] = self._single_field_normalizer_cls.create_manual(
            scale=scale.cpu().numpy(),
            offset=offset.cpu().numpy(),
            input_stats_dict={
                "min":  action_min.cpu().numpy(),
                "max":  action_max.cpu().numpy(),
                "mean": action_mean.cpu().numpy(),
                "std":  action_std.cpu().numpy(),
            },
        )
        for obs_key in self.obs_keys:
            normalizer[obs_key] = self._image_normalizer_factory()
        self.policy.set_normalizer(normalizer)
        policy_device = next(self.policy.parameters()).device
        self.policy.normalizer.to(device=policy_device)

    @staticmethod
    def _frames_to_policy_range(frames: torch.Tensor) -> torch.Tensor:
        return frames.clamp(-1.0, 1.0).add(1.0).mul(0.5)

    def _build_obs_dict(
        self,
        current_frame: torch.Tensor,
        next_frame: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        return {
            "image":           self._frames_to_policy_range(current_frame),
            "predicted_image": self._frames_to_policy_range(next_frame),
        }

    def forward(
        self,
        input_frames: torch.Tensor,
        target_frames: Optional[torch.Tensor] = None,
        actions: Optional[torch.Tensor] = None,
        goal: Optional[torch.Tensor] = None,
    ):
        del goal

        current_frame, next_frame = _select_frame_pair(input_frames, target_frames)
        obs_dict = self._build_obs_dict(current_frame, next_frame)

        loss = None
        pred_actions = None
        if actions is not None:
            if actions.shape[1] != self.n_actions:
                raise ValueError(
                    f"Expected {self.n_actions} actions, got {actions.shape[1]}."
                )
            batch = {"obs": obs_dict, "action": actions}
            loss = self.policy.compute_loss(batch)
        else:
            pred_actions = self.policy.predict_action(obs_dict)["action"]

        return next_frame, pred_actions, loss

    @torch.no_grad()
    def generate(
        self,
        input_frames: torch.Tensor,
        target_frames: torch.Tensor,
        goal: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        pred_frames, pred_actions, _ = self.forward(
            input_frames,
            target_frames=target_frames,
            actions=None,
            goal=goal,
        )
        return pred_frames, pred_actions
