"""
Pure inverse-dynamics models for TTT-WM.

These variants do not use a Stage-1 world model to synthesize next frames.
Instead, they consume the ground-truth next frame directly and predict the
intermediate actions between the current and next observations.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from cosmos_model import (
    ARPatchConfig,
    CosmosBlock,
    RMSNorm,
    patchify,
)
from idm_model import _configure_diffusion_policy_import_path


def _select_frame_pair(
    input_frames: torch.Tensor,
    target_frames: Optional[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
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


class PureInverseDynamicsModel(nn.Module):
    """
    Predict intermediate actions from ``(current_frame, gt_next_frame)``.

    The action head matches ``idm_model.py``:
        (2 * d_model) -> d_model -> (n_actions * action_dim)
    """

    def __init__(self, cfg: ARPatchConfig, n_actions: int):
        super().__init__()
        self.cfg = cfg
        self.n_actions = int(n_actions)

        self.patch_embed = nn.Linear(cfg.patch_dim, cfg.d_model, bias=False)
        self.embed_norm = RMSNorm(cfg.d_model)
        self.blocks = nn.ModuleList([CosmosBlock(cfg) for _ in range(cfg.n_layers)])
        self.out_norm = RMSNorm(cfg.d_model)

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
            elif isinstance(module, RMSNorm):
                nn.init.ones_(module.scale)

        scale = (2 * self.cfg.n_layers) ** -0.5
        for name, param in self.named_parameters():
            if "out.weight" in name or "down_proj.weight" in name:
                param.data.mul_(scale)

    def _embed_frames(self, frames: torch.Tensor) -> torch.Tensor:
        patches = patchify(frames, self.cfg.patch_size)
        return self.embed_norm(self.patch_embed(patches))

    def _run_transformer(
        self,
        tokens: torch.Tensor,
        t_idx: torch.Tensor,
        s_idx: torch.Tensor,
    ) -> torch.Tensor:
        x = tokens
        for block in self.blocks:
            x = block(x, t_idx, s_idx)
        return self.out_norm(x)

    def _build_position_indices(
        self,
        n_frames: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        n_patches = self.cfg.n_patches
        t_idx = torch.repeat_interleave(
            torch.arange(n_frames, device=device, dtype=torch.long),
            n_patches,
        )
        s_idx = torch.arange(
            n_patches, device=device, dtype=torch.long
        ).repeat(n_frames)
        return t_idx, s_idx

    def prebuild_mask(self, device: torch.device, has_goal: bool = False) -> None:
        """No-op. flash_attn applies causal masking at the kernel level."""
        del device, has_goal

    def forward(
        self,
        input_frames: torch.Tensor,
        target_frames: Optional[torch.Tensor] = None,
        actions: Optional[torch.Tensor] = None,
        goal: Optional[torch.Tensor] = None,
    ):
        del goal

        current_frame, next_frame = _select_frame_pair(input_frames, target_frames)
        frame_pair = torch.cat([current_frame, next_frame], dim=1)

        tokens = self._embed_frames(frame_pair)
        t_idx, s_idx = self._build_position_indices(2, tokens.device)
        hidden = self._run_transformer(tokens, t_idx, s_idx)

        n_patches = self.cfg.n_patches
        current_feat = hidden[:, :n_patches].mean(dim=1)
        next_feat = hidden[:, n_patches : 2 * n_patches].mean(dim=1)
        combined = torch.cat([current_feat, next_feat], dim=-1)

        pred_actions = self.action_head(combined).reshape(
            frame_pair.shape[0], self.n_actions, self.cfg.action_dim
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
        pred_frames, pred_actions, _ = self.forward(
            input_frames,
            target_frames=target_frames,
            actions=None,
            goal=goal,
        )
        return pred_frames, pred_actions


class PureInverseDynamicsModelDP(nn.Module):
    """
    Diffusion-policy inverse dynamics from ``(current_frame, gt_next_frame)``.

    This mirrors ``InverseDynamicsModelDP`` from ``idm_model.py``, except the
    second observation is the ground-truth next frame instead of a Stage-1
    prediction.
    """

    def __init__(
        self,
        cfg: ARPatchConfig,
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
        self._linear_normalizer_cls = LinearNormalizer
        self._single_field_normalizer_cls = SingleFieldLinearNormalizer
        self._image_normalizer_factory = get_image_range_normalizer
        for name, param in self.policy.named_parameters():
            if name.endswith("_dummy_variable"):
                param.requires_grad_(False)

    def prebuild_mask(self, device: torch.device, has_goal: bool = False) -> None:
        del device, has_goal

    def set_action_stats(
        self,
        action_stats: dict[str, np.ndarray | torch.Tensor],
    ) -> None:
        action_min = torch.as_tensor(action_stats["min"], dtype=torch.float32)
        action_max = torch.as_tensor(action_stats["max"], dtype=torch.float32)
        action_mean = torch.as_tensor(action_stats["mean"], dtype=torch.float32)
        action_std = torch.as_tensor(action_stats["std"], dtype=torch.float32)
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
                "min": action_min.cpu().numpy(),
                "max": action_max.cpu().numpy(),
                "mean": action_mean.cpu().numpy(),
                "std": action_std.cpu().numpy(),
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
            "image": self._frames_to_policy_range(current_frame),
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
