"""
Stage 2 inverse-dynamics models for TTT-WM.

This file keeps the action-prediction heads separate from the Stage 1
video-prediction backbone in `cosmos_model.py`.
"""

from __future__ import annotations

from contextlib import nullcontext
import importlib.util
import os
import sys
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from cosmos_model import ARVideoPatchTransformer


class InverseDynamicsModel(nn.Module):
    """
    Stage 2 — Inverse Dynamics Model.

    Uses a frozen Stage 1 video predictor to generate predicted frames,
    then predicts intermediate actions from (input_frames, predicted_frames).

    Forward flow
    -------------
        1. Frozen Stage 1: input_frames → predicted_frames
        2. Encode [input | predicted] through Stage 1 backbone
        3. Mean-pool per-frame hidden states, concatenate
        4. Trainable MLP head: (2 × d_model) → (n_actions × action_dim)

    Interface
    ----------
        Training:   pred_frames, pred_actions, loss = model(input, target, actions)
                    (target_frames is ignored; Stage 1 predictions used instead)
        Inference:  pred_frames, pred_actions = model.generate(input)
    """

    def __init__(
        self,
        stage1_model: ARVideoPatchTransformer,
        n_actions: int,
        freeze_backbone: bool = True,
    ):
        super().__init__()
        cfg = stage1_model.cfg
        self.cfg = cfg
        self.n_actions = n_actions
        self.freeze_backbone = freeze_backbone

        self.stage1 = stage1_model
        if freeze_backbone:
            for param in self.stage1.parameters():
                param.requires_grad_(False)
            self.stage1.eval()

        d_model = cfg.d_model
        self.action_head = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, n_actions * cfg.action_dim),
        )
        self._init_action_head()

    def _init_action_head(self) -> None:
        for module in self.action_head.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_backbone:
            self.stage1.eval()
        return self

    def prebuild_mask(self, device: torch.device, has_goal: bool = False) -> None:
        del has_goal
        self.stage1._ensure_mask(
            self.cfg.frames_in, 0, device, has_goal=False,
        )
        self.stage1._ensure_mask(2, 0, device, has_goal=False)

    def forward(
        self,
        input_frames: torch.Tensor,
        target_frames: Optional[torch.Tensor] = None,
        actions: Optional[torch.Tensor] = None,
        goal: Optional[torch.Tensor] = None,
    ):
        del target_frames, goal

        batch_size = input_frames.shape[0]
        n_patches = self.cfg.n_patches
        backbone_ctx = torch.no_grad() if self.freeze_backbone else nullcontext()

        with backbone_ctx:
            pred_frames, _ = self.stage1(input_frames)
            all_frames = torch.cat([input_frames, pred_frames], dim=1)

            tokens = self.stage1._embed_frames(all_frames)
            t_idx, s_idx = self.stage1._build_position_indices(
                2, 0, tokens.device, has_goal=False,
            )
            block_mask = self.stage1._ensure_mask(
                2, 0, tokens.device, has_goal=False,
            )
            hidden = self.stage1._run_transformer(tokens, t_idx, s_idx, block_mask)

        input_feat = hidden[:, :n_patches].mean(dim=1)
        pred_feat = hidden[:, n_patches : 2 * n_patches].mean(dim=1)
        combined = torch.cat([input_feat, pred_feat], dim=-1)

        pred_actions = self.action_head(combined).reshape(
            batch_size, self.n_actions, self.cfg.action_dim,
        )

        loss = None
        if actions is not None:
            loss = F.mse_loss(pred_actions, actions)

        return pred_frames, pred_actions, loss

    @torch.no_grad()
    def generate(
        self,
        input_frames: torch.Tensor,
        goal: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        del goal
        pred_frames, pred_actions, _ = self.forward(input_frames, actions=None)
        return pred_frames, pred_actions


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


class InverseDynamicsModelDP(nn.Module):
    """
    Stage 2 — Inverse Dynamics Model.

    Uses a frozen Stage 1 video predictor to generate predicted frames,
    then predicts intermediate actions from (input_frames, predicted_frames).

    Forward flow
    -------------
        1. Frozen Stage 1: input_frames → predicted_frames
        2. Build a two-image observation: [current frame, predicted next frame]
        3. Feed that observation into the original diffusion-policy image architecture
        4. Predict the intermediate action sequence with the original DP U-Net

    Interface
    ----------
        Training:   pred_frames, pred_actions, loss = model(input, target, actions)
                    (target_frames is ignored; Stage 1 predictions used instead)
        Inference:  pred_frames, pred_actions = model.generate(input)
    """

    def __init__(
        self,
        stage1_model: ARVideoPatchTransformer,
        n_actions: int,
        freeze_backbone: bool = True,
        *,
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
            from diffusion_policy.common.normalize_util import get_image_range_normalizer
            from diffusion_policy.model.common.normalizer import (
                LinearNormalizer,
                SingleFieldLinearNormalizer,
            )
            from diffusion_policy.policy.diffusion_unet_hybrid_image_policy import (
                DiffusionUnetHybridImagePolicy,
            )
        except ImportError as exc:
            raise ImportError(
                "InverseDynamicsModelDP requires `diffusers` and `diffusion_policy`. "
                "Install the external diffusion_policy package and expose its source via "
                "`PYTHONPATH` or `DIFFUSION_POLICY_SRC`."
            ) from exc

        cfg = stage1_model.cfg
        self.cfg = cfg
        self.n_actions = int(n_actions)
        self.freeze_backbone = freeze_backbone

        self.stage1 = stage1_model
        if freeze_backbone:
            for param in self.stage1.parameters():
                param.requires_grad_(False)
            self.stage1.eval()

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
        self.policy = DiffusionUnetHybridImagePolicy(
            shape_meta=shape_meta,
            noise_scheduler=noise_scheduler,
            horizon=self.n_actions,
            n_action_steps=self.n_actions,
            n_obs_steps=1,
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

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_backbone:
            self.stage1.eval()
        return self

    def prebuild_mask(self, device: torch.device, has_goal: bool = False) -> None:
        del has_goal
        self.stage1._ensure_mask(
            self.cfg.frames_in, 0, device, has_goal=False,
        )

    def set_action_stats(
        self, action_stats: dict[str, np.ndarray | torch.Tensor]
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

    @staticmethod
    def _frames_to_policy_range(frames: torch.Tensor) -> torch.Tensor:
        return frames.clamp(-1.0, 1.0).add(1.0).mul(0.5)

    def _build_obs_dict(
        self,
        input_frames: torch.Tensor,
        pred_frames: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        current_frame = self._frames_to_policy_range(input_frames[:, -1:])
        predicted_frame = self._frames_to_policy_range(pred_frames[:, :1])
        return {
            "image": current_frame,
            "predicted_image": predicted_frame,
        }

    def _predict_next_frame(
        self,
        input_frames: torch.Tensor,
    ) -> torch.Tensor:
        backbone_ctx = torch.no_grad() if self.freeze_backbone else nullcontext()
        with backbone_ctx:
            pred_frames, _ = self.stage1(input_frames)
        return pred_frames

    def forward(
        self,
        input_frames: torch.Tensor,
        target_frames: Optional[torch.Tensor] = None,
        actions: Optional[torch.Tensor] = None,
        goal: Optional[torch.Tensor] = None,
    ):
        del target_frames, goal

        pred_frames = self._predict_next_frame(input_frames)
        obs_dict = self._build_obs_dict(input_frames, pred_frames)

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

        return pred_frames, pred_actions, loss

    @torch.no_grad()
    def generate(
        self,
        input_frames: torch.Tensor,
        goal: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        del goal
        pred_frames = self._predict_next_frame(input_frames)
        obs_dict = self._build_obs_dict(input_frames, pred_frames)
        pred_actions = self.policy.predict_action(obs_dict)["action"]
        return pred_frames, pred_actions
