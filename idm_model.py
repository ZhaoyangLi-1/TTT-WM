"""
Stage 2 inverse-dynamics models for TTT-WM.

This file keeps the action-prediction heads separate from the Stage 1
video-prediction backbone in `cosmos_model.py`.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from cosmos_model import ARVideoPatchTransformer


def _validate_stage1_micro_batch(
    stage1_micro_batch: Optional[int],
) -> Optional[int]:
    if stage1_micro_batch is None:
        return None
    value = int(stage1_micro_batch)
    if value <= 0:
        raise ValueError(
            f"stage1_micro_batch must be positive when set, got {value}."
        )
    return value


def _run_stage1_in_chunks(
    stage1: ARVideoPatchTransformer,
    input_frames: torch.Tensor,
    goal: Optional[torch.Tensor],
    micro_batch: int,
) -> torch.Tensor:
    """Run Stage 1 to predict the next frame(s) for every element of the batch.

    Uses `stage1.generate(...)` (patch-level AR with flash_attn kv-cache) to
    match the token-causal training setup. The old code called
    `stage1(input_frames, goal=goal)` which went through the now-removed
    "decode hidden[-N_p:]" else branch — that branch produced garbage under
    token-causal shift-by-1 training because only one ctx position is
    supervised to predict tgt[0].
    """
    if micro_batch >= input_frames.shape[0]:
        return stage1.generate(input_frames, goal=goal)

    pred_chunks = []
    for start in range(0, input_frames.shape[0], micro_batch):
        end = start + micro_batch
        goal_chunk = goal[start:end] if goal is not None else None
        pred_chunks.append(stage1.generate(input_frames[start:end], goal=goal_chunk))
    return torch.cat(pred_chunks, dim=0)


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

    Uses a Stage 1 video predictor to generate predicted frames, then predicts
    intermediate actions from (input_frames, predicted_frames). Stage 1 remains
    trainable during Stage 2 training.

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
        stage1_micro_batch: Optional[int] = None,
        freeze_stage1: bool = False,
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
            from dp.policy import TTTWMDiffusionPolicy
        except ImportError as exc:
            raise ImportError(
                "InverseDynamicsModelDP requires `diffusers` and `diffusion_policy`. "
                "Install the external diffusion_policy package and expose its source via "
                "`PYTHONPATH` or `DIFFUSION_POLICY_SRC`."
            ) from exc

        cfg = stage1_model.cfg
        self.cfg = cfg
        self.n_actions = int(n_actions)
        self.horizon = int(horizon) if horizon is not None else self.n_actions
        self.n_action_steps = (
            int(n_action_steps) if n_action_steps is not None else self.n_actions
        )
        self.n_obs_steps = int(n_obs_steps)
        self.stage1_micro_batch = _validate_stage1_micro_batch(stage1_micro_batch)
        self._auto_stage1_micro_batch_cache: dict[int, int] = {}
        self.freeze_stage1 = bool(freeze_stage1)

        self.stage1 = stage1_model
        if self.freeze_stage1:
            for p in self.stage1.parameters():
                p.requires_grad_(False)
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

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_stage1:
            self.stage1.eval()
        return self

    def prebuild_mask(self, device: torch.device, has_goal: bool = True) -> None:
        """No-op. flash_attn applies causal masking at the kernel level."""
        del device, has_goal

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
        policy_device = next(self.policy.parameters()).device
        self.policy.normalizer.to(device=policy_device)

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
        goal: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        micro_batch = self.stage1_micro_batch
        if micro_batch is not None:
            return _run_stage1_in_chunks(
                self.stage1, input_frames, goal, micro_batch,
            )

        batch_size = int(input_frames.shape[0])
        cached_micro_batch = self._auto_stage1_micro_batch_cache.get(batch_size)
        trial_micro_batch = cached_micro_batch or batch_size

        while True:
            try:
                pred_frames = _run_stage1_in_chunks(
                    self.stage1, input_frames, goal, trial_micro_batch,
                )
                self._auto_stage1_micro_batch_cache[batch_size] = (
                    trial_micro_batch
                )
                return pred_frames
            except torch.OutOfMemoryError:
                if not torch.cuda.is_available() or trial_micro_batch <= 1:
                    raise
                torch.cuda.empty_cache()
                trial_micro_batch = max(1, trial_micro_batch // 2)

    def forward(
        self,
        input_frames: torch.Tensor,
        target_frames: Optional[torch.Tensor] = None,
        actions: Optional[torch.Tensor] = None,
        goal: Optional[torch.Tensor] = None,
        *,
        cached_pred_frames: Optional[torch.Tensor] = None,
    ):
        del target_frames

        if cached_pred_frames is not None:
            # Cache path: pred was prebaked by scripts/prebake_stage1_pred.py
            # using the exact same frozen backbone, so the in-line AR decode
            # would produce the same values — skip it entirely.
            pred_frames = cached_pred_frames
        elif self.freeze_stage1:
            # inference_mode is slightly cheaper than no_grad (skips version
            # counter bookkeeping). pred_frames only flows into out-of-place
            # ops downstream (clamp/add/mul in _build_obs_dict), so the
            # inference-mode flag is safe to propagate.
            with torch.inference_mode():
                pred_frames = self._predict_next_frame(input_frames, goal=goal)
        else:
            pred_frames = self._predict_next_frame(input_frames, goal=goal)
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
        pred_frames = self._predict_next_frame(input_frames, goal=goal)
        obs_dict = self._build_obs_dict(input_frames, pred_frames)
        pred_actions = self.policy.predict_action(obs_dict)["action"]
        return pred_frames, pred_actions
