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
        shape_meta: Optional[dict] = None,
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

        # Single obs layout (no temporal stacking): the n_obs_steps real history
        # frames are exposed as SEPARATE single-timestep rgb keys —
        #   image      = i_k        (most recent)
        #   image_prev1= i_{k-1}
        #   …          = i_{k-(n-1)}
        # plus the single predicted_image (î) produced by the frozen backbone.
        # The DP policy therefore runs with n_obs_steps=1 (every key is T=1);
        # temporal context is carried by the distinct image keys, not the time
        # axis. Proprio history rides along by being flattened across the same
        # n_obs_steps frames into the feature dim (see _build_obs_dict), so it
        # also stays T=1 — this is what lets proprio carry n_obs_steps frames
        # WITHOUT temporal stacking. A single knob, n_obs_steps, drives both the
        # image-key count and the proprio history depth.
        #   n_obs_steps=2 → obs = {image=i_k, image_prev1=i_{k-1},
        #                          predicted_image=î, <proprio>=[p_{k-1};p_k]}
        if self.n_obs_steps < 1:
            raise ValueError(f"n_obs_steps must be >= 1, got {self.n_obs_steps}.")
        self.predicted_image_key = "predicted_image"
        self.history_image_keys: tuple[str, ...] = tuple(
            "image" if i == 0 else f"image_prev{i}"
            for i in range(self.n_obs_steps)
        )
        # The frozen backbone only ever consumes its trained context length
        # (cfg.frames_in); when the dataset hands us extra history frames for
        # the obs, we slice the trailing frames_in before the AR decode.
        self._backbone_fin = int(getattr(cfg, "frames_in", 1))

        # Build shape_meta authoritatively from n_obs_steps. The history +
        # predicted image keys are rebuilt every time (their dims are forced to
        # the Stage-1 backbone resolution so the DP vision encoder agrees with
        # the frames we feed it). Each low_dim (proprio) key the caller declared
        # is expanded from its base dim D to D * n_obs_steps, because its history
        # is flattened into the feature dim; we remember the base dim so
        # _build_obs_dict and set_action_stats can reproduce the expansion.
        image_shape = [cfg.num_channels, cfg.resolution, cfg.resolution]
        passed_obs = dict((shape_meta or {}).get("obs", {}))
        self._proprio_base_dim: dict[str, int] = {}
        proprio_meta: dict = {}
        for k, v in passed_obs.items():
            if v.get("type") != "low_dim":
                continue
            base = int(np.prod([int(s) for s in v["shape"]]))
            self._proprio_base_dim[k] = base
            proprio_meta[k] = {"shape": [base * self.n_obs_steps], "type": "low_dim"}
        obs_meta: dict = {}
        for img_key in (*self.history_image_keys, self.predicted_image_key):
            obs_meta[img_key] = {"shape": list(image_shape), "type": "rgb"}
        obs_meta.update(proprio_meta)
        shape_meta = {
            "action": {"shape": [self.action_dim]},
            "obs": obs_meta,
        }
        self.shape_meta = shape_meta

        obs_meta = shape_meta["obs"]
        self.rgb_keys: tuple[str, ...] = tuple(
            k for k, v in obs_meta.items() if v.get("type", "rgb") == "rgb"
        )
        self.proprio_keys: tuple[str, ...] = tuple(
            k for k, v in obs_meta.items() if v.get("type") == "low_dim"
        )
        # Public alias used elsewhere (e.g. set_action_stats normalizer loop).
        self.obs_keys = self.rgb_keys
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
            # Every obs key is T=1 (image history is separate keys, proprio
            # history is flattened into the feature dim), so the underlying DP
            # always runs with a single observation step. self.n_obs_steps is
            # the user-facing history depth used only when building the obs dict.
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
        self,
        action_stats: dict[str, np.ndarray | torch.Tensor],
        obs_stats: Optional[dict[str, dict[str, np.ndarray | torch.Tensor]]] = None,
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

        normalizer = self._linear_normalizer_cls()
        normalizer["action"] = self._make_range_normalizer(
            action_min, action_max, action_mean, action_std,
        )
        for obs_key in self.rgb_keys:
            normalizer[obs_key] = self._image_normalizer_factory()

        obs_stats = obs_stats or {}
        # Proprio is flattened to n_obs_steps copies along the feature dim, so
        # tile each (base-dim) stat n_obs_steps times to match the (B,1,n*D)
        # obs and the expanded shape_meta dim. Every frame shares the same
        # per-channel stats, so plain repetition is exact regardless of order.
        def _tile_stat(v):
            return (
                torch.as_tensor(v, dtype=torch.float32)
                .reshape(-1)
                .repeat(self.n_obs_steps)
            )
        for obs_key in self.proprio_keys:
            stats = obs_stats.get(obs_key)
            if stats is None:
                raise ValueError(
                    f"Proprio obs key {obs_key!r} declared in shape_meta but "
                    f"obs_stats is missing it (got keys {sorted(obs_stats)})."
                )
            normalizer[obs_key] = self._make_range_normalizer(
                _tile_stat(stats["min"]), _tile_stat(stats["max"]),
                _tile_stat(stats["mean"]), _tile_stat(stats["std"]),
            )
        self.policy.set_normalizer(normalizer)
        policy_device = next(self.policy.parameters()).device
        self.policy.normalizer.to(device=policy_device)

    def _make_range_normalizer(
        self,
        v_min: torch.Tensor,
        v_max: torch.Tensor,
        v_mean: torch.Tensor,
        v_std: torch.Tensor,
    ):
        output_max = 1.0
        output_min = -1.0
        range_eps = 1e-4

        input_range = v_max - v_min
        ignore_dim = input_range < range_eps
        input_range = input_range.clone()
        input_range[ignore_dim] = output_max - output_min

        scale = (output_max - output_min) / input_range
        offset = output_min - scale * v_min
        offset[ignore_dim] = (output_max + output_min) / 2.0 - v_min[ignore_dim]

        return self._single_field_normalizer_cls.create_manual(
            scale=scale.cpu().numpy(),
            offset=offset.cpu().numpy(),
            input_stats_dict={
                "min": v_min.cpu().numpy(),
                "max": v_max.cpu().numpy(),
                "mean": v_mean.cpu().numpy(),
                "std": v_std.cpu().numpy(),
            },
        )

    @staticmethod
    def _frames_to_policy_range(frames: torch.Tensor) -> torch.Tensor:
        return frames.clamp(-1.0, 1.0).add(1.0).mul(0.5)

    @staticmethod
    def _take_history(frames: torch.Tensor, n: int) -> torch.Tensor:
        """Return the trailing ``n`` frames along the time axis, left-padding by
        repeating the earliest available frame when fewer than ``n`` are given
        (e.g. the first step of a rollout). Training supplies exactly ``n``."""
        T = frames.shape[1]
        if T < n:
            pad = frames[:, :1].expand(frames.shape[0], n - T, *frames.shape[2:])
            frames = torch.cat([pad, frames], dim=1)
        return frames[:, -n:]

    def _build_obs_dict(
        self,
        input_frames: torch.Tensor,
        pred_frames: torch.Tensor,
        proprio: Optional[dict[str, torch.Tensor]] = None,
    ) -> dict[str, torch.Tensor]:
        # Separate single-frame rgb keys: image (i_k) + image_prev{i} (i_{k-i}),
        # plus the single predicted_image. Every key is T=1.
        obs: dict[str, torch.Tensor] = {}
        hist = self._take_history(input_frames, self.n_obs_steps)  # (B, n, …)
        for i, key in enumerate(self.history_image_keys):
            # history_image_keys[0]="image" is the most recent frame (hist[-1]);
            # image_prev{i} is i steps older (hist[n-1-i]).
            sl = self.n_obs_steps - 1 - i
            obs[key] = self._frames_to_policy_range(hist[:, sl : sl + 1])
        obs[self.predicted_image_key] = self._frames_to_policy_range(
            pred_frames[:, :1]
        )
        if self.proprio_keys:
            if proprio is None:
                raise ValueError(
                    "InverseDynamicsModelDP was configured with proprio obs keys "
                    f"{self.proprio_keys} but no proprio dict was passed to "
                    "forward()/generate(). Check that the dataset returns it and "
                    "the trainer threads it through."
                )
            for key in self.proprio_keys:
                tensor = proprio.get(key)
                if tensor is None:
                    raise KeyError(
                        f"Proprio dict missing key {key!r}; got {sorted(proprio)}."
                    )
                # Flatten the trailing n_obs_steps proprio frames into the
                # feature dim (oldest→newest), giving (B, 1, n*D) so the key
                # stays T=1 like the rgb keys. _take_history left-pads when the
                # dataset supplies fewer frames (e.g. the first rollout step).
                ph = self._take_history(
                    tensor.to(input_frames.device), self.n_obs_steps
                )  # (B, n, D)
                obs[key] = ph.reshape(ph.shape[0], 1, -1)  # (B, 1, n*D)
        return obs

    def _predict_next_frame(
        self,
        input_frames: torch.Tensor,
        goal: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # The obs may carry more history frames than the backbone was trained
        # on (n_obs_steps > frames_in); the AR backbone only consumes its
        # trailing frames_in context frames.
        if input_frames.shape[1] > self._backbone_fin:
            input_frames = input_frames[:, -self._backbone_fin :]
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
        proprio: Optional[dict[str, torch.Tensor]] = None,
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
        obs_dict = self._build_obs_dict(input_frames, pred_frames, proprio=proprio)

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
        proprio: Optional[dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        pred_frames = self._predict_next_frame(input_frames, goal=goal)
        obs_dict = self._build_obs_dict(input_frames, pred_frames, proprio=proprio)
        pred_actions = self.policy.predict_action(obs_dict)["action"]
        return pred_frames, pred_actions
