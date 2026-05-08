from __future__ import annotations

import argparse
import asyncio
import copy
import functools
import http
import inspect
import logging
import os
from collections import deque
from pathlib import Path
import sys
import time
import traceback
from typing import Any

import dill
import hydra
import msgpack
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
import torch
import websockets
import websockets.asyncio.server as ws_server
import websockets.frames


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cosmos_model import ARPatchConfig, ARVideoPatchTransformer
from dp.common import resolve_checkpoint_path as resolve_dp_checkpoint_path
from dp.common import resolve_device
from dp.runtime import configure_diffusion_policy_path, register_omegaconf_resolvers
from idm_model import InverseDynamicsModelDP
from pure_idm import PureInverseDynamicsModelDP


logger = logging.getLogger(__name__)


def pack_array(obj: Any) -> Any:
    if (isinstance(obj, (np.ndarray, np.generic))) and obj.dtype.kind in ("V", "O", "c"):
        raise ValueError(f"Unsupported dtype: {obj.dtype}")

    if isinstance(obj, np.ndarray):
        return {
            b"__ndarray__": True,
            b"data": obj.tobytes(),
            b"dtype": obj.dtype.str,
            b"shape": obj.shape,
        }

    if isinstance(obj, np.generic):
        return {
            b"__npgeneric__": True,
            b"data": obj.item(),
            b"dtype": obj.dtype.str,
        }

    return obj


def unpack_array(obj: dict[bytes, Any]) -> Any:
    if b"__ndarray__" in obj:
        return np.ndarray(buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]), shape=obj[b"shape"])
    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])
    return obj


Packer = functools.partial(msgpack.Packer, default=pack_array)
packb = functools.partial(msgpack.packb, default=pack_array)
unpackb = functools.partial(msgpack.unpackb, object_hook=unpack_array)


IMAGE_ALIASES = (
    "observation/image",
    "image",
    "agentview_image",
)
WRIST_IMAGE_ALIASES = (
    "observation/wrist_image",
    "wrist_image",
    "robot0_eye_in_hand_image",
    "eye_in_hand_image",
)
STATE_ALIASES = (
    "observation/state",
    "state",
)
GOAL_IMAGE_ALIASES = (
    "observation/goal_image",
    "goal_image",
    "goal",
)
NEXT_IMAGE_ALIASES = (
    "observation/next_image",
    "next_image",
    "oracle_next_image",
)


def _strip_prefix(sd: dict[str, Any], prefix: str) -> dict[str, Any]:
    if any(key.startswith(prefix) for key in sd):
        return {key.removeprefix(prefix): value for key, value in sd.items()}
    return sd


def clean_state_dict(sd: dict[str, Any]) -> dict[str, Any]:
    sd = _strip_prefix(sd, "module._orig_mod.")
    sd = _strip_prefix(sd, "_orig_mod.module.")
    sd = _strip_prefix(sd, "module.")
    sd = _strip_prefix(sd, "_orig_mod.")
    return sd


def load_state_dict_flexible(module: Any, state_dict: dict[str, Any]) -> None:
    load_fn = module.load_state_dict
    try:
        signature = inspect.signature(load_fn)
        if "strict" in signature.parameters:
            load_fn(state_dict, strict=False)
            return
    except (TypeError, ValueError):
        pass
    load_fn(state_dict)


def resolve_checkpoint_path(path_or_dir: str) -> Path:
    path = Path(path_or_dir).expanduser()
    if path.is_file():
        return path
    if not path.is_dir():
        raise FileNotFoundError(f"Checkpoint path does not exist: {path_or_dir}")

    preferred = [
        path / "last.pt",
        path / "best.pt",
        path / "checkpoints" / "latest.ckpt",
    ]
    for candidate in preferred:
        if candidate.is_file():
            return candidate

    candidates: list[Path] = []
    for pattern in ("*.pt", "*.ckpt"):
        candidates.extend(path.glob(pattern))
        candidates.extend((path / "checkpoints").glob(pattern) if (path / "checkpoints").is_dir() else [])
    candidates = [candidate for candidate in candidates if candidate.is_file()]
    if not candidates:
        raise FileNotFoundError(
            f"No checkpoint found in {path_or_dir} or {path_or_dir}/checkpoints"
        )
    return max(candidates, key=lambda item: item.stat().st_mtime)


def load_checkpoint_payload(path: Path, *, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    try:
        return torch.load(path, map_location=map_location, pickle_module=dill, weights_only=False)
    except Exception:
        return torch.load(path, map_location=map_location, weights_only=False)


def select_stage_weights(ckpt: dict[str, Any], use_ema: bool) -> tuple[dict[str, Any], str]:
    has_ema = "ema" in ckpt
    if use_ema and has_ema:
        ema_state = ckpt["ema"]
        if isinstance(ema_state, dict) and "shadow" in ema_state:
            return clean_state_dict(ema_state["shadow"]), "ema"
        return clean_state_dict(ema_state), "ema"
    return clean_state_dict(ckpt["model"]), "live"


def select_dp_weights(payload: dict[str, Any], use_ema: bool) -> tuple[dict[str, Any], str]:
    state_dicts = payload.get("state_dicts", {})
    if use_ema and isinstance(state_dicts.get("ema_model"), dict):
        return clean_state_dict(state_dicts["ema_model"]), "ema_model"
    if isinstance(state_dicts.get("model"), dict):
        return clean_state_dict(state_dicts["model"]), "model"
    raise KeyError("DP checkpoint does not contain `state_dicts['model']` or `state_dicts['ema_model']`.")


def ensure_uint8_hwc_image(value: Any, *, resolution: int) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 3:
        raise ValueError(f"Expected image with 3 dims, got shape {array.shape}.")

    if array.shape[0] in (1, 3) and array.shape[-1] not in (1, 3):
        array = np.moveaxis(array, 0, -1)

    if array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)
    if array.shape[-1] != 3:
        raise ValueError(f"Expected RGB image, got shape {array.shape}.")

    if array.dtype != np.uint8:
        array = array.astype(np.float32)
        if np.nanmax(array) <= 1.0:
            array = array * 255.0
        array = np.clip(array, 0.0, 255.0).astype(np.uint8)

    image = Image.fromarray(array, mode="RGB")
    if image.size != (resolution, resolution):
        image = image.resize((resolution, resolution), Image.BILINEAR)
    return np.asarray(image, dtype=np.uint8)


def image_to_dp_tensor(value: Any, *, resolution: int) -> torch.Tensor:
    array = ensure_uint8_hwc_image(value, resolution=resolution)
    tensor = torch.from_numpy(np.ascontiguousarray(np.moveaxis(array, -1, 0))).float()
    return tensor / 255.0


def image_to_world_model_tensor(value: Any, *, resolution: int) -> torch.Tensor:
    array = ensure_uint8_hwc_image(value, resolution=resolution)
    tensor = torch.from_numpy(np.ascontiguousarray(np.moveaxis(array, -1, 0))).float()
    return tensor / 127.5 - 1.0


def world_tensor_to_uint8_image(tensor: torch.Tensor) -> np.ndarray:
    array = tensor.detach().cpu().float().clamp(-1.0, 1.0)
    array = ((array + 1.0) * 127.5).round().to(torch.uint8).numpy()
    return np.moveaxis(array, 0, -1)


def lowdim_to_tensor(value: Any, *, expected_shape: tuple[int, ...]) -> torch.Tensor:
    array = np.asarray(value, dtype=np.float32)
    try:
        array = array.reshape(expected_shape)
    except ValueError as exc:
        raise ValueError(
            f"Could not reshape low-dim observation from {array.shape} to {expected_shape}."
        ) from exc
    return torch.from_numpy(np.ascontiguousarray(array))


def try_get(obs: dict[str, Any], keys: tuple[str, ...]) -> Any | None:
    for key in keys:
        if key in obs:
            return obs[key]
    return None


def infer_source_alias(obs_key: str, obs_type: str) -> tuple[str, ...]:
    key = obs_key.lower()
    if obs_type == "rgb":
        if "wrist" in key or "hand" in key or "eye" in key:
            return WRIST_IMAGE_ALIASES
        if "goal" in key:
            return GOAL_IMAGE_ALIASES
        if "next" in key or "predicted" in key:
            return NEXT_IMAGE_ALIASES
        return IMAGE_ALIASES
    if "state" in key:
        return STATE_ALIASES
    return (obs_key,)


def maybe_configure_diffusion_policy(explicit_path: str | None, cfg: Any | None = None) -> None:
    config_path = explicit_path
    if cfg is not None and config_path in (None, ""):
        config_path = OmegaConf.select(cfg, "runtime.diffusion_policy_src", default=None)
    configure_diffusion_policy_path(config_path)


class BasePolicyAdapter:
    def new_session(self) -> dict[str, Any]:
        return {}

    def infer(self, obs: dict[str, Any], session: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    @property
    def metadata(self) -> dict[str, Any]:
        raise NotImplementedError


class DPPolicyAdapter(BasePolicyAdapter):
    def __init__(
        self,
        policy: Any,
        cfg: Any,
        *,
        checkpoint_path: Path,
        device: torch.device,
        weight_source: str,
    ) -> None:
        self._policy = policy
        self._cfg = cfg
        self._device = device
        shape_meta = OmegaConf.to_container(cfg.shape_meta, resolve=True)
        self._obs_meta = shape_meta["obs"]
        self._n_obs_steps = int(OmegaConf.select(cfg, "n_obs_steps", default=1))
        self._horizon = int(OmegaConf.select(cfg, "n_action_steps", default=1))
        self._action_dim = int(shape_meta["action"]["shape"][0])
        self._source_aliases = {
            key: infer_source_alias(key, attr.get("type", "low_dim"))
            for key, attr in self._obs_meta.items()
        }
        self._metadata = {
            "model_type": "dp",
            "causal": True,
            "checkpoint": str(checkpoint_path),
            "weight_source": weight_source,
            "input_resolution": self._infer_input_resolution(),
            "action_horizon": self._horizon,
            "action_dim": self._action_dim,
            "n_obs_steps": self._n_obs_steps,
            "policy_obs_keys": list(self._obs_meta.keys()),
            "preferred_image_key": "observation/image",
            "preferred_wrist_key": "observation/wrist_image",
            "preferred_state_key": "observation/state",
        }

    def _infer_input_resolution(self) -> int | None:
        for attr in self._obs_meta.values():
            if attr.get("type", "low_dim") == "rgb":
                shape = tuple(attr["shape"])
                if len(shape) == 3:
                    return int(shape[-1])
        return None

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata

    def new_session(self) -> dict[str, Any]:
        return {
            "history": {
                key: deque(maxlen=self._n_obs_steps)
                for key in self._obs_meta
            }
        }

    def _build_model_obs(self, obs: dict[str, Any], session: dict[str, Any]) -> dict[str, torch.Tensor]:
        if obs.get("reset"):
            for history in session["history"].values():
                history.clear()

        model_obs: dict[str, torch.Tensor] = {}
        for key, attr in self._obs_meta.items():
            raw_value = try_get(obs, self._source_aliases[key])
            if raw_value is None and key in obs:
                raw_value = obs[key]
            if raw_value is None:
                raise KeyError(
                    f"Missing observation for policy key {key!r}. Tried aliases: {self._source_aliases[key]}"
                )

            obs_type = attr.get("type", "low_dim")
            if obs_type == "rgb":
                resolution = int(attr["shape"][-1])
                item = image_to_dp_tensor(raw_value, resolution=resolution)
            else:
                item = lowdim_to_tensor(raw_value, expected_shape=tuple(attr["shape"]))

            history: deque[torch.Tensor] = session["history"][key]
            if not history:
                for _ in range(self._n_obs_steps):
                    history.append(item.clone())
            else:
                history.append(item)

            stacked = torch.stack(list(history), dim=0).unsqueeze(0).to(self._device)
            model_obs[key] = stacked

        return model_obs

    def infer(self, obs: dict[str, Any], session: dict[str, Any]) -> dict[str, Any]:
        model_obs = self._build_model_obs(obs, session)
        start = time.monotonic()
        with torch.no_grad():
            result = self._policy.predict_action(model_obs)
            action = result.get("action_pred", result["action"])
        infer_ms = (time.monotonic() - start) * 1000.0
        action_np = action[0].detach().cpu().float().numpy()
        if action_np.ndim == 1:
            action_np = action_np[None, :]
        return {
            "actions": action_np,
            "policy_timing": {"infer_ms": infer_ms},
        }


class Stage2PolicyAdapter(BasePolicyAdapter):
    def __init__(
        self,
        model: torch.nn.Module,
        cfg: dict[str, Any],
        *,
        checkpoint_path: Path,
        device: torch.device,
        weight_source: str,
    ) -> None:
        self._model = model
        self._cfg = cfg
        self._device = device
        model_cfg = cfg["model"]
        data_cfg = cfg["data"]
        self._frames_in = int(model_cfg["frames_in"])
        self._resolution = int(model_cfg["resolution"])
        self._use_goal = bool(data_cfg.get("use_goal", False))
        self._action_horizon = int(data_cfg.get("frame_gap", 1))
        self._action_dim = int(data_cfg.get("action_dim", 7))
        self._metadata = {
            "model_type": "stage2",
            "causal": True,
            "checkpoint": str(checkpoint_path),
            "weight_source": weight_source,
            "input_resolution": self._resolution,
            "frames_in": self._frames_in,
            "action_horizon": self._action_horizon,
            "action_dim": self._action_dim,
            "goal_conditioned": self._use_goal,
            "goal_optional": True,
            "preferred_image_key": "observation/image",
            "preferred_goal_key": "goal_image",
        }

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata

    def new_session(self) -> dict[str, Any]:
        return {
            "image_history": deque(maxlen=self._frames_in),
            "warned_missing_goal": False,
        }

    def _build_context(self, obs: dict[str, Any], session: dict[str, Any]) -> torch.Tensor:
        if obs.get("reset"):
            session["image_history"].clear()
            session["warned_missing_goal"] = False

        image_value = try_get(obs, IMAGE_ALIASES)
        if image_value is None:
            raise KeyError(f"Missing image observation. Tried aliases: {IMAGE_ALIASES}")

        image = image_to_world_model_tensor(image_value, resolution=self._resolution)
        history: deque[torch.Tensor] = session["image_history"]
        if not history:
            for _ in range(self._frames_in):
                history.append(image.clone())
        else:
            history.append(image)
        return torch.stack(list(history), dim=0).unsqueeze(0).to(self._device)

    def _build_goal(self, obs: dict[str, Any], session: dict[str, Any]) -> torch.Tensor | None:
        goal_value = try_get(obs, GOAL_IMAGE_ALIASES)
        if goal_value is None:
            if self._use_goal and not session["warned_missing_goal"]:
                logger.warning(
                    "Stage2 checkpoint was trained with goal conditioning, but the client did not send a goal image. "
                    "Inference will run with goal=None."
                )
                session["warned_missing_goal"] = True
            return None
        goal = image_to_world_model_tensor(goal_value, resolution=self._resolution)
        return goal.unsqueeze(0).to(self._device)

    def infer(self, obs: dict[str, Any], session: dict[str, Any]) -> dict[str, Any]:
        t0 = time.monotonic()
        context = self._build_context(obs, session)
        goal = self._build_goal(obs, session)
        if self._device.type == "cuda":
            torch.cuda.synchronize()
        t_pre = time.monotonic()

        autocast_device = "cuda" if self._device.type == "cuda" else "cpu"
        with torch.no_grad(), torch.autocast(device_type=autocast_device, dtype=torch.bfloat16):
            pred_frames = self._model._predict_next_frame(context, goal=goal)
            if self._device.type == "cuda":
                torch.cuda.synchronize()
            t_stage1 = time.monotonic()
            obs_dict = self._model._build_obs_dict(context, pred_frames)
            pred_actions = self._model.policy.predict_action(obs_dict)["action"]
        if self._device.type == "cuda":
            torch.cuda.synchronize()
        t_policy = time.monotonic()

        actions = pred_actions[0].detach().cpu().float().numpy()
        if actions.ndim == 1:
            actions = actions[None, :]
        infer_ms = (t_policy - t_pre) * 1000.0
        timings = {
            "pre_ms":     (t_pre     - t0)      * 1000.0,
            "stage1_ms": (t_stage1   - t_pre)   * 1000.0,
            "policy_ms": (t_policy   - t_stage1) * 1000.0,
            "infer_ms":   infer_ms,
        }
        logger.info(
            "stage2 timing: pre=%.0fms stage1=%.0fms policy=%.0fms",
            timings["pre_ms"], timings["stage1_ms"], timings["policy_ms"],
        )
        response: dict[str, Any] = {
            "actions": actions,
            "policy_timing": timings,
        }
        if pred_frames is not None and pred_frames.shape[1] > 0:
            response["predicted_image"] = world_tensor_to_uint8_image(pred_frames[0, 0])
        return response


class PureIDMPolicyAdapter(BasePolicyAdapter):
    def __init__(
        self,
        model: torch.nn.Module,
        cfg: dict[str, Any],
        *,
        checkpoint_path: Path,
        device: torch.device,
        weight_source: str,
    ) -> None:
        self._model = model
        self._cfg = cfg
        self._device = device
        model_cfg = cfg["model"]
        data_cfg = cfg["data"]
        self._resolution = int(model_cfg["resolution"])
        self._action_horizon = int(data_cfg.get("frame_gap", 1))
        self._action_dim = int(data_cfg.get("action_dim", 7))
        self._metadata = {
            "model_type": "pure_idm",
            "causal": False,
            "checkpoint": str(checkpoint_path),
            "weight_source": weight_source,
            "input_resolution": self._resolution,
            "action_horizon": self._action_horizon,
            "action_dim": self._action_dim,
            "requires_next_image": True,
            "preferred_image_key": "observation/image",
            "preferred_next_image_key": "next_image",
            "note": "Pure IDM is non-causal and needs the ground-truth next image.",
        }

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata

    def infer(self, obs: dict[str, Any], session: dict[str, Any]) -> dict[str, Any]:
        del session
        image_value = try_get(obs, IMAGE_ALIASES)
        next_image_value = try_get(obs, NEXT_IMAGE_ALIASES)
        if image_value is None:
            raise KeyError(f"Missing image observation. Tried aliases: {IMAGE_ALIASES}")
        if next_image_value is None:
            raise ValueError(
                "Pure IDM checkpoints are non-causal and require the next image. "
                f"Send one of {NEXT_IMAGE_ALIASES} in the request payload."
            )

        current = image_to_world_model_tensor(image_value, resolution=self._resolution)
        next_frame = image_to_world_model_tensor(next_image_value, resolution=self._resolution)
        context = current.unsqueeze(0).unsqueeze(0).to(self._device)
        target = next_frame.unsqueeze(0).unsqueeze(0).to(self._device)

        start = time.monotonic()
        autocast_device = "cuda" if self._device.type == "cuda" else "cpu"
        with torch.no_grad(), torch.autocast(device_type=autocast_device, dtype=torch.bfloat16):
            _, pred_actions = self._model.generate(context, target_frames=target)
        infer_ms = (time.monotonic() - start) * 1000.0
        actions = pred_actions[0].detach().cpu().float().numpy()
        if actions.ndim == 1:
            actions = actions[None, :]
        return {
            "actions": actions,
            "policy_timing": {"infer_ms": infer_ms},
        }


def build_world_model_cfg(cfg: dict[str, Any]) -> ARPatchConfig:
    model_cfg = cfg["model"]
    data_cfg = cfg["data"]
    return ARPatchConfig(
        resolution=int(model_cfg["resolution"]),
        num_channels=int(model_cfg["num_channels"]),
        patch_size=int(model_cfg["patch_size"]),
        d_model=int(model_cfg["d_model"]),
        n_heads=int(model_cfg["n_heads"]),
        n_layers=int(model_cfg["n_layers"]),
        mlp_ratio=float(model_cfg["mlp_ratio"]),
        dropout=float(model_cfg.get("dropout", 0.0)),
        frames_in=int(model_cfg["frames_in"]),
        frames_out=int(model_cfg["frames_out"]),
        action_dim=int(data_cfg.get("action_dim", 7)),
        qk_norm=bool(model_cfg.get("qk_norm", True)),
        parallel_attn=bool(model_cfg.get("parallel_attn", False)),
    )


def load_dp_adapter(
    checkpoint_path: Path,
    *,
    device: torch.device,
    use_ema: bool,
    diffusion_policy_src: str | None,
) -> BasePolicyAdapter:
    payload = load_checkpoint_payload(checkpoint_path, map_location="cpu")
    cfg = copy.deepcopy(payload["cfg"])
    register_omegaconf_resolvers()
    maybe_configure_diffusion_policy(diffusion_policy_src, cfg)
    OmegaConf.resolve(cfg)

    policy = hydra.utils.instantiate(cfg.policy)
    state_dict, weight_source = select_dp_weights(payload, use_ema=use_ema)
    load_state_dict_flexible(policy, state_dict)
    policy.to(device)
    policy.eval()
    logger.info("Loaded diffusion policy checkpoint %s (%s)", checkpoint_path, weight_source)
    return DPPolicyAdapter(
        policy,
        cfg,
        checkpoint_path=checkpoint_path,
        device=device,
        weight_source=weight_source,
    )


def load_stage2_adapter(
    checkpoint_path: Path,
    *,
    device: torch.device,
    use_ema: bool,
    diffusion_policy_src: str | None,
) -> BasePolicyAdapter:
    ckpt = load_checkpoint_payload(checkpoint_path, map_location="cpu")
    cfg = ckpt["cfg"]
    substep = str(cfg.get("train", {}).get("substep", "2.1")).strip().lower()
    if substep in {"2.1", "1", "part1"}:
        raise ValueError(
            "Stage2 substep 2.1 checkpoints only predict frames and do not have an action head. "
            "Use a Stage2 substep 2.2 checkpoint instead."
        )

    model_cfg = build_world_model_cfg(cfg)
    raw_wm = ARVideoPatchTransformer(model_cfg).to(device)
    maybe_configure_diffusion_policy(diffusion_policy_src)
    idm_kwargs = dict(cfg["train"].get("idm", {}) or {})
    for k in ("use_stage1_cache", "stage1_cache_dir", "require_cache_sha1"):
        idm_kwargs.pop(k, None)
    model = InverseDynamicsModelDP(
        raw_wm,
        n_actions=int(cfg["data"].get("frame_gap", 1)),
        freeze_stage1=True,
        **idm_kwargs,
    ).to(device)

    state_dict, weight_source = select_stage_weights(ckpt, use_ema=use_ema)
    load_state_dict_flexible(model, state_dict)
    # DictOfTensorMixin._load_from_state_dict rebuilds the policy's normalizer
    # ParameterDict on CPU during load; move everything back to ``device``.
    model.to(device)
    model.eval()
    logger.info("Loaded stage2 checkpoint %s (%s)", checkpoint_path, weight_source)
    return Stage2PolicyAdapter(
        model,
        cfg,
        checkpoint_path=checkpoint_path,
        device=device,
        weight_source=weight_source,
    )


def load_pure_idm_adapter(
    checkpoint_path: Path,
    *,
    device: torch.device,
    use_ema: bool,
    diffusion_policy_src: str | None,
) -> BasePolicyAdapter:
    ckpt = load_checkpoint_payload(checkpoint_path, map_location="cpu")
    cfg = ckpt["cfg"]
    model_cfg = build_world_model_cfg(cfg)
    maybe_configure_diffusion_policy(diffusion_policy_src)
    model = PureInverseDynamicsModelDP(
        model_cfg,
        n_actions=int(cfg["data"].get("frame_gap", 1)),
        **cfg["train"].get("idm", {}),
    ).to(device)

    state_dict, weight_source = select_stage_weights(ckpt, use_ema=use_ema)
    load_state_dict_flexible(model, state_dict)
    model.to(device)
    model.eval()
    logger.info("Loaded pure_idm checkpoint %s (%s)", checkpoint_path, weight_source)
    return PureIDMPolicyAdapter(
        model,
        cfg,
        checkpoint_path=checkpoint_path,
        device=device,
        weight_source=weight_source,
    )


def detect_model_type(payload: dict[str, Any]) -> str:
    if "state_dicts" in payload:
        return "dp"
    cfg = payload.get("cfg", {})
    train_cfg = cfg.get("train", {}) if isinstance(cfg, dict) else {}
    stage = str(train_cfg.get("stage", "")).lower()
    substep = str(train_cfg.get("substep", "")).lower()
    if stage == "pure_idm":
        return "pure_idm"
    if stage == "2" or substep in {"2.1", "2.2", "part1", "part2"}:
        return "stage2"
    if stage == "1":
        return "stage1"
    raise ValueError("Could not auto-detect checkpoint type. Pass --model-type explicitly.")


def load_adapter(
    checkpoint: str,
    *,
    model_type: str,
    device: torch.device,
    use_ema: bool,
    diffusion_policy_src: str | None,
) -> BasePolicyAdapter:
    if model_type == "dp":
        checkpoint_path = resolve_dp_checkpoint_path(checkpoint)
        return load_dp_adapter(
            checkpoint_path,
            device=device,
            use_ema=use_ema,
            diffusion_policy_src=diffusion_policy_src,
        )

    checkpoint_path = resolve_checkpoint_path(checkpoint)
    if model_type == "auto":
        payload = load_checkpoint_payload(checkpoint_path, map_location="cpu")
        model_type = detect_model_type(payload)
    if model_type == "dp":
        return load_dp_adapter(
            checkpoint_path,
            device=device,
            use_ema=use_ema,
            diffusion_policy_src=diffusion_policy_src,
        )
    if model_type == "stage1":
        raise ValueError("Stage1 checkpoints only predict frames and cannot serve actions.")
    if model_type == "stage2":
        return load_stage2_adapter(
            checkpoint_path,
            device=device,
            use_ema=use_ema,
            diffusion_policy_src=diffusion_policy_src,
        )
    if model_type == "pure_idm":
        return load_pure_idm_adapter(
            checkpoint_path,
            device=device,
            use_ema=use_ema,
            diffusion_policy_src=diffusion_policy_src,
        )
    raise ValueError(f"Unsupported model_type={model_type!r}")


class WebsocketPolicyServer:
    def __init__(
        self,
        adapter: BasePolicyAdapter,
        *,
        host: str = "0.0.0.0",
        port: int = 8000,
    ) -> None:
        self._adapter = adapter
        self._host = host
        self._port = port
        logging.getLogger("websockets.server").setLevel(logging.INFO)

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self) -> None:
        async with ws_server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
            process_request=_health_check,
        ) as server:
            await server.serve_forever()

    async def _handler(self, websocket: ws_server.ServerConnection) -> None:
        logger.info("Connection from %s opened", websocket.remote_address)
        packer = Packer()
        session = self._adapter.new_session()

        await websocket.send(packer.pack(self._adapter.metadata))

        prev_total_time = None
        infer_count = 0
        while True:
            try:
                start_time = time.monotonic()
                obs = unpackb(await websocket.recv())
                infer_start = time.monotonic()
                result = self._adapter.infer(obs, session)
                infer_ms = (time.monotonic() - infer_start) * 1000.0
                result.setdefault("server_timing", {})
                if prev_total_time is not None:
                    result["server_timing"]["prev_total_ms"] = prev_total_time * 1000.0
                await websocket.send(packer.pack(result))
                prev_total_time = time.monotonic() - start_time
                infer_count += 1
                logger.info(
                    "infer #%d ok (infer_ms=%.0f, total_ms=%.0f)",
                    infer_count, infer_ms, prev_total_time * 1000.0,
                )
            except websockets.ConnectionClosed:
                logger.info("Connection from %s closed", websocket.remote_address)
                break
            except Exception:
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                raise


def _health_check(
    connection: ws_server.ServerConnection,
    request: ws_server.Request,
) -> ws_server.Response | None:
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Serve TTT-WM checkpoints over the LIBERO websocket protocol."
    )
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint file or output directory.")
    parser.add_argument(
        "--model-type",
        type=str,
        default="auto",
        choices=["auto", "dp", "stage1", "stage2", "pure_idm"],
        help="Checkpoint family. `auto` inspects the checkpoint payload.",
    )
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--use-ema", dest="use_ema", action="store_true")
    parser.add_argument("--no-ema", dest="use_ema", action="store_false")
    parser.set_defaults(use_ema=True)
    parser.add_argument(
        "--diffusion-policy-src",
        type=str,
        default=None,
        help="Optional diffusion_policy source checkout for DP-based checkpoints.",
    )
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=None,
        help=(
            "Override the diffusion policy's denoising step count at inference. "
            "Lower values (e.g. 16-32) trade a small fidelity loss for big latency wins."
        ),
    )
    parser.add_argument(
        "--use-ddim",
        action="store_true",
        help=(
            "Replace the policy's DDPM sampler with a DDIMScheduler that reuses the "
            "same betas/alphas. Pairs naturally with --num-inference-steps to take far "
            "fewer denoising steps at inference without retraining."
        ),
    )
    parser.add_argument(
        "--stage1-frames-out",
        type=int,
        default=None,
        help=(
            "Override the world-model's frames_out at inference. The IDM head only "
            "consumes pred_frames[:, :1] (see idm_model._build_obs_dict), so setting "
            "this to 1 preserves the first predicted frame's value (AR is causal) but "
            "skips ~75%% of stage1 decode steps when training used frames_out=4."
        ),
    )
    parser.add_argument(
        "--compile-policy",
        action="store_true",
        help=(
            "Wrap the diffusion U-Net with torch.compile (mode='reduce-overhead') to "
            "reduce per-step overhead. First inference incurs a 30-60s trace; steady "
            "state is typically 30-50%% faster."
        ),
    )
    parser.add_argument(
        "--compile-stage1",
        action="store_true",
        help=(
            "Also compile the stage1 world model. Higher risk of trace failure due to "
            "flash_attn / dynamic AR loop control flow; usually a smaller win since "
            "stage1 is not the bottleneck."
        ),
    )
    parser.add_argument(
        "--stage1-bf16",
        action="store_true",
        help=(
            "Cast stage1 weights to bf16 directly (instead of relying on autocast). "
            "Removes per-op dtype check/cast overhead in the AR decode loop; only "
            "safe on Ampere+ GPUs (A10 / A100 / H100 / RTX 30+/40+) which support "
            "bf16 natively."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    adapter = load_adapter(
        args.checkpoint,
        model_type=args.model_type,
        device=device,
        use_ema=bool(args.use_ema),
        diffusion_policy_src=args.diffusion_policy_src,
    )
    def _iter_diffusion_policies():
        # Stage2 / Pure IDM keep the diffusion policy under adapter._model.policy.
        owner = getattr(adapter, "_model", None)
        nested = getattr(owner, "policy", None) if owner is not None else None
        if nested is not None:
            yield nested
        # DP adapter exposes the policy directly as adapter._policy.
        dp_policy = getattr(adapter, "_policy", None)
        if dp_policy is not None:
            yield dp_policy

    if args.use_ddim:
        try:
            from diffusers.schedulers.scheduling_ddim import DDIMScheduler
        except ImportError as exc:
            raise ImportError(
                "--use-ddim requires `diffusers`; install via `pip install diffusers`."
            ) from exc
        swapped = 0
        for policy in _iter_diffusion_policies():
            ddpm = getattr(policy, "noise_scheduler", None)
            if ddpm is None:
                continue
            cfg_d = dict(ddpm.config)
            ddim = DDIMScheduler(
                num_train_timesteps=cfg_d.get("num_train_timesteps", 100),
                beta_start=cfg_d.get("beta_start", 0.0001),
                beta_end=cfg_d.get("beta_end", 0.02),
                beta_schedule=cfg_d.get("beta_schedule", "squaredcos_cap_v2"),
                clip_sample=cfg_d.get("clip_sample", True),
                prediction_type=cfg_d.get("prediction_type", "epsilon"),
                set_alpha_to_one=True,
                steps_offset=0,
            )
            policy.noise_scheduler = ddim
            swapped += 1
        if swapped:
            logger.info("Swapped %d policy scheduler(s) to DDIM.", swapped)
        else:
            logger.warning("--use-ddim requested but no policy with noise_scheduler found.")

    if args.num_inference_steps is not None:
        steps = int(args.num_inference_steps)
        applied = 0
        for policy in _iter_diffusion_policies():
            if hasattr(policy, "num_inference_steps"):
                policy.num_inference_steps = steps
                applied += 1
        if applied:
            logger.info("Overriding num_inference_steps to %d on %d policy(ies).", steps, applied)
        else:
            logger.warning(
                "num_inference_steps override requested but no compatible policy found."
            )

    if args.stage1_frames_out is not None:
        new_fout = int(args.stage1_frames_out)
        owner = getattr(adapter, "_model", None)
        stage1 = getattr(owner, "stage1", None) if owner is not None else None
        cfg = getattr(stage1, "cfg", None) if stage1 is not None else None
        if cfg is not None and hasattr(cfg, "frames_out"):
            old_fout = int(cfg.frames_out)
            cfg.frames_out = new_fout
            logger.info(
                "Stage1 frames_out override: %d -> %d (saves ~%.0f%% of AR decode steps).",
                old_fout, new_fout, max(0.0, (1 - new_fout / max(old_fout, 1)) * 100),
            )
        else:
            logger.warning(
                "--stage1-frames-out requested but adapter has no model.stage1.cfg.frames_out."
            )

    if args.compile_policy:
        compiled = 0
        for policy in _iter_diffusion_policies():
            unet = getattr(policy, "model", None)
            if unet is None:
                continue
            try:
                # ``default`` (inductor JIT) instead of ``reduce-overhead``: the
                # diffusion U-Net forward receives ``timesteps`` as a CPU scalar
                # tensor (it does ``timesteps.to(sample.device)`` internally),
                # which makes CUDA Graph capture fall back to eager. ``default``
                # still kernel-fuses without requiring all-cuda inputs.
                policy.model = torch.compile(
                    unet,
                    mode="default",
                    fullgraph=False,
                    dynamic=False,
                )
                compiled += 1
            except Exception as exc:
                logger.warning("torch.compile(policy.model) failed: %s", exc)
        if compiled:
            logger.info(
                "torch.compile applied to %d diffusion U-Net(s); first infer will be slow (warmup).",
                compiled,
            )
        else:
            logger.warning("--compile-policy requested but no policy.model attribute found.")

    if args.compile_stage1:
        owner = getattr(adapter, "_model", None)
        stage1 = getattr(owner, "stage1", None) if owner is not None else None
        if stage1 is not None:
            try:
                owner.stage1 = torch.compile(
                    stage1,
                    mode="reduce-overhead",
                    fullgraph=False,
                    dynamic=False,
                )
                logger.info("torch.compile applied to stage1; first infer will be slow.")
            except Exception as exc:
                logger.warning("torch.compile(stage1) failed: %s", exc)
        else:
            logger.warning("--compile-stage1 requested but adapter has no model.stage1.")

    if args.stage1_bf16:
        owner = getattr(adapter, "_model", None)
        stage1 = getattr(owner, "stage1", None) if owner is not None else None
        if stage1 is not None:
            stage1.to(torch.bfloat16)
            logger.info("Cast stage1 weights to bf16.")
        else:
            logger.warning("--stage1-bf16 requested but adapter has no model.stage1.")
    logger.info("Serving %s on ws://%s:%s", adapter.metadata.get("model_type"), args.host, args.port)
    server = WebsocketPolicyServer(adapter, host=args.host, port=args.port)
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
