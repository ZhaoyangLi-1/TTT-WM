from __future__ import annotations

import copy
import os
import pathlib
import threading
from typing import Any

import dill
import torch
from omegaconf import OmegaConf

try:
    from hydra.core.hydra_config import HydraConfig

    HAS_HYDRA = True
except ImportError:
    HAS_HYDRA = False


def copy_to_cpu(x: Any) -> Any:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu()
    if isinstance(x, dict):
        return {key: copy_to_cpu(value) for key, value in x.items()}
    if isinstance(x, list):
        return [copy_to_cpu(value) for value in x]
    return copy.deepcopy(x)


class BaseWorkspace:
    include_keys: tuple[str, ...] = ("global_step", "epoch")
    exclude_keys: tuple[str, ...] = ()

    def __init__(self, cfg: OmegaConf, output_dir: str | None = None):
        self.cfg = cfg
        self._output_dir = output_dir
        cfg_output_dir = OmegaConf.select(cfg, "training.output_dir", default=None)
        if cfg_output_dir not in (None, "None"):
            self._output_dir = cfg_output_dir
        self._saving_thread: threading.Thread | None = None

    @property
    def output_dir(self) -> str:
        if self._output_dir is not None:
            return str(self._output_dir)
        if HAS_HYDRA:
            try:
                return HydraConfig.get().runtime.output_dir
            except Exception:
                pass
        return os.getcwd()

    def run(self) -> None:
        raise NotImplementedError

    def save_checkpoint(
        self,
        path: str | None = None,
        tag: str = "latest",
        exclude_keys: tuple[str, ...] | None = None,
        include_keys: tuple[str, ...] | None = None,
        use_thread: bool = True,
    ) -> str:
        ckpt_path = pathlib.Path(path) if path is not None else pathlib.Path(self.output_dir) / "checkpoints" / f"{tag}.ckpt"
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)

        exclude_keys = tuple(self.exclude_keys if exclude_keys is None else exclude_keys)
        include_keys = tuple(self.include_keys if include_keys is None else include_keys) + ("_output_dir",)

        payload = {"cfg": self.cfg, "state_dicts": {}, "pickles": {}}
        for key, value in self.__dict__.items():
            if hasattr(value, "state_dict") and hasattr(value, "load_state_dict"):
                if key not in exclude_keys:
                    payload["state_dicts"][key] = copy_to_cpu(value.state_dict()) if use_thread else value.state_dict()
            elif key in include_keys:
                payload["pickles"][key] = dill.dumps(value)

        def _save() -> None:
            torch.save(payload, ckpt_path.open("wb"), pickle_module=dill)

        if use_thread:
            self._saving_thread = threading.Thread(target=_save)
            self._saving_thread.start()
        else:
            _save()

        return str(ckpt_path.absolute())

    def get_checkpoint_path(self, tag: str = "latest") -> pathlib.Path:
        return pathlib.Path(self.output_dir) / "checkpoints" / f"{tag}.ckpt"

    def load_payload(
        self,
        payload: dict[str, Any],
        exclude_keys: tuple[str, ...] | None = None,
        include_keys: tuple[str, ...] | None = None,
        **kwargs: Any,
    ) -> None:
        exclude_keys = tuple() if exclude_keys is None else exclude_keys
        include_keys = tuple(payload["pickles"].keys()) if include_keys is None else include_keys

        for key, value in payload["state_dicts"].items():
            if key in exclude_keys or key not in self.__dict__:
                continue
            self.__dict__[key].load_state_dict(value, **kwargs)

        for key in include_keys:
            if key in payload["pickles"]:
                self.__dict__[key] = dill.loads(payload["pickles"][key])

    def load_checkpoint(
        self,
        path: str | None = None,
        tag: str = "latest",
        exclude_keys: tuple[str, ...] | None = None,
        include_keys: tuple[str, ...] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        ckpt_path = pathlib.Path(path) if path is not None else self.get_checkpoint_path(tag=tag)
        payload = torch.load(ckpt_path.open("rb"), pickle_module=dill, **kwargs)
        self.load_payload(
            payload,
            exclude_keys=exclude_keys,
            include_keys=include_keys,
        )
        return payload

    def save_snapshot(self, tag: str = "latest") -> str:
        snapshot_path = pathlib.Path(self.output_dir) / "snapshots" / f"{tag}.pkl"
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self, snapshot_path.open("wb"), pickle_module=dill)
        return str(snapshot_path.absolute())

