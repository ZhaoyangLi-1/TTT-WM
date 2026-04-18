from __future__ import annotations

import builtins
import logging
import os
import sys
import traceback

import hydra
from omegaconf import DictConfig, OmegaConf
from torch.distributed.elastic.multiprocessing.errors import record

from dp.runtime import configure_diffusion_policy_path, register_omegaconf_resolvers


def _is_main_rank() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


def _silence_non_main_rank_logging() -> None:
    if _is_main_rank():
        return

    def _silent_print(*args, **kwargs) -> None:
        return None

    builtins.print = _silent_print
    logging.disable(logging.WARNING)


_silence_non_main_rank_logging()
register_omegaconf_resolvers()


def _apply_selected_task_overrides(cfg: DictConfig) -> None:
    selected_task = OmegaConf.select(cfg, "data.selected_task", default=None)
    if selected_task in (None, "", "None"):
        return

    cfg.data.split_mode = "episode"
    if _is_main_rank():
        print(
            "Using task-filtered episode split for diffusion policy training: "
            f"data.selected_task={selected_task!r}, "
            f"data.split_mode={cfg.data.split_mode!r}, "
            f"data.val_ratio={float(cfg.data.val_ratio):.3f}"
        )


@hydra.main(version_base=None, config_path="configs", config_name="dp_config")
def _hydra_main(cfg: DictConfig) -> None:
    diffusion_policy_src = OmegaConf.select(
        cfg, "runtime.diffusion_policy_src", default=None
    )
    configured_paths = configure_diffusion_policy_path(diffusion_policy_src)
    if configured_paths and _is_main_rank():
        print("Configured diffusion_policy search path:")
        for path in configured_paths:
            print(f"  - {path}")

    _apply_selected_task_overrides(cfg)
    OmegaConf.resolve(cfg)

    from dp.common import cleanup_distributed
    from dp.train_workspace import TrainDiffusionWorkspace

    try:
        workspace = TrainDiffusionWorkspace(cfg)
        workspace.run()
    except BaseException as exc:
        rank = os.environ.get("RANK", "0")
        sys.stderr.write(
            f"[rank{rank}] Unhandled exception in train_dp.py: "
            f"{type(exc).__name__}: {exc}\n"
        )
        traceback.print_exc(file=sys.stderr)
        sys.stderr.flush()
        raise
    finally:
        cleanup_distributed()


@record
def main() -> None:
    _hydra_main()


if __name__ == "__main__":
    main()
