from __future__ import annotations

import builtins
import logging
import os

import hydra
from omegaconf import DictConfig, OmegaConf

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


@hydra.main(version_base=None, config_path="configs", config_name="dp_config")
def main(cfg: DictConfig) -> None:
    diffusion_policy_src = OmegaConf.select(
        cfg, "runtime.diffusion_policy_src", default=None
    )
    configured_paths = configure_diffusion_policy_path(diffusion_policy_src)
    if configured_paths and _is_main_rank():
        print("Configured diffusion_policy search path:")
        for path in configured_paths:
            print(f"  - {path}")

    OmegaConf.resolve(cfg)

    from dp.train_workspace import TrainDiffusionWorkspace

    workspace = TrainDiffusionWorkspace(cfg)
    workspace.run()


if __name__ == "__main__":
    main()
