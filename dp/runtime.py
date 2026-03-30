from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from typing import Iterable

from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
DIFFUSION_POLICY_ENV_KEYS = (
    "DIFFUSION_POLICY_SRC",
    "TTT_WM_DIFFUSION_POLICY_SRC",
)


def register_omegaconf_resolvers() -> None:
    OmegaConf.register_new_resolver("eval", eval, replace=True)
    OmegaConf.register_new_resolver(
        "if", lambda cond, t, f: t if cond else f, replace=True
    )


def _iter_candidate_paths(explicit_path: str | None) -> Iterable[Path]:
    seen: set[Path] = set()

    candidates = []
    if explicit_path not in (None, ""):
        candidates.append(Path(explicit_path).expanduser())

    for env_key in DIFFUSION_POLICY_ENV_KEYS:
        value = os.environ.get(env_key)
        if value:
            candidates.append(Path(value).expanduser())

    candidates.append(REPO_ROOT / "src" / "diffusion-policy")

    for path in candidates:
        try:
            resolved = path.resolve()
        except FileNotFoundError:
            resolved = path
        if resolved in seen:
            continue
        seen.add(resolved)
        yield resolved


def configure_diffusion_policy_path(explicit_path: str | None = None) -> list[str]:
    inserted_paths: list[str] = []

    for candidate in _iter_candidate_paths(explicit_path):
        if not candidate.exists():
            continue
        candidate_str = str(candidate)
        if candidate_str in sys.path:
            continue
        sys.path.insert(0, candidate_str)
        inserted_paths.append(candidate_str)

    if importlib.util.find_spec("diffusion_policy") is None:
        raise ImportError(
            "Could not import `diffusion_policy`. Install it with "
            "`-e git+https://github.com/real-stanford/diffusion_policy.git"
            "@5ba07ac6661db573af695b419a7947ecb704690f#egg=diffusion_policy` "
            "and either set `PYTHONPATH=/path/to/diffusion-policy:$PYTHONPATH` "
            "or export `DIFFUSION_POLICY_SRC=/path/to/diffusion-policy`."
        )

    return inserted_paths

