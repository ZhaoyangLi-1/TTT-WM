"""
Minimal compatibility shim for Python 3.8 + jaxtyping 0.2.19.

This module originally used jaxtyping's runtime shape/dtype checking and
various typing features (TypeAlias, PEP 604 union syntax, jaxtyping.config,
Num/Real dtype markers) that are only available in Python >= 3.10 and
jaxtyping >= 0.2.20. None of those are installable on Python 3.8, so we
degrade all type annotations to `Any` and turn the runtime type-checking
decorator into a pass-through. Behaviorally neutral for inference/eval code.
"""
import contextlib
from typing import Any, TypeVar, Union, cast
from typing_extensions import TypeAlias

import jax
import jax._src.tree_util as private_tree_util
import jax.core
import torch

# --- jaxtyping symbols: degrade to Any ------------------------------------
# These are used as type hints only; at runtime they act like `typing.Any`,
# so any subscription like `Float[Array, "b c"]` evaluates to `Any` and is a no-op.
ArrayLike = Any
Bool = Any
DTypeLike = Any
Float = Any
Int = Any
Key = Any
Num = Any
PyTree = Any
Real = Any
UInt8 = Any

# --- Core type aliases -----------------------------------------------------
Array = Union[jax.Array, torch.Tensor]
KeyArrayLike: TypeAlias = jax.typing.ArrayLike
Params: TypeAlias = Any

T = TypeVar("T")


# --- Decorators / context managers (no-ops under this shim) ---------------
def typecheck(t: T) -> T:
    """No-op replacement for the jaxtyping+beartype runtime typecheck decorator."""
    return t


@contextlib.contextmanager
def disable_typechecking():
    """No-op: jaxtyping 0.2.19 has no `config` module and does no runtime checks by default."""
    yield


# --- Real helper that callers actually rely on ----------------------------
def check_pytree_equality(
    *,
    expected: Any,
    got: Any,
    check_shapes: bool = False,
    check_dtypes: bool = False,
):
    """Checks that two PyTrees have the same structure and optionally shapes/dtypes."""
    errors = list(private_tree_util.equality_errors(expected, got))
    if errors:
        raise ValueError(
            "PyTrees have different structure:\n"
            + "\n".join(
                f"   - at keypath '{jax.tree_util.keystr(path)}': expected {thing1}, got {thing2}, so {explanation}.\n"
                for path, thing1, thing2, explanation in errors
            )
        )

    if check_shapes or check_dtypes:
        def check(kp, x, y):
            if check_shapes and x.shape != y.shape:
                raise ValueError(
                    f"Shape mismatch at {jax.tree_util.keystr(kp)}: expected {x.shape}, got {y.shape}"
                )
            if check_dtypes and x.dtype != y.dtype:
                raise ValueError(
                    f"Dtype mismatch at {jax.tree_util.keystr(kp)}: expected {x.dtype}, got {y.dtype}"
                )

        jax.tree_util.tree_map_with_path(check, expected, got)