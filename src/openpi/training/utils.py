from collections.abc import Callable
from typing import Any

from flax import nnx
from flax import struct
import jax
import numpy as np
import optax

from openpi.models import model as _model
from openpi.shared import array_typing as at


@at.typecheck
@struct.dataclass
class TrainState:
    step: at.Int[at.ArrayLike, ""]
    params: nnx.State
    model_def: nnx.GraphDef[_model.BaseModel]
    opt_state: optax.OptState
    tx: optax.GradientTransformation = struct.field(pytree_node=False)

    ema_decay: float | None = struct.field(pytree_node=False)
    ema_params: nnx.State | None = None


@at.typecheck
def tree_to_info(tree: at.PyTree, interp_func: Callable[[Any], str] = str) -> str:
    """Converts a PyTree into a human-readable string for logging. Optionally, `interp_func` can be provided to convert
    the leaf values to more meaningful strings.
    """
    tree, _ = jax.tree_util.tree_flatten_with_path(tree)
    return "\n".join(f"{jax.tree_util.keystr(path)}: {interp_func(value)}" for path, value in tree)


@at.typecheck
def array_tree_to_info(tree: at.PyTree) -> str:
    """Converts a PyTree of arrays into a human-readable string for logging."""
    return tree_to_info(tree, lambda x: f"{x.shape}@{x.dtype}")


def to_local_array(x):
    """Return a NumPy view/copy of the *process-local* portion of a jax.Array.

    - If the array is fully addressable, device_get the whole thing.
    - Otherwise, concatenate only the addressable shards along leading axis.
    - Avoids blocking on non-addressable globals on CPU/TPU.
    """
    if x is None:
        return None

    # Fast path for plain NumPy or non-JAX types
    if not isinstance(x, jax.Array):
        try:
            return np.asarray(x)
        except Exception:
            return x

    # If everything is local to this process, just transfer once.
    try:
        if getattr(x, "is_fully_addressable", False):
            return np.asarray(x.block_until_ready())
    except Exception:
        pass  # fall through to shard path

    # Otherwise, only bring back this process's shards.
    shards = getattr(x, "addressable_shards", None)
    if shards:
        # Scalar-shard case
        a0 = np.asarray(shards[0].data.block_until_ready())
        if a0.ndim == 0:
            return a0
        parts = [np.asarray(s.data.block_until_ready()) for s in shards]
        return np.concatenate(parts, axis=0)

    # Last resort: return the JAX array itself rather than risking a hang.
    return x


def to_local_scalar(x) -> int:
    """Extract a Python scalar from a possibly-global jax.Array, process-local only."""
    if x is None:
        return 0
    try:
        shards = getattr(x, "addressable_shards", None)
        if shards is not None and len(shards) > 0:
            return int(np.asarray(shards[0].data).item())
    except Exception:
        pass
    return int(np.asarray(x).item())
