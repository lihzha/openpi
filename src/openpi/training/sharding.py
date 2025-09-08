import contextlib
import logging

import jax
from jax.experimental import mesh_utils
import numpy as np

BATCH_AXIS = "data"
FSDP_AXIS = "fsdp"
# In FSDP, we shard the data across both the batch and FSDP axes.
DATA_AXIS = (BATCH_AXIS, FSDP_AXIS)


class _MeshState:
    active_mesh: jax.sharding.Mesh | None = None


def make_mesh(fsdp_devices: int) -> jax.sharding.Mesh:
    P = jax.process_count()  # number of hosts
    D = jax.local_device_count()  # devices per host
    N = jax.device_count()  # total devices (P * D)

    if N % fsdp_devices != 0:
        raise ValueError(f"Total devices {N} must be divisible by fsdp_devices {fsdp_devices}.")

    # Host-major device layout: shape [P, D] with each row = one host's devices.
    # This has no "data/model" meaning by itself; it's just a physical arrangement.
    devmesh = mesh_utils.create_device_mesh((P, D))  # shape (P, D)

    if fsdp_devices <= D:
        # Intra-host FSDP: split each host's devices into [dp_per_host, fsdp_devices]
        if D % fsdp_devices != 0:
            raise ValueError(f"local_device_count {D} not divisible by fsdp_devices {fsdp_devices}")
        dp_per_host = D // fsdp_devices
        # Final mesh: collapse hosts and dp_per_host into one DATA axis; FSDP axis is local.
        # Shape: (P * dp_per_host, fsdp_devices)
        devmesh = devmesh.reshape(P * dp_per_host, fsdp_devices)

    else:
        # Cross-host FSDP: group whole hosts along FSDP axis.
        # Require FSDP groups to be a multiple of per-host devices.
        if fsdp_devices % D != 0:
            raise ValueError(
                f"When fsdp_devices > local_device_count, fsdp_devices ({fsdp_devices}) "
                f"must be a multiple of local_device_count ({D}) to group whole hosts."
            )
        fsdp_hosts = fsdp_devices // D
        if P % fsdp_hosts != 0:
            raise ValueError(
                f"process_count {P} must be divisible by fsdp_hosts {fsdp_hosts} (= fsdp_devices/local_device_count)."
            )
        dp_groups = P // fsdp_hosts

        # Special case: when fsdp_devices equals total devices, we want pure FSDP
        # with no data parallelism across hosts
        if fsdp_devices == N:
            # All devices go to FSDP, no data parallelism across hosts
            # Shape: (1, fsdp_devices) - single data parallel group, all devices for FSDP
            devmesh = devmesh.reshape(1, fsdp_devices)
        else:
            # Combine fsdp_hosts hosts along FSDP axis (each host contributes D devices).
            # Shape: (dp_groups, fsdp_hosts * D) = (dp_groups, fsdp_devices)
            devmesh = devmesh.reshape(dp_groups, fsdp_hosts * D)

    # 2D logical mesh: first axis used for data-parallel sharding, second for FSDP
    return jax.sharding.Mesh(devmesh, (BATCH_AXIS, FSDP_AXIS))


# def make_mesh(num_fsdp_devices: int) -> jax.sharding.Mesh:
#     if jax.device_count() % num_fsdp_devices != 0:
#         raise ValueError(
#             f"Number of devices {jax.device_count()} must be divisible by the number of FSDP devices {num_fsdp_devices}."
#         )
#     mesh_shape = (jax.device_count() // num_fsdp_devices, num_fsdp_devices)
#     return jax.make_mesh(mesh_shape, (BATCH_AXIS, FSDP_AXIS))


@contextlib.contextmanager
def set_mesh(mesh: jax.sharding.Mesh):
    """Plumbing the mesh deep into the module tree is extremeley cumbersome; until the JAX team lands a better API, a
    custom context manager like this one is the recommended way to maintain a reference to a global mesh. This is only used
    in `activation_sharding_constraint` below."""
    if _MeshState.active_mesh is not None:
        raise ValueError("Cannot nest set_mesh context managers.")
    _MeshState.active_mesh = mesh
    try:
        yield
    finally:
        _MeshState.active_mesh = None


def activation_sharding_constraint(pytree):
    if _MeshState.active_mesh is None:
        return pytree
    return jax.lax.with_sharding_constraint(
        pytree, jax.sharding.NamedSharding(_MeshState.active_mesh, jax.sharding.PartitionSpec(DATA_AXIS))
    )


def fsdp_sharding(
    pytree,
    mesh: jax.sharding.Mesh,
    *,
    min_size_mbytes: int = 4,  # 4 MiB
    log: bool = False,
):
    """Apply FSDP sharding to a pytree of arrays based on the mesh shape.

    Args:
        pytree: A pytree to be apply sharding specified by the mesh, note that only array types (eg. contains .shape attr)
          will be considered for sharding.
        mesh: The mesh being used for applying sharding on to pytree.
        min_size_mbytes: The minimum size of the array in MiB to be considered for sharding, any array smaller than this
          will be replicated.
        log: If true, will log the sharding decisions for arrays that are being considered for sharding.

    Returns:
        The sharded pytree.
    """
    min_size_bytes = min_size_mbytes * 2**20

    def _shard_arr(kp, array: jax.ShapeDtypeStruct):
        # if fsdp is not actually going to be used, replicate everything to avoid extraneous logging
        if mesh.shape[FSDP_AXIS] == 1:
            return jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
        # replicate scalar and vector arrays
        if not hasattr(array, "shape"):
            return jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
        if len(array.shape) < 2:
            return jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
        # replicate small arrays
        if (arr_size := np.prod(array.shape) * np.dtype(array.dtype).itemsize) < min_size_bytes:
            return jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

        # shard matrices and larger tensors along the largest axis that is divisible by the fsdp dimension
        axes = np.argsort(array.shape)[::-1]
        spec = [None] * len(axes)
        for i in axes:
            if array.shape[i] % mesh.shape[FSDP_AXIS] == 0:
                if log:
                    logging.info(
                        f"Sharding {jax.tree_util.keystr(kp)} of shape {array.shape} ({arr_size / 2**20:.2f} MiB) along axis {i}"
                    )
                spec[i] = FSDP_AXIS
                return jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(*spec))

        # replicate if no valid sharding was found
        if log:
            logging.warning(
                f"Could not find a valid sharding for {jax.tree_util.keystr(kp)} of shape {array.shape} with mesh of shape {mesh.shape}"
            )
        return jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    return jax.tree_util.tree_map_with_path(_shard_arr, pytree)
