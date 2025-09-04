#!/usr/bin/env python3
import os


def main():
    # 1. Parse rank from hostname (e.g., "tpu-name-worker-0")

    # 2. Read coordinator info from env or defaults
    coord = (
        os.environ.get("TPU_MESH_CONTROLLER_ADDRESS", "127.0.0.1")
        + ":"
        + os.environ.get("TPU_MESH_CONTROLLER_PORT", "8476")
    )
    print(coord)
    import jax

    # 3. Initialize JAX distributed on 2 hosts only
    jax.distributed.initialize(
        # coordinator_address=coord,
        # num_processes=2,
    )

    # 4. Print device info
    print(f"Local: {jax.local_device_count()} P: {jax.process_count()} i: {jax.process_index()} Devs: {jax.devices()}")


if __name__ == "__main__":
    main()
