import dataclasses
import functools
import hashlib
import logging
import os
import platform
from typing import Any

import etils.epath as epath
import flax.nnx as nnx
from flax.training import common_utils
import flax.traverse_util as traverse_util
import jax
import jax.experimental
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
from rail_tpu_utils import prevent_cross_region
import tqdm_loggable.auto as tqdm
import wandb

import openpi.models.model as _model
import openpi.models.tokenizer as _tokenizer
import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.optimizer as _optimizer
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils
import openpi.training.weight_loaders as _weight_loaders
import openpi.transforms as _transforms


def init_logging():
    """Custom logging format for better readability."""
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers[0].setFormatter(formatter)


def _format_sharding(shard) -> str:
    try:
        import jax
    except Exception:
        return "<no-jax>"
    if isinstance(shard, jax.sharding.NamedSharding):
        mesh = shard.mesh
        mesh_desc = ", ".join(f"{k}={v}" for k, v in mesh.shape.items())
        return f"NamedSharding(mesh=[{mesh_desc}], spec={shard.spec})"
    if hasattr(shard, "devices"):
        # PositionalSharding and others expose .devices()
        try:
            ndev = len(shard.devices())
        except Exception:
            ndev = "?"
        return f"{type(shard).__name__}(devices={ndev})"
    return str(shard)


def _get_array(obj):
    # nnx.Param-like leaves store the array in .value
    if hasattr(obj, "value") and hasattr(obj.value, "sharding"):
        return obj.value
    return obj


def _pytree_array_leaves(tree):
    leaves = []
    for path, leaf in jax.tree_util.tree_flatten_with_path(tree)[0]:
        arr = _get_array(leaf)
        if hasattr(arr, "shape") and hasattr(arr, "sharding"):
            leaves.append((path, arr))
    return leaves


def log_mesh_and_sharding_header(mesh: jax.sharding.Mesh, *, title: str):
    mesh_desc = ", ".join(f"{k}={v}" for k, v in mesh.shape.items())
    try:
        import numpy as _np

        total = int(_np.prod(list(mesh.shape.values())))
    except Exception:
        total = "?"
    logging.info(f"{title}: mesh axes [{mesh_desc}] total_devices={total}")


def log_batch_sharding(batch):
    def fmt_path(path):
        return jax.tree_util.keystr(path)

    lines = []
    for path, arr in _pytree_array_leaves(batch):
        try:
            ex_shape = None
            # Example addressable shard shape on this host (if available)
            if hasattr(arr, "addressable_shards") and arr.addressable_shards:
                ex_shape = arr.addressable_shards[0].data.shape
            shard_str = _format_sharding(arr.sharding)
            line = f"{fmt_path(path)}: global={tuple(arr.shape)} dtype={arr.dtype} | {shard_str}"
            if ex_shape is not None:
                line += f" | local_shard={tuple(ex_shape)}"
            lines.append(line)
        except Exception as e:
            lines.append(f"{fmt_path(path)}: <error formatting sharding: {e}>")
    if lines:
        logging.info("Batch sharding summary:\n" + "\n".join(lines))


def log_param_sharding_planned(state_sharding):
    from openpi.training import sharding as _sh

    planned = state_sharding.params
    entries = []
    sharded = replicated = 0
    for path, shard in jax.tree_util.tree_flatten_with_path(planned)[0]:
        if isinstance(shard, jax.sharding.NamedSharding):
            # Count as sharded if any dim uses FSDP axis
            uses_fsdp = False
            try:
                spec = shard.spec
                # spec is a PartitionSpec; check members for axis name
                if isinstance(spec, jax.sharding.PartitionSpec):
                    uses_fsdp = _sh.FSDP_AXIS in tuple(spec)
            except Exception:
                pass
            if uses_fsdp:
                sharded += 1
            else:
                replicated += 1
        else:
            replicated += 1
        entries.append(f"{jax.tree_util.keystr(path)}: {_format_sharding(shard)}")
    logging.info(
        "Planned parameter sharding (from fsdp_sharding): sharded=%d replicated=%d\n%s",
        sharded,
        replicated,
        "\n".join(entries),
    )


def log_param_sharding_actual(params):
    lines = []
    for path, arr in _pytree_array_leaves(params):
        try:
            ex_shape = None
            if hasattr(arr, "addressable_shards") and arr.addressable_shards:
                ex_shape = arr.addressable_shards[0].data.shape
            shard_str = _format_sharding(arr.sharding)
            line = f"{jax.tree_util.keystr(path)}: global={tuple(arr.shape)} dtype={arr.dtype} | {shard_str}"
            if ex_shape is not None:
                line += f" | local_shard={tuple(ex_shape)}"
            lines.append(line)
        except Exception as e:
            lines.append(f"{jax.tree_util.keystr(path)}: <error formatting sharding: {e}>")
    if lines:
        logging.info("Actual parameter sharding:\n" + "\n".join(lines))


def init_wandb(config: _config.TrainConfig, *, resuming: bool, log_code: bool = False, enabled: bool = True):
    if not enabled:
        wandb.init(mode="disabled")
        return

    # Only initialize wandb in the main process
    if jax.process_index() != 0:
        wandb.init(mode="disabled")
        return

    ckpt_dir = config.checkpoint_dir
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist.")
    if resuming:
        run_id = (ckpt_dir / "wandb_id.txt").read_text().strip()
        wandb.init(id=run_id, resume="must", project=config.project_name)
    else:
        wandb.init(
            name=config.exp_name,
            config=dataclasses.asdict(config),
            project=config.project_name,
        )
        (ckpt_dir / "wandb_id.txt").write_text(wandb.run.id)

    if log_code:
        wandb.run.log_code(epath.Path(__file__).parent.parent)


def _maybe_initialize_jax_distributed():
    """Initialize JAX distributed only when multi-process is configured.

    This allows single-process GPU/CPU runs without requiring coordination service.
    """
    # Already initialized → nothing to do
    try:
        if getattr(jax.distributed, "is_initialized", lambda: False)():
            logging.info("JAX distributed runtime already initialized.")
            return
        logging.info("JAX distributed runtime not yet initialized.")
    except Exception:
        # Older JAX versions may not have is_initialized
        logging.info("JAX distributed runtime not yet initialized.")
        pass

    env = os.environ
    should_init = False
    # Common envs signaling multi-process setups
    if env.get("JAX_COORDINATION_SERVICE_ADDR"):
        should_init = True
    try:
        if int(getattr(jax, "process_count", lambda: 1)()) > 1:
            should_init = True
        if int(env.get("COORDINATOR_NUM_PROCESSES", "1")) > 1:
            should_init = True
    except Exception:
        pass

    if not should_init:
        logging.info("Single-process run detected; skipping jax.distributed.initialize().")
        return

    try:
        jax.distributed.initialize()
        logging.info("Initialized JAX distributed runtime.")
    except Exception as e:
        logging.info("Failed to initialize jax.distributed (continuing single-process): %s", e)


def break_into_single_batches(batch: tuple[_model.Observation, _model.Actions]) -> list[tuple[_model.Observation, _model.Actions]]:
    """Break down a batch into individual single-item batches.
    
    Args:
        batch: A tuple of (Observation, Actions) where both have a leading batch dimension
        
    Returns:
        A list of single-item batches, each with batch size 1
    """
    observation, actions = batch
    
    # Get the batch size from the observation state (or any other field)
    batch_size = observation.state.shape[0]
    
    single_batches = []
    for i in range(batch_size):
        # Extract single item from observation
        single_obs = jax.tree.map(lambda x: x[i:i+1], observation)
        
        # Extract single item from actions
        single_actions = actions[i:i+1]
        
        single_batches.append((single_obs, single_actions))
    
    return single_batches


def _is_tpu_runtime() -> bool:
    try:
        return any(d.platform == "tpu" for d in jax.devices())
    except Exception:
        return False


def _load_weights_and_validate(loader: _weight_loaders.WeightLoader, params_shape: at.Params) -> at.Params:
    """Loads and validates the weights. Returns a loaded subset of the weights."""
    loaded_params = loader.load(params_shape)
    at.check_pytree_equality(expected=params_shape, got=loaded_params, check_shapes=True, check_dtypes=True)

    # Remove jax.ShapeDtypeStruct from the loaded params. This makes sure that only the loaded params are returned.
    return traverse_util.unflatten_dict(
        {k: v for k, v in traverse_util.flatten_dict(loaded_params).items() if not isinstance(v, jax.ShapeDtypeStruct)}
    )


@at.typecheck
def init_train_state(
    config: _config.TrainConfig, init_rng: at.KeyArrayLike, mesh: jax.sharding.Mesh, *, resume: bool
) -> tuple[training_utils.TrainState, Any]:
    tx = _optimizer.create_optimizer(config.optimizer, config.lr_schedule, weight_decay_mask=None)

    def init(rng: at.KeyArrayLike, partial_params: at.Params | None = None) -> training_utils.TrainState:
        rng, model_rng = jax.random.split(rng)
        # initialize the model (and its parameters).
        model = config.model.create(model_rng)

        # Merge the partial params into the model.
        if partial_params is not None:
            graphdef, state = nnx.split(model)
            # This will produce an error if the partial params are not a subset of the state.
            state.replace_by_pure_dict(partial_params)
            model = nnx.merge(graphdef, state)

        params = nnx.state(model)
        # Convert frozen params to bfloat16.
        params = nnx_utils.state_map(params, config.freeze_filter, lambda p: p.replace(p.value.astype(jnp.bfloat16)))

        return training_utils.TrainState(
            step=0,
            params=params,
            model_def=nnx.graphdef(model),
            tx=tx,
            opt_state=tx.init(params.filter(config.trainable_filter)),
            ema_decay=config.ema_decay,
            ema_params=None if config.ema_decay is None else params,
        )

    train_state_shape = jax.eval_shape(init, init_rng)
    state_sharding = sharding.fsdp_sharding(train_state_shape, mesh, log=True)

    if resume:
        return train_state_shape, state_sharding

    partial_params = _load_weights_and_validate(config.weight_loader, train_state_shape.params.to_pure_dict())
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    # Initialize the train state and mix in the partial params.
    train_state = jax.jit(
        init,
        donate_argnums=(1,),  # donate the partial params buffer.
        in_shardings=replicated_sharding,
        out_shardings=state_sharding,
    )(init_rng, partial_params)

    return train_state, state_sharding


@at.typecheck
def train_step(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions],
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    model = nnx.merge(state.model_def, state.params)
    model.train()

    @at.typecheck
    def loss_fn(
        model: _model.BaseModel, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions
    ):
        chunked_loss = model.compute_loss(rng, observation, actions, train=True)
        return jnp.mean(chunked_loss)

    train_rng = jax.random.fold_in(rng, state.step)
    observation, actions = batch

    # Filter out frozen params.
    diff_state = nnx.DiffState(0, config.trainable_filter)
    loss, grads = nnx.value_and_grad(loss_fn, argnums=diff_state)(model, train_rng, observation, actions)

    params = state.params.filter(config.trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)

    # Update the model in place and return the new full state.
    nnx.update(model, new_params)
    new_params = nnx.state(model)

    new_state = dataclasses.replace(state, step=state.step + 1, params=new_params, opt_state=new_opt_state)
    if state.ema_decay is not None:
        new_state = dataclasses.replace(
            new_state,
            ema_params=jax.tree.map(
                lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new, state.ema_params, new_params
            ),
        )

    # Filter out params that aren't kernels.
    kernel_params = nnx.state(
        model,
        nnx.All(
            nnx.Param,
            nnx.Not(nnx_utils.PathRegex(".*/(bias|scale|pos_embedding|input_embedding)")),
            lambda _, x: x.value.ndim > 1,
        ),
    )
    info = {
        "loss": loss,
        "grad_norm": optax.global_norm(grads),
        "param_norm": optax.global_norm(kernel_params),
    }
    return new_state, info


@at.typecheck
def val_step(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions],
) -> dict[str, at.Array]:
    model = nnx.merge(state.model_def, state.params)
    model.eval()

    @at.typecheck
    def loss_fn(
        model: _model.BaseModel, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions
    ):
        # compute_loss may return per-example; reduce to scalar
        val_loss = model.compute_loss(rng, observation, actions, train=False)
        return jnp.mean(val_loss)

    eval_rng = jax.random.fold_in(rng, state.step)
    observation, actions = batch
    if hasattr(model, "compute_eval_metrics"):
        # type: ignore[attr-defined]
        metrics = model.compute_eval_metrics(eval_rng, observation, actions)  # pyright: ignore
        return metrics
    loss = loss_fn(model, eval_rng, observation, actions)
    return {"val_loss": loss}


@at.typecheck
def eval_step(
    tok: _tokenizer.PaligemmaTokenizer,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions],
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    model = nnx.merge(state.model_def, state.params)
    model.eval()
    observation = batch[0]
    logits, t, _, _, _ = model.sample_reasoning(observation)
    logging.info(f"Logits: {logits.shape}, t: {jax.device_get(t)}")
    tokenized_reasoning = tok.decode(jax.device_get(logits).squeeze()[:jax.device_get(t)])
    return tokenized_reasoning


def main(config: _config.TrainConfig):
    _maybe_initialize_jax_distributed()
    data_dir = save_dir = config.data.rlds_data_dir
    if _is_tpu_runtime() and (str(data_dir).startswith("gs://") or str(save_dir).startswith("gs://")):
        prevent_cross_region(data_dir, save_dir)

    # Determine effective FSDP devices for single-process GPU/CPU runs.
    process_count = getattr(jax, "process_count", lambda: 1)()
    local_devices = getattr(jax, "local_device_count", lambda: 1)()
    global_devices = getattr(jax, "device_count", lambda: local_devices)()
    if process_count == 1:
        # Choose the largest divisor of available devices not exceeding configured fsdp_devices
        target = min(config.fsdp_devices, max(1, local_devices))
        effective_fsdp_devices = 1
        for d in range(target, 0, -1):
            if global_devices % d == 0:
                effective_fsdp_devices = d
                break
        if effective_fsdp_devices != config.fsdp_devices:
            logging.info(
                "Using fsdp_devices=%d for single-process run (available devices=%d)",
                effective_fsdp_devices,
                global_devices,
            )
    else:
        effective_fsdp_devices = config.fsdp_devices
        assert global_devices % effective_fsdp_devices == 0
    # Prefer intra-host FSDP when single process.
    # assert jax.local_device_count() % effective_fsdp_devices == 0

    init_logging()
    logging.info(
        f"Summation steps: {config.data.summation_steps}, left_pad: {config.data.left_pad}, sum_decimal: {config.data.sum_decimal}, ema_decay: {config.ema_decay}"
    )
    logging.info(f"Running on: {platform.node()}")

    # if config.batch_size % jax.device_count() != 0:
    #     raise ValueError(
    #         f"Batch size {config.batch_size} must be divisible by the number of devices {jax.device_count()}."
    #     )

    jax.config.update("jax_compilation_cache_dir", str(epath.Path("~/.cache/jax").expanduser()))

    rng = jax.random.key(config.seed)
    train_rng, init_rng = jax.random.split(rng)

    mesh = sharding.make_mesh(effective_fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    # Human-readable mesh overview
    log_mesh_and_sharding_header(mesh, title="Device mesh")
    logging.info("Data sharding spec: %s", _format_sharding(data_sharding))
    logging.info("Replicated sharding spec: %s", _format_sharding(replicated_sharding))

    data_loader = _data_loader.create_data_loader(
        config,
        sharding=data_sharding,
        shuffle=True,
        seed=config.seed,
    )
    data_iter = iter(data_loader)

    do_val = False
    do_eval = True
    if do_val:
        # Validation data loader (non-shuffled, val split)
        val_loader = _data_loader.create_data_loader(
            config,
            sharding=data_sharding,
            shuffle=False,
            split="val",
        )
        val_iter = iter(val_loader)
    # Defer fetching the first batch until after (potential) checkpoint restore
    # so we can fast-forward the iterator on resume for deterministic continuity.

    # # Optional sanity check: exhaust data_iter until a repeated sample is seen
    # # when training with a capped sample set (e.g., max_samples in RLDS CoT).
    tok = data_loader._data_loader._dataset._transform.transforms[-1].tokenizer
    # max_samples_cfg = getattr(config.data, "max_samples", None)
    # logging.info("Running capped-samples sanity check (expect repeat after ~%s samples)", max_samples_cfg)
    # seen = set()
    # total = 0
    # repeated = False
    # test_iter = iter(data_loader)
    # while not repeated:
    #     test_batch = next(test_iter)
    #     # test_batch is (Observation, Actions)
    #     obs = test_batch[0]
    #     lang_actions_encoded = obs.tokenized_prompt
    #     # Use the first available camera stream as a stable fingerprint basis
    #     first_cam = next(iter(obs.images.values()))
    #     B = first_cam.shape[0]
    #     for i in range(B):
    #         img_bytes = bytes(memoryview(jax.device_get(first_cam[i]).astype("uint8").tobytes()))
    #         h = hashlib.sha1(img_bytes).hexdigest()
    #         if h in seen:
    #             repeated = True
    #             break
    #         lang_action = jax.device_get(tok.decode(lang_actions_encoded[i]))
    #         images = jax.device_get(first_cam[i])
    #         images = (images + 1) / 2
    #         # img0 = images[0]
    #         # img1 = images[-1]
    #         img0 = images
    #         img1 = images

    #         fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    #         axes[0].imshow(img0)
    #         axes[0].axis("off")
    #         axes[0].set_title("t=0s")
    #         # Write language action on t=0s image
    #         _action_text = str(lang_action)
    #         axes[0].text(
    #             0.01,
    #             0.99,
    #             _action_text,
    #             transform=axes[0].transAxes,
    #             va="top",
    #             ha="left",
    #             fontsize=10,
    #             color="white",
    #             bbox=dict(facecolor="black", alpha=0.5, pad=3),
    #         )
    #         axes[1].imshow(img1)
    #         axes[1].axis("off")
    #         axes[1].set_title("t≈+1s")
    #         plt.suptitle("Initial vs +1s")
    #         plt.savefig(f"initial_vs_1s_{total}.png")
    #         seen.add(h)
    #         total += 1
    # logging.info("Capped-samples sanity: unique before repeat=%d (configured max_samples=%s)", total, max_samples_cfg)

    # # save the hash list to a file, which is easliy loaded in the future
    # with open("hash4.txt", "w") as f:
    #     for h in seen:
    #         f.write(h + "\n")

    checkpoint_manager, resuming = _checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir,
        keep_period=config.keep_period,
        overwrite=config.overwrite,
        resume=config.resume,
    )
    init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)

    train_state, train_state_sharding = init_train_state(config, init_rng, mesh, resume=resuming)
    jax.block_until_ready(train_state)
    logging.info(f"Initialized train state (param shapes):\n{training_utils.array_tree_to_info(train_state.params)}")
    # Planned vs actual parameter sharding
    log_param_sharding_planned(train_state_sharding)
    log_param_sharding_actual(train_state.params)
    if resuming:
        train_state = _checkpoints.restore_state(checkpoint_manager, train_state, data_loader)

    ptrain_step = jax.jit(
        functools.partial(train_step, config),
        in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
        out_shardings=(train_state_sharding, replicated_sharding),
        donate_argnums=(1,),
    )

    if do_val:
        pval_step = jax.jit(
            functools.partial(val_step, config),
            in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
        )
    if do_eval:
        peval_step = jax.jit(
            functools.partial(eval_step, tok),
        )

    # Fetch the correct first batch, advancing the iterator on resume
    start_step = int(train_state.step)
    logging.info("Before getting batch (start_step=%d, resuming=%s)", start_step, resuming)
    # if resuming and start_step > 0:
    #     # Fast-forward the iterator so that step `start_step` uses batch index `start_step`.
    #     for _ in range(start_step):
    #         _ = next(data_iter)
    batch = next(data_iter)
    logging.info("After getting batch")
    logging.info(f"Initialized data loader (shapes):\n{training_utils.array_tree_to_info(batch)}")
    logging.info(f"Batch[0].tokenized_prompt: {batch[0].tokenized_prompt}")
    # Sharding details for the first batch
    log_batch_sharding(batch)

    # Log images from first batch to sanity check.
    try:
        if jax.process_index() == 0:
            images_to_log = [
                wandb.Image(np.concatenate([np.array(img[i]) for img in batch[0].images.values()], axis=1))
                for i in range(min(5, len(next(iter(batch[0].images.values())))))
            ]
            wandb.log({"camera_views": images_to_log}, step=0)
    except Exception:
        pass

    # Optional: warm up eval after batch is available
    if do_val:
        with sharding.set_mesh(mesh):
            _warm_val = pval_step(train_rng, train_state, batch)
        # Block on one leaf to ensure compile completes before timing-sensitive loops
        try:
            jax.tree_util.tree_leaves(_warm_val)[0].block_until_ready()
        except Exception:
            pass
    # if do_eval:
    #     with sharding.set_mesh(mesh):
    #         _warm_eval = peval_step(train_state, batch)
    #     # Block on one leaf to ensure compile completes before timing-sensitive loops
    #     try:
    #         jax.tree_util.tree_leaves(_warm_eval)[0].block_until_ready()
    #     except Exception:
    #         pass
    pbar = tqdm.tqdm(
        range(start_step, config.num_train_steps),
        initial=start_step,
        total=config.num_train_steps,
        dynamic_ncols=True,
    )

    infos = []
    train_batches = []
    seen = set()
    num_val_batches = getattr(config, "num_val_batches", 400)
    for step in pbar:
        with sharding.set_mesh(mesh):
            train_state, info = ptrain_step(train_rng, train_state, batch)
        obs = batch[0]
        single_batches = break_into_single_batches(batch)
        for single_batch in single_batches:
            first_cam = next(iter(single_batch[0].images.values()))
            B = first_cam.shape[0]
            assert B == 1
            for i in range(B):
                img_bytes = bytes(memoryview(jax.device_get(first_cam[i]).astype("uint8").tobytes()))
                h = hashlib.sha1(img_bytes).hexdigest()
                if h not in seen:
                    train_batches.append(single_batch)
                    seen.add(h)
            logging.info(f"Seen: {len(seen)}")
            if len(seen) == config.data.max_samples:
                break
        infos.append(info)
        stacked_infos = common_utils.stack_forest(infos)
        reduced_info = jax.device_get(jax.tree.map(jnp.mean, stacked_infos))
        info_str = ", ".join(f"{k}={v:.4f}" for k, v in reduced_info.items())
        logging.info(f"Step {step}: {info_str}")
        infos = []

        if step % config.log_interval == 0:
            infos.append(info)
            stacked_infos = common_utils.stack_forest(infos)
            reduced_info = jax.device_get(jax.tree.map(jnp.mean, stacked_infos))
            info_str = ", ".join(f"{k}={v:.4f}" for k, v in reduced_info.items())
            pbar.write(f"Step {step}: {info_str}")
            logging.info(f"Step {step}: {info_str}")
            if jax.process_index() == 0:
                wandb.log(reduced_info, step=step)
            infos = []
        # Periodic validation
        if do_val and step % getattr(config, "val_interval", 5000) == 0:
            # use a pbar to track the validation progress
            val_pbar = tqdm.tqdm(
                range(num_val_batches),
                initial=0,
                total=num_val_batches,
                dynamic_ncols=True,
            )
            with sharding.set_mesh(mesh):
                val_infos = []
                for _ in val_pbar:
                    val_batch = next(val_iter)
                    val_info = pval_step(train_rng, train_state, val_batch)
                    val_infos.append(val_info)
                stacked_val = common_utils.stack_forest(val_infos)
                reduced_val = jax.device_get(jax.tree.map(jnp.mean, stacked_val))
                val_pbar.write(
                    "Step %d (val): %s"
                    % (
                        step,
                        ", ".join(f"{k}={v:.4f}" for k, v in reduced_val.items()),
                    )
                )
                if jax.process_index() == 0:
                    wandb.log({**reduced_val, "split": "val"}, step=step)
        if do_eval and len(seen) == 150:
            with sharding.set_mesh(mesh):
                for batch in train_batches:
                    reasoning = peval_step(train_state, batch)
                    gt = tok.decode(batch[0].tokenized_prompt)
                    logging.info(f"GT: {gt}")
                    logging.info(f"Pred: {reasoning}")
        batch = next(data_iter)

        if (step % config.save_interval == 0 and step > start_step) or step == config.num_train_steps - 1:
            _checkpoints.save_state(checkpoint_manager, train_state, data_loader, step)

    logging.info("Waiting for checkpoint manager to finish")
    checkpoint_manager.wait_until_finished()


if __name__ == "__main__":
    main(_config.cli())
