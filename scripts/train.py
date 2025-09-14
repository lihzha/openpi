import dataclasses
import functools
import logging
import math
import os
import platform
from typing import Any

import etils.epath as epath
import flax.nnx as nnx
from flax.training import common_utils
import flax.traverse_util as traverse_util
import jax
import jax.numpy as jnp
import numpy as np
import optax
from rail_tpu_utils import prevent_cross_region
import tqdm_loggable.auto as tqdm
import wandb

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils
from openpi.training import eval_helper as _eval_helper
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.optimizer as _optimizer
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils
import openpi.training.weight_loaders as _weight_loaders


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


# ──────────────────────────────────────────────────────────────────────────────
# Validation-time helpers for language actions visualization/metric
# ──────────────────────────────────────────────────────────────────────────────


def init_wandb(
    config: _config.TrainConfig,
    *,
    resuming: bool,
    log_code: bool = False,
    enabled: bool = True,
    rewind_to_step: int | None = None,
):
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
        if rewind_to_step is not None:
            # Use wandb's rewind feature to resume from a specific step
            wandb.init(resume_from=f"{run_id}?_step={rewind_to_step}", project=config.project_name)
        else:
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
        per_sample_loss = model.compute_loss(rng, observation, actions, train=True)
        # If model returns time/horizon dims, reduce to per-example
        while per_sample_loss.ndim > 1:
            per_sample_loss = jnp.mean(per_sample_loss, axis=-1)
        return jnp.mean(per_sample_loss), per_sample_loss

    train_rng = jax.random.fold_in(rng, state.step)
    observation, actions = batch

    # Filter out frozen params.
    diff_state = nnx.DiffState(0, config.trainable_filter)
    (loss, per_sample_loss), grads = nnx.value_and_grad(loss_fn, argnums=diff_state, has_aux=True)(
        model, train_rng, observation, actions
    )

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
        "per_sample_loss": per_sample_loss,
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
        return model.compute_eval_metrics(eval_rng, observation, actions)
    loss = loss_fn(model, eval_rng, observation, actions)
    return {"val_loss": loss}


def main(config: _config.TrainConfig):
    if (
        ("v6" in config.name and config.fsdp_devices > 8)
        or ("v4" in config.name and config.fsdp_devices > 4)
        or ("v5" in config.name and config.fsdp_devices > 8)
    ):
        jax.distributed.initialize()
    data_dir = save_dir = config.data.rlds_data_dir
    cache_dir = os.environ.get("OPENPI_DATA_HOME", None)
    if _is_tpu_runtime() and (str(data_dir).startswith("gs://") or str(save_dir).startswith("gs://")):
        prevent_cross_region(data_dir, save_dir)
        if cache_dir is not None:
            prevent_cross_region(cache_dir, save_dir)
    # Determine effective FSDP devices for single-process GPU/CPU runs.
    process_count = getattr(jax, "process_count", lambda: 1)()
    local_devices = getattr(jax, "local_device_count", lambda: 1)()
    global_devices = getattr(jax, "device_count", lambda: local_devices)()
    init_logging()
    logging.info(f"Local devices: {local_devices}, Global devices: {global_devices}, Process count: {process_count}")
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

    logging.info(f"Running on: {platform.node()}")

    jax.config.update("jax_compilation_cache_dir", str(epath.Path("~/.cache/jax").expanduser()))

    rng = jax.random.key(config.seed)
    train_rng, init_rng = jax.random.split(rng)

    mesh = sharding.make_mesh(effective_fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    # Human-readable mesh overview
    sharding.log_mesh_and_sharding_header(mesh, title="Device mesh")
    logging.info("Data sharding spec: %s", sharding.format_sharding(data_sharding))
    logging.info("Replicated sharding spec: %s", sharding.format_sharding(replicated_sharding))

    checkpoint_manager, resuming = _checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir,
        keep_period=config.keep_period,
        overwrite=config.overwrite,
        resume=config.resume,
    )
    init_wandb(
        config, resuming=resuming, enabled=config.wandb_enabled, rewind_to_step=getattr(config, "rewind_to_step", None)
    )

    data_loader = _data_loader.create_data_loader(
        config,
        sharding=data_sharding,
        shuffle=True,
        seed=config.seed,
    )

    data_iter = iter(data_loader)
    # Fetch the correct first batch, advancing the iterator on resume
    logging.info("Before getting batch")
    # if resuming and start_step > 0:
    #     # Fast-forward the iterator so that step `start_step` uses batch index `start_step`.
    #     for _ in range(start_step):
    #         _ = next(data_iter)
    batch = next(data_iter)
    # images_to_log = [
    #     wandb.Image(
    #         np.concatenate([np.array(_utils.to_local_array(img[i])) for img in batch[0].images.values()], axis=1)
    #     )
    #     for i in range(min(5, len(next(iter(batch[0].images.values())))))
    # ]
    # wandb.log({"camera_views": images_to_log}, step=0)
    logging.info("After getting batch")
    logging.info(f"Initialized data loader (shapes):\n{training_utils.array_tree_to_info(batch)}")
    # Sharding details for the first batch
    sharding.log_batch_sharding(batch)

    train_state, train_state_sharding = init_train_state(config, init_rng, mesh, resume=resuming)
    jax.block_until_ready(train_state)
    logging.info(f"Initialized train state (param shapes):\n{training_utils.array_tree_to_info(train_state.params)}")
    # Planned vs actual parameter sharding
    sharding.log_param_sharding_planned(train_state_sharding)
    sharding.log_param_sharding_actual(train_state.params)

    if resuming:
        train_state = _checkpoints.restore_state(checkpoint_manager, train_state, data_loader)

    ptrain_step = jax.jit(
        functools.partial(train_step, config),
        in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
        out_shardings=(train_state_sharding, replicated_sharding),
        donate_argnums=(1,),
    )

    if config.do_val:
        # Validation data loader (non-shuffled, val split)
        val_loader = _data_loader.create_data_loader(
            config,
            sharding=data_sharding,
            shuffle=False,
            split="val",
            max_samples=getattr(config.data, "val_max_samples", None),
        )
        # Try to obtain the tokenizer from the transform pipeline for decoding
        tok = data_loader._data_loader._dataset._transform.transforms[-1].tokenizer  # type: ignore[attr-defined]
        pval_step = jax.jit(
            functools.partial(val_step, config),
            in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
        )

        # Jitted reasoning sampler returning only (id_buf, t)
        def _sample_reasoning_ids_t(state: training_utils.TrainState, observation: _model.Observation):
            model_local = nnx.merge(state.model_def, state.params)
            id_buf, t, *_ = model_local.sample_reasoning(observation)
            return id_buf, t

        psample_reasoning = jax.jit(
            _sample_reasoning_ids_t,
            # Expect observation replicated; return replicated outputs for consistent host access
            in_shardings=(train_state_sharding, replicated_sharding),
            out_shardings=(replicated_sharding, replicated_sharding),
        )
        # Determine how many validation batches to evaluate each time.
        # If a fixed validation subset size is configured, compute batches from it;
        # otherwise fall back to a heuristic constant divided by global batch size.
        if getattr(config.data, "val_max_samples", None):
            # local batch size per host mirrors RLDS dataset batching
            process_count = getattr(jax, "process_count", lambda: 1)()
            local_bs = max(1, config.batch_size // process_count)
            num_val_batches = math.ceil(config.data.val_max_samples / local_bs)
        else:
            num_val_batches = int(60000 / config.batch_size)  # adjust if needed
    start_step = int(train_state.step)
    pbar = tqdm.tqdm(
        range(start_step, config.num_train_steps),
        initial=start_step,
        total=config.num_train_steps,
        dynamic_ncols=True,
        # disable=(jax.process_index() != 0),
    )

    infos = []
    for step in pbar:
        with sharding.set_mesh(mesh):
            train_state, info = ptrain_step(train_rng, train_state, batch)
        infos.append(info)
        if step % config.log_interval == 0:
            # infos.append(info)
            stacked_infos = common_utils.stack_forest(infos)
            reduced_info = jax.device_get(jax.tree.map(jnp.mean, stacked_infos))
            # Extract forward/backward/opt durations from callbacks
            info_str = ", ".join(f"{k}={v:.4f}" for k, v in reduced_info.items())
            pbar.write(f"Step {step}: {info_str}")
            if jax.process_index() == 0:
                wandb.log(reduced_info, step=step)
                # After warmup, log a few hardest samples with image and language action
                warmup_steps = getattr(config.lr_schedule, "warmup_steps", 0)
                if step >= warmup_steps and "per_sample_loss" in reduced_info:
                    try:
                        # Use the most recent info (non-averaged across the window) for ranking
                        per_ex = info.get("per_sample_loss", None)
                        if per_ex is not None:
                            per_ex_np = np.asarray(training_utils.to_local_array(per_ex))
                            # Threshold: only consider samples > 2x the step's average loss
                            try:
                                step_mean = float(np.asarray(training_utils.to_local_array(info.get("loss", None))))
                            except Exception:
                                step_mean = float(per_ex_np.mean())
                            threshold = 2.0 * step_mean
                            # Prepare images and decoded reasoning
                            first_cam_key = next(iter(batch[0].images))
                            imgs = training_utils.to_local_array(batch[0].images[first_cam_key])
                            imgs_u8 = ((np.asarray(imgs) + 1.0) * 0.5 * 255.0).clip(0, 255).astype(np.uint8)
                            # Decode reasoning if available
                            lang_texts = []
                            try:
                                if (
                                    getattr(batch[0], "tokenized_prompt", None) is not None
                                    and getattr(batch[0], "tokenized_reasoning_mask", None) is not None
                                ):
                                    lang_texts = _eval_helper._decode_reasoning_strings(batch[0], tok)
                            except Exception:
                                lang_texts = []

                            k = int(min(8, per_ex_np.shape[0], imgs_u8.shape[0]))
                            if k > 0:
                                candidates = np.where(per_ex_np > threshold)[0]
                                if candidates.size > 0:
                                    if candidates.size > k:
                                        cand_vals = per_ex_np[candidates]
                                        sel = np.argpartition(-cand_vals, kth=k - 1)[:k]
                                        top_idx = candidates[sel]
                                        order = np.argsort(-per_ex_np[top_idx])
                                        top_idx = top_idx[order]
                                    else:
                                        top_idx = candidates[np.argsort(-per_ex_np[candidates])]
                                else:
                                    top_idx = np.array([], dtype=int)
                                images_to_log = []
                                for i in top_idx:
                                    cap = f"loss={float(per_ex_np[i]):.4f}"
                                    if lang_texts and i < len(lang_texts):
                                        cap = cap + "\n" + str(lang_texts[i])
                                    images_to_log.append(wandb.Image(imgs_u8[i], caption=cap))
                                if images_to_log:
                                    wandb.log({"train/hard_examples": images_to_log}, step=step)
                    except Exception:
                        pass
            infos = []
        # Periodic validation
        if config.do_val and step % getattr(config, "val_interval", 500) == 0:
            # use a pbar to track the validation progress
            val_pbar = tqdm.tqdm(
                range(num_val_batches),
                initial=0,
                total=num_val_batches,
                dynamic_ncols=True,
                disable=(jax.process_index() != 0),
            )
            img_log_step_idx = np.random.randint(0, num_val_batches)
            # img_log_step_idx = 0
            num_images_to_log = 64
            with sharding.set_mesh(mesh):
                val_infos = []
                # Collect L2 distances (in cm) between parsed vectors from GT and predicted texts
                l2_cm_values: list[float] = []
                # Recreate a fresh iterator to ensure the same fixed validation subset each time.
                val_iter = iter(val_loader)
                for val_step_idx in val_pbar:
                    val_batch = next(val_iter)
                    val_info = pval_step(train_rng, train_state, val_batch)
                    val_infos.append(val_info)

                    if config.data.vis_dataset and val_step_idx == img_log_step_idx:
                        k_local = int(min(num_images_to_log, val_batch[0].state.shape[0]))
                        eval_batch = _eval_helper.prepare_eval_batch(val_batch)
                        eval_idx = jax.random.choice(rng, eval_batch[0].state.shape[0], shape=(k_local,), replace=False)
                        eval_batch = _eval_helper.subsample_batch(eval_batch, eval_idx)
                        gt_batch = _eval_helper.subsample_batch(val_batch, eval_idx)
                        # Ensure observation for sampling is replicated across devices
                        obs_local = jax.device_put(eval_batch[0], replicated_sharding)
                        id_buf, t_final = psample_reasoning(train_state, obs_local)
                        if jax.process_index() == 0:
                            l2_cm_values, to_log = _eval_helper.eval_step(gt_batch, id_buf, t_final, tok, k_local)
                        if to_log and jax.process_index() == 0:
                            images_to_log = [wandb.Image(img) for img in to_log]
                            wandb.log({"val/annotated": images_to_log}, step=step)

                stacked_val = common_utils.stack_forest(val_infos)
                reduced_val = jax.device_get(jax.tree.map(jnp.mean, stacked_val))
                # Add movement L2 metric if any collected
                if l2_cm_values:
                    reduced_val = {**reduced_val, "val_movement_l2_cm": float(np.mean(l2_cm_values))}
                val_info_str = ", ".join(f"{k}={v:.4f}" for k, v in reduced_val.items())
                val_pbar.write(f"Step {step} (val): {val_info_str}")
                if jax.process_index() == 0:
                    wandb.log(reduced_val, step=step)

        batch = next(data_iter)

        if (step % config.save_interval == 0 and step > start_step) or step == config.num_train_steps:
            _checkpoints.save_state(checkpoint_manager, train_state, data_loader, step)

    logging.info("Waiting for checkpoint manager to finish")
    checkpoint_manager.wait_until_finished()


if __name__ == "__main__":
    main(_config.cli())
