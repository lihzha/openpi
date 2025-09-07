import dataclasses
import functools
import logging
import math
import os
import platform
import re
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


# ──────────────────────────────────────────────────────────────────────────────
# Validation-time helpers for language actions visualization/metric
# ──────────────────────────────────────────────────────────────────────────────


def _decode_reasoning_strings(obs: _model.Observation, tokenizer) -> list[str]:
    """Extract and decode the reasoning (language action) tokens per example.

    Returns one decoded string per example. If reasoning fields are absent, returns [].
    """
    if obs.tokenized_prompt is None or obs.tokenized_reasoning_mask is None:
        return []
    tokens = _to_local_array(obs.tokenized_prompt)
    rmask = _to_local_array(obs.tokenized_reasoning_mask)
    out: list[str] = []
    for i in range(tokens.shape[0]):
        sel = tokens[i][rmask[i].astype(bool)]
        try:
            text = tokenizer.decode(sel.astype(np.int32))
        except Exception:
            text = ""
        out.append(text)
    return out


def _parse_language_delta_cm(text: str) -> np.ndarray | None:
    """Parse summed language action text -> net [right, forward, down] in cm.

    Accepts parts joined by " and ", like: "move right 10.3cm and move up 1.2cm and move forward 1.35cm".
    Recognized directions: left/right, forward/backward, up/down. Units: mm, cm, m.
    Returns None if no valid movements found.
    """
    if text is None:
        return None
    totals = {"left": 0.0, "right": 0.0, "forward": 0.0, "backward": 0.0, "up": 0.0, "down": 0.0}
    any_valid = False
    for part in filter(None, [p.strip() for p in text.split(" and ")]):
        m = re.match(r"move\s+(\w+)\s+([-+]?\d*\.?\d+)\s*(\w+)", part, flags=re.IGNORECASE)
        if not m:
            continue
        direction = m.group(1).lower()
        try:
            value = float(m.group(2))
        except Exception:
            continue
        unit = m.group(3).lower()
        # Normalize to cm
        if unit.startswith("mm"):
            value = value / 10.0
        elif unit == "m" or (unit.startswith("m") and not unit.startswith("mm")):
            value = value * 100.0
        totals[direction] = totals.get(direction, 0.0) + value
        any_valid = True
    if not any_valid:
        return None
    right = totals["right"] - totals["left"]
    forward = totals["forward"] - totals["backward"]
    down = totals["down"] - totals["up"]
    return np.array([right, forward, down], dtype=np.float32)


def _draw_text_block(img: np.ndarray, lines: list[str]) -> np.ndarray:
    """Draw a small semi-transparent box with text lines at the bottom-left of the image.

    Uses OpenCV if available; otherwise returns the original image.
    """
    try:
        import cv2
    except Exception:
        return img

    out = img.copy()
    h, w = out.shape[:2]
    pad = 8
    line_h = max(14, h // 36)
    box_h = pad * 2 + line_h * len(lines)
    y0 = h - box_h - 2
    x0 = 2
    x1 = min(w - 2, w - 2)
    y1 = min(h - 2, h - 2)
    overlay = out.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 0, 0), thickness=-1)
    out = cv2.addWeighted(overlay, 0.45, out, 0.55, 0)
    y = y0 + pad + line_h
    for line in lines:
        cv2.putText(out, line, (x0 + pad, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        y += line_h
    return out


def _to_local_array(x):
    """Return a numpy view of the process-local portion of a possibly-global jax.Array.

    Avoids jax.device_get on non-addressable arrays by using addressable_shards.
    Assumes leading axis is batch when concatenating shards.
    """
    if x is None:
        return None
    try:
        shards = getattr(x, "addressable_shards", None)
        if shards is not None and len(shards) > 0:
            # Scalar case: pick first shard
            arr0 = np.asarray(shards[0].data)
            if arr0.ndim == 0:
                return arr0
            return np.concatenate([np.asarray(s.data) for s in shards], axis=0)
    except Exception:
        pass
    try:
        return np.asarray(x)
    except Exception:
        return x


def _to_local_scalar(x) -> int:
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


def _invert_camera_axis_map(v_cm: np.ndarray) -> np.ndarray:
    """Invert AXIS_PERM mapping to camera-frame delta in metres.

    Mirrors scripts/visualization/train_vis_gripper.py logic.
    """
    AXIS_PERM = np.array([0, 2, 1], dtype=np.int32)
    AXIS_SIGN = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    t_cam = np.zeros(3, dtype=np.float32)
    t_cam[AXIS_PERM] = (v_cm / 100.0) / AXIS_SIGN
    return t_cam


def _project_point(
    base_xyz: np.ndarray, cam_T_base: np.ndarray, intr: np.ndarray, out_hw: tuple[int, int]
) -> tuple[int, int] | None:
    """Project base-frame 3D point to pixel coordinates respecting resize_with_pad letterboxing.

    intr: [fx, fy, cx, cy] measured at calibration resolution (Wc≈2*cx, Hc≈2*cy).
    """
    if base_xyz is None or intr is None or cam_T_base is None:
        return None
    if intr.shape[-1] != 4 or cam_T_base.shape[-2:] != (4, 4):
        return None
    fx, fy, cx, cy = intr.tolist()
    if fx == 0 or fy == 0:
        return None
    base_to_cam = np.linalg.inv(cam_T_base)
    p_base_h = np.array([base_xyz[0], base_xyz[1], base_xyz[2], 1.0], dtype=np.float32)
    p_cam = base_to_cam @ p_base_h
    z = float(p_cam[2])
    if z <= 1e-6:
        return None
    # Calibration pixel coordinates (before resize/pad)
    u = fx * (p_cam[0] / z) + cx
    v = fy * (p_cam[1] / z) + cy
    # Derive calibration resolution from principal point
    Ht, Wt = int(out_hw[0]), int(out_hw[1])
    Wc = max(1.0, 2.0 * cx)
    Hc = max(1.0, 2.0 * cy)
    # Compute resized (letterboxed) dimensions identical to resize_with_pad
    ratio = max(Wc / Wt, Hc / Ht)
    resized_w = int(Wc / ratio)
    resized_h = int(Hc / ratio)
    pad_w0 = (Wt - resized_w) // 2
    pad_h0 = (Ht - resized_h) // 2
    # Scale and offset
    x = int(np.round(u * (resized_w / Wc) + pad_w0))
    y = int(np.round(v * (resized_h / Hc) + pad_h0))
    x = int(np.clip(x, 0, Wt - 1))
    y = int(np.clip(y, 0, Ht - 1))
    return x, y


def _draw_dot(img_u8: np.ndarray, xy: tuple[int, int] | None, color: tuple[int, int, int]) -> np.ndarray:
    out = img_u8.copy()
    if xy is None:
        return out
    x, y = xy
    H, W = out.shape[:2]
    rr = 4
    y0, y1 = max(0, y - rr), min(H, y + rr + 1)
    x0, x1 = max(0, x - rr), min(W, x + rr + 1)
    for yy in range(y0, y1):
        for xx in range(x0, x1):
            if (yy - y) ** 2 + (xx - x) ** 2 <= rr * rr:
                out[yy, xx] = color
    return out


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
    log_mesh_and_sharding_header(mesh, title="Device mesh")
    logging.info("Data sharding spec: %s", _format_sharding(data_sharding))
    logging.info("Replicated sharding spec: %s", _format_sharding(replicated_sharding))

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
    logging.info("After getting batch")
    logging.info(f"Initialized data loader (shapes):\n{training_utils.array_tree_to_info(batch)}")
    # Sharding details for the first batch
    log_batch_sharding(batch)

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
        tok = None
        try:
            # RLDS path
            tok = val_loader._data_loader._dataset._transform.transforms[-1].tokenizer  # type: ignore[attr-defined]
        except Exception:
            try:
                # Torch path
                tok = data_loader._data_loader._dataset._transform.transforms[-1].tokenizer  # type: ignore[attr-defined]
            except Exception:
                tok = None
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
            in_shardings=(train_state_sharding, data_sharding),
            out_shardings=(data_sharding, replicated_sharding),
        )
        # Determine how many validation batches to evaluate each time.
        # If a fixed validation subset size is configured, compute batches from it;
        # otherwise fall back to a heuristic constant divided by global batch size.
        if getattr(config.data, "val_max_samples", None):
            # local batch size per host mirrors RLDS dataset batching
            process_count = getattr(jax, "process_count", lambda: 1)()
            local_bs = max(1, config.batch_size // process_count)
            num_val_batches = int(math.ceil(config.data.val_max_samples / local_bs))
        else:
            num_val_batches = int(60000 / config.batch_size)  # adjust if needed
    start_step = int(train_state.step)
    pbar = tqdm.tqdm(
        range(start_step, config.num_train_steps),
        initial=start_step,
        total=config.num_train_steps,
        dynamic_ncols=True,
        disable=(jax.process_index() != 0),
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
            info_str = ", ".join(f"{k}={v:.4f}" for k, v in reduced_info.items())
            pbar.write(f"Step {step}: {info_str}")
            if jax.process_index() == 0:
                wandb.log(reduced_info, step=step)
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
            with sharding.set_mesh(mesh):
                val_infos = []
                # Collect L2 distances (in cm) between parsed vectors from GT and predicted texts
                l2_cm_values: list[float] = []
                # Limit number of annotated images per validation run
                logged_images = 0
                max_images_to_log = 8
                # Recreate a fresh iterator to ensure the same fixed validation subset each time.
                val_iter = iter(val_loader)
                for _ in val_pbar:
                    val_batch = next(val_iter)
                    val_info = pval_step(train_rng, train_state, val_batch)
                    val_infos.append(val_info)
                    # Always run reasoning sampling across all processes; restrict decoding/logging to process 0.
                    obs = val_batch[0]
                    if tok is not None and logged_images < max_images_to_log:
                        id_buf, t_final = psample_reasoning(train_state, obs)
                    # Host-side visualization/metric (non-jitted)
                    if tok is not None and logged_images < max_images_to_log and jax.process_index() == 0:
                        # Decode ground-truth reasoning strings
                        gt_texts = _decode_reasoning_strings(obs, tok)
                        # Decode sampled reasoning tokens
                        ids = _to_local_array(id_buf)
                        # Be robust to bounds: clamp final index
                        t_host = int(np.clip(_to_local_scalar(t_final), 0, ids.shape[1] - 1))
                        B = ids.shape[0]
                        pred_texts: list[str] = []
                        for bi in range(B):
                            seq = ids[bi, : t_host + 1, 0].astype(np.int32)
                            try:
                                pred_texts.append(tok.decode(seq))
                            except Exception:
                                pred_texts.append("")

                        # Compute L2 metric over parsed movement vectors (in cm)
                        for bi in range(min(B, 64)):
                            gt_vec = _parse_language_delta_cm(gt_texts[bi] if bi < len(gt_texts) else "")
                            pred_vec = _parse_language_delta_cm(pred_texts[bi] if bi < len(pred_texts) else "")
                            if gt_vec is None or pred_vec is None:
                                continue
                            l2_cm = float(np.linalg.norm(gt_vec - pred_vec))
                            l2_cm_values.append(l2_cm)

                        # Prepare annotated images for a subset
                        # Choose a camera to display
                        first_cam_key = next(iter(obs.images))
                        imgs = _to_local_array(obs.images[first_cam_key])
                        # Convert from [-1,1] to uint8
                        imgs_u8 = ((np.asarray(imgs) + 1.0) * 0.5 * 255.0).clip(0, 255).astype(np.uint8)
                        # Optional 3D->2D projection inputs
                        cart = getattr(obs, "cartesian_position_window", None)
                        intr_all = getattr(obs, "camera_intrinsics", None)
                        extr_all = getattr(obs, "camera_extrinsics", None)
                        cart_np = _to_local_array(cart) if cart is not None else None
                        intr_np = _to_local_array(intr_all) if intr_all is not None else None
                        extr_np = _to_local_array(extr_all) if extr_all is not None else None
                        to_log = []
                        for bi in range(min(B, max_images_to_log - logged_images)):
                            vis = imgs_u8[bi]
                            H, W = vis.shape[:2]
                            start_xyz = end_xyz = None
                            intr = extr = None
                            if cart_np is not None and cart_np.shape[1] >= 1:
                                # [T,6]
                                seq = np.asarray(cart_np[bi])
                                if seq.ndim == 2 and seq.shape[-1] >= 3:
                                    start_xyz = seq[0, :3]
                                    end_xyz = seq[-1, :3]
                            if intr_np is not None:
                                ci = np.asarray(intr_np[bi])
                                intr = ci[0] if ci.ndim == 2 else ci
                            if extr_np is not None:
                                ce = np.asarray(extr_np[bi])
                                extr = ce[0] if ce.ndim == 3 else ce
                            # Project GT start/end if available
                            start_xy = _project_point(start_xyz, extr, intr, (H, W)) if start_xyz is not None else None
                            end_true_xy = _project_point(end_xyz, extr, intr, (H, W)) if end_xyz is not None else None
                            # Predicted end via language delta
                            pred_end_xy = None
                            v_cm = _parse_language_delta_cm(pred_texts[bi] if bi < len(pred_texts) else "")
                            if v_cm is not None and extr is not None and start_xyz is not None:
                                t_cam = _invert_camera_axis_map(v_cm)
                                R_cb = extr[:3, :3]
                                t_base = R_cb @ t_cam
                                pred_xyz = start_xyz + t_base
                                pred_end_xy = _project_point(pred_xyz, extr, intr, (H, W))
                            # Draw dots
                            vis2 = vis
                            if start_xy is not None:
                                vis2 = _draw_dot(vis2, start_xy, (0, 255, 255))  # GT start (yellow)
                            if pred_end_xy is not None:
                                vis2 = _draw_dot(vis2, pred_end_xy, (0, 0, 255))  # Pred end (red)
                            if end_true_xy is not None:
                                vis2 = _draw_dot(vis2, end_true_xy, (0, 255, 0))  # GT end (green)
                            lines = [
                                f"GT: {gt_texts[bi] if bi < len(gt_texts) else ''}",
                                f"Pred: {pred_texts[bi] if bi < len(pred_texts) else ''}",
                            ]
                            vis2 = _draw_text_block(vis2, lines)
                            to_log.append(wandb.Image(vis2))
                        if to_log and jax.process_index() == 0:
                            wandb.log({"val/annotated": to_log}, step=step)
                            logged_images += len(to_log)
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
