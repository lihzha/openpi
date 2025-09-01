import dataclasses
import logging
from math import acos
from math import pi
import re
import warnings

import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import image_tools
import tyro
import os

import etils.epath as epath
from rail_tpu_utils import prevent_cross_region

import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.sharding as sharding

from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config

warnings.filterwarnings("ignore", category=FutureWarning)


# ───────────────────────── helpers ─────────────────────────

MOVE_RE = re.compile(r"(left|right|forward|backward|up|down)\s+([-+]?\d+(?:\.\d+)?)\s*cm", re.IGNORECASE)
GRIPPER_RE = re.compile(r"set\s+gripper\s+to\s+([-+]?\d+(?:\.\d+)?)", re.IGNORECASE)


def parse_vec_and_gripper(s: str):
    """
    Parse a movement string into (vec, gripper).
    Convention:
      x: left (+), right (-)
      y: backward (+), forward (-)
      z: down (+), up (-)
    Returns:
      vec: np.ndarray shape (3,)
      gripper: float (None if not found -> 0.0 by default)
    """
    x = y = z = 0.0
    for direction, val in MOVE_RE.findall(s or ""):
        v = float(val)
        d = direction.lower()
        if d == "left":
            x += v
        elif d == "right":
            x -= v
        elif d == "backward":
            y += v
        elif d == "forward":
            y -= v
        elif d == "down":
            z += v
        elif d == "up":
            z -= v

    g = 0.0
    m = GRIPPER_RE.search(s or "")
    if m:
        g = float(m.group(1))
    return np.array([x, y, z], dtype=float), g


def safe_cosine_and_angle(a: np.ndarray, b: np.ndarray):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0.0 and nb == 0.0:
        return 1.0, 0.0  # identical zero vectors
    if na == 0.0 or nb == 0.0:
        return 0.0, pi / 2  # orthogonal-ish when one is zero
    cos = float(np.dot(a, b) / (na * nb))
    cos = max(-1.0, min(1.0, cos))  # numerical safety
    return cos, float(acos(cos))


def losses_from_strings(pred_str: str, gt_str: str):
    """
    Compute a suite of losses between predicted and ground-truth movement strings.
    a = ground-truth vector, b = predicted vector
    Returns a dict of components + a 'total_loss' you can tweak via weights.
    """
    a, g_gt = parse_vec_and_gripper(gt_str)
    b, g_pr = parse_vec_and_gripper(pred_str)

    # vector metrics
    l2 = float(np.linalg.norm(a - b))
    cos, angle = safe_cosine_and_angle(a, b)  # angle in radians
    len_a, len_b = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    length_abs_diff = abs(len_a - len_b)

    # scale-invariant residual: best s*b to match a
    if (b @ b) == 0.0:
        s = 0.0
        residual = float(np.linalg.norm(a))
    else:
        s = float((a @ b) / (b @ b))
        residual = float(np.linalg.norm(a - s * b))

    # gripper loss (L1 by default; switch to L2 if you prefer)
    gripper_l1 = abs(g_pr - g_gt)

    # You can tune these weights as you like
    w_l2 = 1.0
    w_angle = 1.0  # radians; consider scaling (e.g., * len_a) if you want distance-like units
    w_len = 1.0
    w_resid = 1.0
    w_grip = 1.0

    total = w_l2 * l2 + w_angle * angle + w_len * length_abs_diff + w_resid * residual + w_grip * gripper_l1

    return {
        "vec_pred": b,
        "vec_gt": a,
        "grip_pred": g_pr,
        "grip_gt": g_gt,
        "l2": l2,
        "angle_rad": angle,
        "length_abs_diff": length_abs_diff,
        "scale_factor": s,
        "scale_inv_residual": residual,
        "gripper_l1": gripper_l1,
        "total_loss": total,
    }


@dataclasses.dataclass
class Checkpoint:
    """Load a policy from a trained checkpoint."""

    # Training config name (e.g., "pi0_aloha_sim").
    config: str
    # Checkpoint directory (e.g., "checkpoints/pi0_aloha_sim/exp/10000").
    dir: str


@dataclasses.dataclass
class Args:
    """Arguments for the serve_policy script."""

    policy: Checkpoint = dataclasses.field()
    # If provided, will be used in case the "prompt" key is not present in the data, or if the model doesn't have a default
    # prompt.
    default_prompt: str | None = None

    # Port to serve the policy on.
    port: int = 8000
    # Record the policy's behavior for debugging.
    record: bool = False

    # Specifies how to load the policy. If not provided, the default policy for the environment will be used.

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

def create_policy(args: Args, policy_config: _policy_config.PolicyConfig) -> _policy.Policy:
    """Create a policy from the given arguments."""
    if isinstance(args.policy, Checkpoint):
        return _policy_config.create_trained_policy(
            policy_config,
            _config.get_config(args.policy.config),
            args.policy.dir,
            default_prompt=args.default_prompt,
        )
    raise ValueError(f"Invalid policy type. Expected Checkpoint, got: {type(args.policy)}")

def _is_tpu_runtime() -> bool:
    try:
        return any(d.platform == "tpu" for d in jax.devices())
    except Exception:
        return False

def main(args: Args):
    policy_config = _policy_config.PolicyConfig(
        policy_type=_policy.PolicyType.CoTPolicy,
        use_norm_stats=False,  # We don't use norm stats in this script.
    )
    policy = create_policy(args, policy_config=policy_config)

    config = _config.get_config(args.policy.config)
    # config = dc.replace(config, data=dc.replace(config.data, max_samples=150, left_pad=True), batch_size=8)
    
    if ("v6" in config.name and config.fsdp_devices > 8) or ("v4" in config.name and config.fsdp_devices > 4):
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

    logging.info(
        f"Summation steps: {config.data.summation_steps}, left_pad: {config.data.left_pad}, sum_decimal: {config.data.sum_decimal}, ema_decay: {config.ema_decay}"
    )

    jax.config.update("jax_compilation_cache_dir", str(epath.Path("~/.cache/jax").expanduser()))


    mesh = sharding.make_mesh(effective_fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))

    data_loader = _data_loader.create_data_loader(
        config,
        sharding=data_sharding,
        shuffle=True,
        seed=config.seed,
        split="val",
    )
    
    tok = data_loader._data_loader._dataset._transform.transforms[-1].tokenizer
    ds = iter(data_loader)

    totals = {
        "num_batches": 0,
        "l2": 0.0,
        "angle_rad": 0.0,
        "length_abs_diff": 0.0,
        "scale_inv_residual": 0.0,
        "gripper_l1": 0.0,
        "total_loss": 0.0,
    }
    for idx, batch in enumerate(ds):
        curr_obs = batch[0]
        arr = curr_obs.tokenized_prompt[0]
        pos_2 = jnp.where(arr == 2, size=1, fill_value=-1)[0]
        pos_108 = jnp.where(arr == 108, size=1, fill_value=-1)[0]
        pos_1 = jnp.where(arr == 1, size=1, fill_value=-1)[0]
        prompt = tok.decode(arr[pos_2.item() + 1 : pos_108.item()])
        gt_lang_action = tok.decode(arr[pos_108.item() + 1 : pos_1.item()])
        data = {
            "observation/exterior_image_1_left": image_tools.resize_with_pad(
                curr_obs.images["base_0_rgb"][0], 224, 224
            ),
            "observation/cartesian_position": curr_obs.state[0, :6],
            "observation/gripper_position": curr_obs.state[0, 6:7],
            "prompt": prompt,
        }
        outputs = policy.infer_reasoning(data)["reasoning"]  # predicted string
        assert outputs is not None

        totals["num_batches"] += 1

        print(f"Batch {idx}")
        print("  Pred:", outputs)
        print("  GT:  ", gt_lang_action)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
