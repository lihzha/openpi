import dataclasses
import dataclasses as dc
import hashlib
import logging
from math import acos
from math import pi
import re
import warnings

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from openpi_client import image_tools
import tyro

from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader

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


def main(args: Args):
    policy_config = _policy_config.PolicyConfig(
        policy_type=_policy.PolicyType.CoTPolicy,
        use_norm_stats=False,  # We don't use norm stats in this script.
    )
    policy = create_policy(args, policy_config=policy_config)

    config = _config.get_config(args.policy.config)
    config = dc.replace(config, data=dc.replace(config.data, max_samples=150, left_pad=True), batch_size=8)

    logging.info(
        f"Summation steps: {config.data.summation_steps}, left_pad: {config.data.left_pad}, sum_decimal: {config.data.sum_decimal}, ema_decay: {config.ema_decay}"
    )

    data_loader = _data_loader.create_data_loader(
        config,
        sharding=None,
        shuffle=True,
        seed=config.seed,
    )
    ds = iter(data_loader)

    tok = data_loader._data_loader._dataset._transform.transforms[-1].tokenizer
    max_samples_cfg = getattr(config.data, "max_samples", None)
    logging.info("Running capped-samples sanity check (expect repeat after ~%s samples)", max_samples_cfg)
    seen = set()
    total = 0
    repeated = False
    test_iter = iter(data_loader)
    while not repeated:
        test_batch = next(test_iter)
        # test_batch is (Observation, Actions)
        obs = test_batch[0]
        lang_actions_encoded = obs.tokenized_prompt
        # Use the first available camera stream as a stable fingerprint basis
        first_cam = next(iter(obs.images.values()))
        B = first_cam.shape[0]
        for i in range(B):
            img_bytes = bytes(memoryview(jax.device_get(first_cam[i]).astype("uint8").tobytes()))
            h = hashlib.sha1(img_bytes).hexdigest()
            if h in seen:
                repeated = True
                break
            lang_action = jax.device_get(tok.decode(lang_actions_encoded[i]))
            images = jax.device_get(first_cam[i])
            images = (images + 1) / 2
            # img0 = images[0]
            # img1 = images[-1]
            img0 = images
            img1 = images

            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            axes[0].imshow(img0)
            axes[0].axis("off")
            axes[0].set_title("t=0s")
            # Write language action on t=0s image
            _action_text = str(lang_action)
            axes[0].text(
                0.01,
                0.99,
                _action_text,
                transform=axes[0].transAxes,
                va="top",
                ha="left",
                fontsize=10,
                color="white",
                bbox=dict(facecolor="black", alpha=0.5, pad=3),
            )
            axes[1].imshow(img1)
            axes[1].axis("off")
            axes[1].set_title("t≈+1s")
            plt.suptitle("Initial vs +1s")
            plt.savefig(f"initial_vs_1s_{total}.png")
            seen.add(h)
            total += 1
    logging.info("Capped-samples sanity: unique before repeat=%d (configured max_samples=%s)", total, max_samples_cfg)

    # save the hash list to a file, which is easliy loaded in the future
    with open("hash0.txt", "w") as f:
        for h in seen:
            f.write(h + "\n")

    # max_samples_cfg = getattr(config.data, "max_samples", None)
    tok = data_loader._data_loader._dataset._transform.transforms[-1].tokenizer
    # logging.info("Running capped-samples sanity check (expect repeat after ~%s samples)", max_samples_cfg)
    # seen = set()
    # total = 0
    # repeated = False
    # test_iter = iter(data_loader)
    # while not repeated:
    #     test_batch = next(test_iter)
    #     # test_batch is (Observation, Actions)
    #     obs = test_batch[0]
    #     # Use the first available camera stream as a stable fingerprint basis
    #     first_cam = next(iter(obs.images.values()))
    #     for i in range(first_cam.shape[0]):
    #         img_bytes = bytes(memoryview(jax.device_get(first_cam[i]).astype("uint8").tobytes()))
    #         h = hashlib.sha1(img_bytes).hexdigest()
    #         if h in seen:
    #             repeated = True
    #             break
    #         seen.add(h)
    #         total += 1
    # logging.info("Capped-samples sanity: unique before repeat=%d (configured max_samples=%s)", total, max_samples_cfg)

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
        # summed = _sum_language_actions(
        #     seq,
        #     sum_decimal="2f",
        # )  # ground-truth string

        # compute losses
        # comp = losses_from_strings(outputs, summed)

        # accumulate
        totals["num_batches"] += 1
        # for k in ["l2", "angle_rad", "length_abs_diff", "scale_inv_residual", "gripper_l1", "total_loss"]:
        #     totals[k] += comp[k]

        # (optional) per-batch logging
        print(f"Batch {idx}")
        print("  Pred:", outputs)
        print("  GT:  ", gt_lang_action)
        # print("  vec_pred:", comp["vec_pred"], "vec_gt:", comp["vec_gt"])
        # print(
        #     "  l2:",
        #     comp["l2"],
        #     "angle(rad):",
        #     comp["angle_rad"],
        #     "len|Δ|:",
        #     comp["length_abs_diff"],
        #     "residual:",
        #     comp["scale_inv_residual"],
        #     "grip|Δ|:",
        #     comp["gripper_l1"],
        #     "total:",
        #     comp["total_loss"],
        # )

        # if idx % 10 == 0:
        #     # final totals and (optional) averages
        #     N = max(1, totals["num_batches"])
        #     print("\n=== Totals ===")
        #     print(f"batches: {totals['num_batches']}")
        #     print(f"L2 sum: {totals['l2']:.6f}")
        #     print(f"Angle(rad) sum: {totals['angle_rad']:.6f}")
        #     print(f"Length |Δ| sum: {totals['length_abs_diff']:.6f}")
        #     print(f"Scale-invariant residual sum: {totals['scale_inv_residual']:.6f}")
        #     print(f"Gripper L1 sum: {totals['gripper_l1']:.6f}")
        #     print(f"TOTAL loss sum: {totals['total_loss']:.6f}")

        #     print("\n=== Averages per batch ===")
        #     print(f"L2 avg: {totals['l2'] / N:.6f}")
        #     print(f"Angle(rad) avg: {totals['angle_rad'] / N:.6f}")
        #     print(f"Length |Δ| avg: {totals['length_abs_diff'] / N:.6f}")
        #     print(f"Scale-invariant residual avg: {totals['scale_inv_residual'] / N:.6f}")
        #     print(f"Gripper L1 avg: {totals['gripper_l1'] / N:.6f}")
        #     print(f"TOTAL loss avg: {totals['total_loss'] / N:.6f}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
