import dataclasses
import logging

from openpi_client import image_tools
import tyro

image_tools
from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config
from openpi.training.droid_rlds_dataset import DroidCoTRldsDataset


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
    match args.policy:
        case Checkpoint():
            return _policy_config.create_trained_policy(
                policy_config,
                _config.get_config(args.policy.config),
                args.policy.dir,
                default_prompt=args.default_prompt,
            )
        case _:
            raise ValueError(f"Invalid policy type. Expected Checkpoint, got: {type(args.policy)}")


def main(args: Args) -> None:
    policy_config = _policy_config.PolicyConfig(
        policy_type=_policy.PolicyType.CoTPolicy,
        use_norm_stats=False,  # We don't use norm stats in this script.
    )
    policy = create_policy(args, policy_config=policy_config)

    # Record the policy's behavior.
    if args.record:
        policy = _policy.PolicyRecorder(policy, "policy_records")

    ds = DroidCoTRldsDataset(
        data_dir="/n/fs/vla-mi/datasets/OXE/",
        language_action_dir="/n/fs/robot-data/vlm-syn/posed_droid",
        batch_size=1,
        shuffle_buffer_size=200,
    )
    ds = iter(ds)
    for _, batch in enumerate(ds):
        curr_obs = batch["observation"]
        data = {
            "observation/exterior_image_1_left": image_tools.resize_with_pad(curr_obs["image"].squeeze(0), 224, 224),
            # "observation/wrist_image_left": image_tools.resize_with_pad(curr_obs["wrist_image"], 224, 224),
            "observation/cartesian_position": curr_obs["cartesian_position"].squeeze(0),
            "observation/gripper_position": curr_obs["gripper_position"].squeeze(0),
            "prompt": batch["prompt"].squeeze(0).item().decode(),
        }
        outputs = policy.infer_reasoning(data)
        breakpoint()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
