# ruff: noqa

import contextlib
import dataclasses
import faulthandler
import signal
import time
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy
from droid.robot_env import RobotEnv
import tqdm
import tyro
import re
from scipy.spatial.transform import Rotation as R

AXIS_PERM = np.array([0, 2, 1])
AXIS_SIGN = np.array([1, 1, 1])

faulthandler.enable()

# DROID data collection frequency -- we slow down execution to match this frequency
DROID_CONTROL_FREQUENCY = 15


@dataclasses.dataclass
class Args:
    # Rollout parameters
    max_timesteps: int = 600
    # How many actions to execute from a predicted action chunk before querying policy server again
    # 8 is usually a good default (equals 0.5 seconds of action execution).
    open_loop_horizon: int = 8

    # Remote server parameters
    remote_host: str = "10.249.9.33"  # point this to the IP address of the policy server, e.g., "192.168.1.100"
    remote_port: int = (
        8000  # point this to the port of the policy server, default server port for openpi servers is 8000
    )


# We are using Ctrl+C to optionally terminate rollouts early -- however, if we press Ctrl+C while the policy server is
# waiting for a new action chunk, it will raise an exception and the server connection dies.
# This context manager temporarily prevents Ctrl+C and delays it after the server call is complete.
@contextlib.contextmanager
def prevent_keyboard_interrupt():
    """Temporarily prevent keyboard interrupts by delaying them until after the protected code."""
    interrupted = False
    original_handler = signal.getsignal(signal.SIGINT)

    def handler(signum, frame):
        nonlocal interrupted
        interrupted = True

    signal.signal(signal.SIGINT, handler)
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, original_handler)
        if interrupted:
            raise KeyboardInterrupt


def main(args: Args):
    # Make sure external camera is specified by user -- we only use one external camera for the policy

    # Initialize the Panda environment. Using joint velocity action space and gripper position action space is very important.
    # env = RobotEnv(action_space="joint_velocity", gripper_action_space="position")
    env = RobotEnv(
        robot_type="panda",
        action_space="cartesian_position",
        gripper_action_space="position",
    )
    extrinsics = [0.7214792, 0.73091813, 0.723, -0.05750051, 0.19751727, 0.90085098, -0.38229326]  # (x,y,z),(w,x,y,z)
    # Turn (pos, quat_wxyz) into 4x4 cam->base extrinsics matrix
    pos = np.array(extrinsics[:3], dtype=float)
    w, x, y, z = extrinsics[3:7]
    quat_xyzw = np.array([x, y, z, w], dtype=float)  # convert (w,x,y,z) -> (x,y,z,w)
    rot_mat = R.from_quat(quat_xyzw).as_matrix()
    cam_to_base_extrinsics_matrix = np.eye(4, dtype=float)
    cam_to_base_extrinsics_matrix[:3, :3] = rot_mat
    cam_to_base_extrinsics_matrix[:3, 3] = pos
    print("Created the droid env!")

    # Connect to the policy server
    policy_client = websocket_client_policy.WebsocketClientPolicy(args.remote_host, args.remote_port)

    while True:
        # instruction = input("Enter instruction: ")
        instruction = "pick up the tomato and put it into the metal plate"

        # Prepare to save video of rollout
        bar = tqdm.tqdm(range(args.max_timesteps))
        print("Running rollout... press Ctrl+C to stop early.")
        for t_step in bar:
            # start_time = time.time()
            try:
                # Get the current observation
                curr_obs = _extract_observation(
                    args,
                    env.get_observation(),
                    # Save the first observation to disk
                    save_to_disk=t_step == 0,
                )

                request_data = {
                    "observation/image": image_tools.resize_with_pad(curr_obs["observation/image"], 224, 224),
                    "observation/wrist_image": image_tools.resize_with_pad(
                        curr_obs["observation/wrist_image"], 224, 224
                    ),
                    "observation/state": curr_obs["observation/state"],
                    "prompt": instruction,
                    "batch_size": None,
                }

                # Wrap the server call in a context manager to prevent Ctrl+C from interrupting it
                # Ctrl+C will be handled after the server call is complete
                with prevent_keyboard_interrupt():
                    # this returns natural language reasoning steps; convert to deltas then to absolute action
                    st = time.time()
                    pred = policy_client.infer_reasoning(request_data)["reasoning"]
                    poses_cam, grip_actions = _reasoning_to_action(pred)

                    # Take the first step translation in camera frame (meters)
                    t_cam = poses_cam[1][:3, 3] - poses_cam[0][:3, 3]
                    # Map translation delta to robot/base frame using rotation only
                    R_cb = cam_to_base_extrinsics_matrix[:3, :3]
                    delta_base = R_cb @ t_cam

                    # Turn delta into absolute cartesian action using current state (position + euler angles)
                    curr_state = np.asarray(curr_obs["observation/state"], dtype=float)
                    curr_pos = curr_state[:3]
                    curr_rpy = curr_state[3:6] if curr_state.shape[0] >= 6 else np.zeros(3, dtype=float)
                    next_pos = curr_pos + delta_base
                    next_grip = float(grip_actions[0]) if grip_actions.size > 0 else float(curr_state[-1])
                    action = np.concatenate([next_pos, curr_rpy, np.array([next_grip], dtype=float)])

                    et = time.time()
                    print(f"Time taken for inference: {et - st}")
                # Binarize gripper action
                if action[-1].item() > 0.5:
                    action = np.concatenate([action[:-1], np.ones((1,))])
                else:
                    action = np.concatenate([action[:-1], np.zeros((1,))])

                # TODO: from delta action to absolute action

                env.step(action)

                # Sleep to match DROID data collection frequency
                # elapsed_time = time.time() - start_time
                # if elapsed_time < 1 / DROID_CONTROL_FREQUENCY:
                #     time.sleep(1 / DROID_CONTROL_FREQUENCY - elapsed_time)
            except KeyboardInterrupt:
                break

        answer = input("Do one more eval? (enter y or n) ")
        if "n" in answer.lower():
            break
        while True:
            env.reset()
            answer = input("Correctly reset (enter y or n)? ")
            if "n" in answer.lower():
                continue
            break


def _extract_observation(args: Args, obs_dict, *, save_to_disk=False):
    image_observations = obs_dict["image"]
    # In addition to image observations, also capture the proprioceptive state
    robot_state = obs_dict["robot_state"]
    cartesian_position = np.array(robot_state["cartesian_position"])
    gripper_position = np.array([robot_state["gripper_position"]])

    if gripper_position > 0.2:
        gripper_position = 1.0
    else:
        gripper_position = 0.0

    return {
        "observation/image": image_observations["0"],
        "observation/wrist_image": image_observations["1"],
        "observation/state": np.concatenate([cartesian_position, [gripper_position]]),
    }


def _reasoning_to_action(
    sentences: list[str],
    start_pose: np.ndarray = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Invert `describe_movement`.

    Given a list of reasoning sentences (one per transition), reconstruct the
    sequence of gripper poses (4x4) and the per-step gripper actions.

    - Each sentence is expected to contain phrases like
      "move right 1.23 cm and move forward 4.56 cm and move down 7.89 cm and set gripper to 0.50".
    - Rotation is assumed constant (no rotation changes), matching describe_movement.
    - The first pose is `start_pose` (defaults to identity), then each sentence
      adds a translation delta to produce the next pose.

    Returns:
        (gripper_poses, gripper_actions)
        - gripper_poses: array of shape (len(sentences)+1, 4, 4)
        - gripper_actions: array of shape (len(sentences),)
    """
    if start_pose is None:
        start_pose = np.eye(4, dtype=float)
    else:
        start_pose = np.array(start_pose, dtype=float)

    num_steps = len(sentences)
    poses = np.zeros((num_steps + 1, 4, 4), dtype=float)
    poses[0] = start_pose
    actions = np.zeros((num_steps,), dtype=float)

    # Regex patterns
    move_pat = re.compile(r"move\s+(right|left|forward|backward|up|down)\s+([\-\d\.]+)\s*cm", re.IGNORECASE)
    grip_pat = re.compile(r"set\s+gripper\s+to\s+([\-\d\.]+)", re.IGNORECASE)

    for i, sentence in enumerate(sentences):
        # Parse movements; accumulate in centimeters along (dx, dy, dz)
        dx_cm = dy_cm = dz_cm = 0.0
        for m in move_pat.finditer(sentence):
            direction = m.group(1).lower()
            value_cm = float(m.group(2))
            if direction == "right":
                dx_cm += value_cm
            elif direction == "left":
                dx_cm -= value_cm
            elif direction == "forward":
                dy_cm += value_cm
            elif direction == "backward":
                dy_cm -= value_cm
            elif direction == "down":
                dz_cm += value_cm
            elif direction == "up":
                dz_cm -= value_cm

        # Parse gripper action (defaults to previous if missing, else 0.0 for first)
        grip_match = grip_pat.search(sentence)
        if grip_match:
            actions[i] = float(grip_match.group(1))
        else:
            actions[i] = actions[i - 1] if i > 0 else 0.0

        # Convert from language (dx,dy,dz) in cm back to camera-frame translation (meters)
        v_cm = np.array([dx_cm, dy_cm, dz_cm], dtype=float)
        v_m = v_cm / 100.0

        # Recall: v = (AXIS_SIGN * t_cam[AXIS_PERM]) * 100
        # Therefore: t_cam[AXIS_PERM] = v_m / AXIS_SIGN
        t_cam = np.zeros(3, dtype=float)
        # Avoid division-by-zero if AXIS_SIGN contains zeros (shouldn't, but safe)
        sign_safe = np.where(AXIS_SIGN == 0, 1.0, AXIS_SIGN.astype(float))
        t_mapped = v_m / sign_safe
        t_cam[AXIS_PERM] = t_mapped

        # Integrate translation to form next pose; keep rotation unchanged
        next_pose = poses[i].copy()
        next_pose[:3, 3] = poses[i][:3, 3] + t_cam
        poses[i + 1] = next_pose

    return poses, actions


def cam_to_robot(pose_cam: np.ndarray, cam_to_base_extrinsics_matrix: np.ndarray) -> np.ndarray:
    """
    Convert pose(s) from camera frame to robot/base frame.

    This is the inverse of the transform used below when computing camera-frame
    poses from base-frame poses:
        gripper_pose_in_camera_frame = (np.linalg.inv(cam_to_base_extrinsics_matrix) @ gripper_pose_matrices.T).T

    Args:
        pose_cam: A single 4x4 homogeneous pose matrix in camera frame, or an array of shape (N, 4, 4).
        cam_to_base_extrinsics_matrix: The 4x4 homogeneous transform from camera to base frame.

    Returns:
        Pose(s) in the robot/base frame with the same shape as input (4x4 or (N, 4, 4)).
    """
    T = np.array(cam_to_base_extrinsics_matrix, dtype=float)
    if T.shape != (4, 4):
        raise ValueError(f"cam_to_base_extrinsics_matrix must be 4x4, got {T.shape}")

    pose_cam = np.array(pose_cam, dtype=float)
    if pose_cam.ndim == 2:
        if pose_cam.shape != (4, 4):
            raise ValueError(f"pose_cam must be 4x4 when 2D, got {pose_cam.shape}")
        return T @ pose_cam
    elif pose_cam.ndim == 3:
        if pose_cam.shape[1:] != (4, 4):
            raise ValueError(f"pose_cam must be (N,4,4) when 3D, got {pose_cam.shape}")
        # Batch multiply: for each i, base_pose[i] = T @ pose_cam[i]
        return np.einsum("ab,nbc->nac", T, pose_cam)
    else:
        raise ValueError(f"pose_cam must be 2D (4x4) or 3D (N,4,4), got ndim={pose_cam.ndim}")


if __name__ == "__main__":
    args: Args = tyro.cli(Args)
    print(args)
    main(args)
