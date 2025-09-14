# ruff: noqa

import dataclasses
import faulthandler
import numpy as np
from openpi_client import image_tools
from droid.robot_env import RobotEnv
import tyro
from scipy.spatial.transform import Rotation as R
import sys
sys.path.append(".")
from shared import BaseEvalRunner

AXIS_PERM = np.array([0, 2, 1])
AXIS_SIGN = np.array([1, 1, 1])

faulthandler.enable()

# DROID data collection frequency -- we slow down execution to match this frequency
DROID_CONTROL_FREQUENCY = 15


@dataclasses.dataclass
class Args:
    # Hardware parameters
    left_camera_id: str = "31177322"  # e.g., "24259877"
    right_camera_id: str = "38872458"  # e.g., "24514023"
    wrist_camera_id: str = "10501775"  # e.g., "13062452"

    # Policy parameters
    external_camera: str = (
        None  # which external camera should be fed to the policy, choose from ["left", "right"]
    )

    # Rollout parameters
    max_timesteps: int = 600
    # How many actions to execute from a predicted action chunk before querying policy server again
    # 8 is usually a good default (equals 0.5 seconds of action execution).
    open_loop_horizon: int = 8

    # Remote server parameters
    remote_host: str = "0.0.0.0"  # point this to the IP address of the policy server, e.g., "192.168.1.100"
    remote_port: int = (
        8000  # point this to the port of the policy server, default server port for openpi servers is 8000
    )

    in_camera_frame: bool = False  # whether the predicted actions are in camera frame (True) or robot/base frame (False)


class DroidEvalRunner(BaseEvalRunner):

    def __init__(self, args):
        super().__init__(args)
    
    def init_env(self):
        return RobotEnv(
            action_space="cartesian_position",
            gripper_action_space="position",
        )


    def obs_to_request(self, curr_obs, instruction):
        return {
                "observation/exterior_image_1_left": image_tools.resize_with_pad(
                curr_obs[f"{self.args.external_camera}_image"], 224, 224
            ),
            "observation/wrist_image_left": image_tools.resize_with_pad(curr_obs["wrist_image"], 224, 224),
            "observation/cartesian_position": curr_obs["cartesian_position"],
            "observation/gripper_position": curr_obs["gripper_position"],
            "prompt": instruction,
            "batch_size": None,
        }

    def binarize_gripper(self, action):
        return action
    
    def _extract_observation(self, obs_dict, save_to_disk=False):
        image_observations = obs_dict["image"]
        left_image, right_image, wrist_image = None, None, None
        for key in image_observations:
            # Note the "left" below refers to the left camera in the stereo pair.
            # The model is only trained on left stereo cams, so we only feed those.
            if self.args.left_camera_id in key and "left" in key:
                left_image = image_observations[key]
            elif self.args.right_camera_id in key and "left" in key:
                right_image = image_observations[key]
            elif self.args.wrist_camera_id in key and "left" in key:
                wrist_image = image_observations[key]

        # Drop the alpha dimension
        # left_image = left_image[..., :3]
        right_image = right_image[..., :3]
        wrist_image = wrist_image[..., :3]

        # Convert to RGB
        # left_image = left_image[..., ::-1]
        right_image = right_image[..., ::-1]
        wrist_image = wrist_image[..., ::-1]

        # In addition to image observations, also capture the proprioceptive state
        robot_state = obs_dict["robot_state"]
        cartesian_position = np.array(robot_state["cartesian_position"])
        joint_position = np.array(robot_state["joint_positions"])
        gripper_position = np.array([robot_state["gripper_position"]])

        # Save the images to disk so that they can be viewed live while the robot is running
        # Create one combined image to make live viewing easy


        return {
            # "left_image": left_image,
            "right_image": right_image,
            "wrist_image": wrist_image,
            "cartesian_position": cartesian_position,
            "joint_position": joint_position,
            "gripper_position": gripper_position,
        }

class DroidExtrEvalRunner(DroidEvalRunner):

    def __init__(self, args):
        super().__init__(args)
    
    
    def set_extrinsics(self):
        extrinsics = [0.15297898357307485, -0.46533509871932016, 0.488261593272068, -2.1393635673966935, -0.01094009086539871, -0.7221553756328747]
        if len(extrinsics) == 6:
            extrinsics = extrinsics[:3] + R.from_euler("xyz", extrinsics[3:6], degrees=False).as_quat().tolist()
        # Turn (pos, quat_wxyz) into 4x4 cam->base extrinsics matrix
        pos = np.array(extrinsics[:3], dtype=float)
        w, x, y, z = extrinsics[3:7]
        quat_xyzw = np.array([x, y, z, w], dtype=float)  # convert (w,x,y,z) -> (x,y,z,w)
        rot_mat = R.from_quat(quat_xyzw).as_matrix()
        cam_to_base_extrinsics_matrix = np.eye(4, dtype=float)
        cam_to_base_extrinsics_matrix[:3, :3] = rot_mat
        cam_to_base_extrinsics_matrix[:3, 3] = pos
        return cam_to_base_extrinsics_matrix
    

if __name__ == "__main__":
    args: Args = tyro.cli(Args)
    print(args)
    if args.in_camera_frame:
        eval_runner = DroidExtrEvalRunner(args)
    else:
        eval_runner = DroidEvalRunner(args)
    eval_runner.run()
