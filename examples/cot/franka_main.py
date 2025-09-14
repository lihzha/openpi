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


faulthandler.enable()



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



class FrankaEvalRunner(BaseEvalRunner):

    def __init__(self, args):
        super().__init__(args)

    def init_env(self):
        return RobotEnv(
            robot_type="panda",
            action_space="cartesian_position",
            gripper_action_space="position",
        )
    
    def set_extrinsics(self):
        extrinsics = [0.7214792, 0.73091813, 0.723, -0.05750051, 0.19751727, 0.90085098, -0.38229326]  # (x,y,z),(w,x,y,z)
        # Turn (pos, quat_wxyz) into 4x4 cam->base extrinsics matrix
        pos = np.array(extrinsics[:3], dtype=float)
        w, x, y, z = extrinsics[3:7]
        quat_xyzw = np.array([x, y, z, w], dtype=float)  # convert (w,x,y,z) -> (x,y,z,w)
        rot_mat = R.from_quat(quat_xyzw).as_matrix()
        cam_to_base_extrinsics_matrix = np.eye(4, dtype=float)
        cam_to_base_extrinsics_matrix[:3, :3] = rot_mat
        cam_to_base_extrinsics_matrix[:3, 3] = pos

        return cam_to_base_extrinsics_matrix

    def _extract_observation(self, obs_dict, *, save_to_disk=False):
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
            "image": image_observations["0"],
            "wrist_image": image_observations["1"],
            "cartesian_position": cartesian_position,
            "gripper_position": np.array([gripper_position]),
        }

    def obs_to_request(self, curr_obs, instruction):
        return {
                "observation/exterior_image_1_left": image_tools.resize_with_pad(
                    curr_obs["image"], 224, 224
                ),
                # "observation/wrist_image_left": image_tools.resize_with_pad(
                #     curr_obs["wrist_image"], 224, 224
                # ),
                "observation/cartesian_position": curr_obs["cartesian_position"],
                "observation/gripper_position": curr_obs["gripper_position"],
                "prompt": instruction,
                "batch_size": None,
            }

    def binarize_gripper(self, action):
        # Binarize gripper action
        if action[-1].item() > 0.5:
            action = np.concatenate([action[:-1], np.ones((1,))])
        else:
            action = np.concatenate([action[:-1], np.zeros((1,))])
        return action

if __name__ == "__main__":
    args: Args = tyro.cli(Args)
    print(args)
    eval_runner = FrankaEvalRunner(args)
    eval_runner.run()