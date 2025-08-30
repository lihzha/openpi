import json
import os
import sys

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

sys.path.append("/n/fs/robot-data/vlm-syn")
from scripts.visualize_gripper import (
    describe_movement,
)

os.environ["CURL_CA_BUNDLE"] = (
    "/etc/pki/tls/certs/ca-bundle.crt"  # Ensure the CA bundle is set for SSL verification
)

data_name = "droid"
# data_dir = "gs://gresearch/robotics"
# data_dir = "/n/fs/vla-mi/datasets/OXE/"
data_dir = "/n/fs/robot-data/"
save_dir = "/n/fs/robot-data/vlm-syn/droid_lang_actions_json/"
os.makedirs(save_dir, exist_ok=True)


# builder = tfds.builder_from_directory(builder_dir="gs://gresearch/robotics/droid/1.0.1")
# ds = builder.as_dataset(split="test")

ds = tfds.load(data_name, data_dir=data_dir, split="train")
path_to_droid_repo = "/n/fs/robot-data/vlm-syn/droid_utils"

ds = ds.prefetch(tf.data.AUTOTUNE)
ds = ds.map(lambda x: x, num_parallel_calls=tf.data.AUTOTUNE)

# Load the extrinsics
cam2base_extrinsics_path = f"{path_to_droid_repo}/cam2base_extrinsics.json"
with open(cam2base_extrinsics_path, "r") as f:
    cam2base_extrinsics = json.load(f)

# Load the intrinsics
intrinsics_path = f"{path_to_droid_repo}/intrinsics.json"
with open(intrinsics_path, "r") as f:
    intrinsics = json.load(f)

# Load mapping from episode ID to path, then invert
episode_id_to_path_path = f"{path_to_droid_repo}/episode_id_to_path.json"
with open(episode_id_to_path_path, "r") as f:
    episode_id_to_path = json.load(f)
episode_path_to_id = {v: k for k, v in episode_id_to_path.items()}

# Load camera serials
camera_serials_path = f"{path_to_droid_repo}/camera_serials.json"
with open(camera_serials_path, "r") as f:
    camera_serials = json.load(f)

# Counter for tracking successful visualizations
successful_gripper_visualizations = 0
total_processed_episodes = 0


def get_gripper_cam_pose(extracted_extrinsics, extracted_intrinsics, calib_image_name, example):
    """Takes one episode dict, computes cam-frame poses, returns new dict."""
    # ------------- unpack bookkeeping -------------

    pos = extracted_extrinsics[0:3]  # translation
    rot_mat = R.from_euler("xyz", extracted_extrinsics[3:6]).as_matrix()  # rotation

    # Make homogenous transformation matrix
    cam_to_base_extrinsics_matrix = np.eye(4)
    cam_to_base_extrinsics_matrix[:3, :3] = rot_mat
    cam_to_base_extrinsics_matrix[:3, 3] = pos

    # print("Extrinsics:", cam_to_base_extrinsics_matrix)

    # Save all observations for the calibrated camera and corresponding gripper positions
    images = []
    cartesian_poses = []
    actions = []
    gripper_actions = []
    for curr_step in example["steps"]:
        image = curr_step["observation"][calib_image_name].numpy()
        images.append(image)
        actions.append(curr_step["action"].numpy())
        gripper_actions.append(curr_step["action_dict"]["gripper_position"].numpy().item())
        cartesian_pose = curr_step["observation"]["cartesian_position"].numpy()
        cartesian_poses.append(cartesian_pose)

    gripper_actions = np.array(gripper_actions)
    # Get the first

    # length images x 6
    cartesian_poses = np.array(cartesian_poses)
    positions = cartesian_poses[:, :3]  # Only take the position part
    rotations = cartesian_poses[:, 3:6]  # Only take the rotation

    gripper_pose_matrices = np.eye(4)[np.newaxis, :, :]  # Initialize with identity matrices
    gripper_pose_matrices = np.repeat(gripper_pose_matrices, len(positions), axis=0)
    gripper_pose_matrices[:, :3, 3] = positions  # Set the translation part
    gripper_pose_matrices[:, :3, :3] = R.from_euler(
        "xyz", rotations
    ).as_matrix()  # Set the rotation part

    gripper_pose_in_camera_frame = (
        np.linalg.inv(cam_to_base_extrinsics_matrix) @ gripper_pose_matrices.T
    ).T
    return gripper_pose_in_camera_frame, images, gripper_actions


for i, example in enumerate(tqdm(ds)):
    file_path = example["episode_metadata"]["file_path"].numpy().decode()
    episode_path = file_path.split("r2d2-data-full/")[1].split("/trajectory")[0]
    if episode_path not in episode_path_to_id:
        continue
    episode_id = episode_path_to_id[episode_path]

    print(f"\n--- Processing episode {episode_id} ({i + 1}/{len(ds)}) ---")

    if os.path.exists(os.path.join(save_dir, f"{episode_id}_language_action.json")):
        print(f"Already processed {episode_id}")
        continue  # already processed
    if episode_id not in cam2base_extrinsics:
        continue
    if episode_id not in intrinsics:
        continue
    if episode_id not in camera_serials:
        continue

    # pick the calibration camera for this episode (as you did earlier)
    extr = cam2base_extrinsics[episode_id]
    intr = intrinsics[episode_id]
    cams = camera_serials[episode_id]

    # ----------- compute gripper pose in cam frame -----------
    for k, v in extr.items():
        if k.isdigit():
            camera_serial = k
            extracted_extrinsics = v
            break

    # Also lets us get the intrinsics
    if camera_serial not in intr:
        continue
    extracted_intrinsics = intr[camera_serial]

    # Using the camera serial, find the corresponding camera name (which is used to determine
    # which image stream in the episode to use)
    camera_serials_to_name = {v: k for k, v in cams.items()}
    if camera_serial not in camera_serials_to_name:
        continue
    calib_camera_name = camera_serials_to_name[camera_serial]

    if calib_camera_name == "ext1_cam_serial":
        calib_image_name = "exterior_image_1_left"
    elif calib_camera_name == "ext2_cam_serial":
        calib_image_name = "exterior_image_2_left"
    else:
        raise ValueError(f"Unknown camera name: {calib_camera_name}")

    print(f"Camera with calibration data: {calib_camera_name} --> {calib_image_name}")
    # Convert the extrinsics to a homogeneous transformation matrix

    gripper_pose_cam, images, gripper_actions = get_gripper_cam_pose(
        extracted_extrinsics, extracted_intrinsics, calib_image_name, example
    )

    # Verify that we have matching numbers of images and gripper poses
    if len(images) != len(gripper_pose_cam):
        print(
            f"Warning: Mismatch between images ({len(images)}) and gripper poses ({len(gripper_pose_cam)}) for episode {episode_id}"
        )
        continue

    print(f"  - Number of frames: {len(images)}")
    print(f"  - Number of gripper poses: {len(gripper_pose_cam)}")
    try:
        gripper_scalars = gripper_actions
        print(f"  - Extracted gripper scalars shape: {gripper_scalars.shape}")
    except Exception as e:
        print(f"Warning: failed to extract gripper scalars: {e}")
        gripper_scalars = None

    language_actions = describe_movement(
        gripper_poses=gripper_pose_cam, gripper_actions=gripper_actions
    )
    # language_actions is a list of string. Compress it and save it
    with open(os.path.join(save_dir, f"{episode_id}_language_action.json"), "w") as f:
        json.dump(language_actions, f)
        print(f"Saved {episode_id}_language_action.json")

    # Extract camera intrinsics for visualization
    fx, cx, fy, cy = extracted_intrinsics["cameraMatrix"]
    camera_intrinsics = [fx, fy, cx, cy]

    # Validate camera intrinsics
    if any(val <= 0 for val in [fx, fy]) or any(val < 0 for val in [cx, cy]):
        print(
            f"Warning: Invalid camera intrinsics for episode {episode_id}: fx={fx}, fy={fy}, cx={cx}, cy={cy}"
        )
        continue

    # Create camera extrinsics matrix for visualization
    pos = extracted_extrinsics[0:3]  # translation
    rot_mat = R.from_euler("xyz", extracted_extrinsics[3:6]).as_matrix()  # rotation
    cam_to_base_extrinsics_matrix = np.eye(4)
    cam_to_base_extrinsics_matrix[:3, :3] = rot_mat
    cam_to_base_extrinsics_matrix[:3, 3] = pos

    # Validate camera extrinsics
    if np.any(np.isnan(cam_to_base_extrinsics_matrix)) or np.any(
        np.isinf(cam_to_base_extrinsics_matrix)
    ):
        print(f"Warning: Invalid camera extrinsics for episode {episode_id}")
        continue

    # Convert gripper poses from camera frame back to base frame for visualization
    # (since the visualization function expects base frame poses)
    # The transformation chain is: camera_frame -> base_frame -> camera_frame (for projection)
    gripper_poses_base = (cam_to_base_extrinsics_matrix @ gripper_pose_cam.T).T

    # Validate gripper poses
    if len(gripper_poses_base) == 0:
        print(f"Warning: No valid gripper poses for episode {episode_id}, skipping visualization")
        continue

    if np.any(np.isnan(gripper_poses_base)) or np.any(np.isinf(gripper_poses_base)):
        print(
            f"Warning: Invalid gripper poses detected for episode {episode_id}, skipping visualization"
        )
        continue

    # visualize_movement_no_arrow(
    #     frames=images,
    #     sentences=language_actions,
    #     gripper_poses=gripper_poses_base,  # Pass gripper poses in base frame
    #     camera_intrinsics=camera_intrinsics,  # Pass camera intrinsics
    #     camera_extrinsics=cam_to_base_extrinsics_matrix,  # Pass camera extrinsics
    #     gripper_scalars=gripper_scalars,
    #     out_path=os.path.join(save_dir, f"{episode_id}_motion_vis.mp4"),
    #     fps=0.5,
    #     subsample_factor=15,
    # )
    # print(
    #     f"Successfully created visualization with gripper points for episode {episode_id}"
    # )
    # successful_gripper_visualizations += 1
    # if successful_gripper_visualizations > 10:
    #     break


print("Summary:")
print(f"  - Total episodes processed: {total_processed_episodes}")
# print(f"  - Successful gripper visualizations: {successful_gripper_visualizations}")
