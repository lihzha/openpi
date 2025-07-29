import json
import os
import shutil

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import tensorflow_datasets as tfds
from tqdm import tqdm

os.environ["CURL_CA_BUNDLE"] = "/etc/pki/tls/certs/ca-bundle.crt"  # Ensure the CA bundle is set for SSL verification


def process_pi_dataset(
    dataset_name="",
    push_to_hub=False,
):
    # Create output directory
    output_path = dataset_name
    # os.makedirs(output_path, exist_ok=True)
    if os.path.exists(f"/n/fs/robot-data/cache/huggingface/lerobot/{output_path}"):
        if os.path.exists(f"/n/fs/robot-data/cache/huggingface/lerobot/{output_path}/data"):
            shutil.rmtree(f"/n/fs/robot-data/cache/huggingface/lerobot/{output_path}")
            print(f"Dataset {output_path} already exists, skipping...")
            # return
        # shutil.rmtree(f"/n/fs/robot-data/cache/huggingface/lerobot/{output_path}")
    dataset = LeRobotDataset.create(
        repo_id=output_path,
        robot_type="panda",
        fps=15,
        features={
            "image": {
                "dtype": "image",
                "shape": (180, 320, 3),
                "names": ["height", "width", "channel"],
            },
            # "wrist_image": {
            #     "dtype": "image",
            #     "shape": (480, 640, 3),
            #     "names": ["height", "width", "channel"],
            # },
            "state": {
                "dtype": "float64",
                "shape": (6,),
                "names": ["state"],
            },
            "actions": {
                "dtype": "float64",
                "shape": (7,),
                "names": ["actions"],
            },
            "language_actions": {
                "dtype": "string",
                "shape": (1,),
                "names": ["language_actions"],
            },
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    data_name = "droid_100"
    data_dir = "gs://gresearch/robotics"
    save_dir = "/n/fs/robot-data/vlm-syn/posed_droid"
    path_to_droid_repo = "/n/fs/robot-data/vlm-syn/droid"

    ds = tfds.load(data_name, data_dir=data_dir, split="train")

    # Load the extrinsics
    cam2base_extrinsics_path = f"{path_to_droid_repo}/cam2base_extrinsics.json"
    with open(cam2base_extrinsics_path) as f:
        cam2base_extrinsics = json.load(f)

    # Load mapping from episode ID to path, then invert
    episode_id_to_path_path = f"{path_to_droid_repo}/episode_id_to_path.json"
    with open(episode_id_to_path_path) as f:
        episode_id_to_path = json.load(f)

    # Load camera serials
    camera_serials_path = f"{path_to_droid_repo}/camera_serials.json"
    with open(camera_serials_path) as f:
        camera_serials = json.load(f)

    episode_path_to_id = {v: k for k, v in episode_id_to_path.items()}

    for idx, example in enumerate(tqdm(ds)):
        file_path = example["episode_metadata"]["file_path"].numpy().decode()
        episode_path = file_path.split("r2d2-data-full/")[1].split("/trajectory")[0]
        if episode_path not in episode_path_to_id:
            continue
        episode_id = episode_path_to_id[episode_path]
        if episode_id not in cam2base_extrinsics:
            continue
        extr = cam2base_extrinsics[episode_id]
        cams = camera_serials[episode_id]
        for k in extr:
            if k.isdigit():
                camera_serial = k
                break

        camera_serials_to_name = {v: k for k, v in cams.items()}
        calib_camera_name = camera_serials_to_name[camera_serial]
        if calib_camera_name == "ext1_cam_serial":
            calib_image_name = "exterior_image_1_left"
        elif calib_camera_name == "ext2_cam_serial":
            calib_image_name = "exterior_image_2_left"
        else:
            raise ValueError(f"Unknown camera name: {calib_camera_name}")
        # if episode_id not in cam2base_extrinsics:
        #     continue
        with open(os.path.join(save_dir, f"{episode_id}_language_action.json")) as f:
            language_actions = json.load(f)

        language_actions.append("stop")

        for t, curr_step in enumerate(example["steps"]):
            dataset.add_frame(
                {
                    "image": curr_step["observation"][calib_image_name].numpy(),
                    "state": curr_step["observation"]["cartesian_position"].numpy(),
                    "actions": curr_step["action"].numpy(),
                    "language_actions": language_actions[t],
                    "task": curr_step["language_instruction"].numpy().decode(),
                }
            )

        if idx >= 5:
            break

        dataset.save_episode()

    # Optionally push to the Hugging Face Hub
    if push_to_hub:
        dataset.push_to_hub(
            tags=["cot", "panda", "rlds"],
            private=False,
            push_videos=True,
            license="apache-2.0",
        )


if __name__ == "__main__":
    process_pi_dataset(
        dataset_name="posed_droid",
        push_to_hub=False,
    )
