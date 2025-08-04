"""
RLDS-based data loader for DROID with CoT-style language actions.

This revision merges the standalone pre-processing logic from
`process_dataset.py` directly into the `DroidCoTRldsDataset` class so
that everything from raw RLDS → ready-to-train batches happens in one
place.  The main additions are:

*   **Episode calibration lookup** – we pre-compute, on initialisation,
    which exterior camera (ext 1 vs ext 2) should be used for every
    DROID episode based on the extrinsics files.  At run-time the correct
    image is selected without expensive Python branching inside the
    TensorFlow graph.
*   **Language-action loading** – natural-language low-level action
    strings are loaded from the `<episode_id>_language_action.json`
    files and stored in a `tf.lookup.StaticHashTable`, so they can be
    joined with the trajectory entirely on the TF side.  The final batch
    therefore contains a `language_actions` tensor of shape
    `(B, T_chunk)` aligned with the action chunk.
*   **Restructure pass rewritten** – now relies on the lookup tables for
    both the calibrated image key and language actions; the remaining
    logic stays in TF ops (no `py_function`) so the pipeline is fully
    traceable and fast.

Usage example
-------------
```python
loader = DroidCoTRldsDataset(
    data_dir="gs://gresearch/robotics",
    batch_size=32,
    action_space=DroidActionSpace.CARTESIAN_POSITION,
    language_action_dir="/n/fs/robot-data/vlm-syn/posed_droid",
)
for batch in loader:
    images   = batch["observation"]["image"]          # (B, L, H, W, 3)
    actions  = batch["actions"]                        # (B, L, 7)
    lang_act = batch["language_actions"]               # (B, L)
    ...
```

The rest of the public API and the chunking / filtering behaviour remain
unchanged.
"""

from __future__ import annotations

from enum import Enum
from enum import auto
import json
import os
from pathlib import Path

import psutil

METADATA_PATH = "/n/fs/robot-data/vlm-syn/droid"
IMAGE_LIST = [
    "exterior_image_1_left",
    "exterior_image_2_left",
]


def print_memory_usage(label):
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / (1024**2)  # in MB
    print(f"[{label}] Memory usage: {mem:.2f} MB")


class DroidActionSpace(Enum):
    """Action space for DROID dataset."""

    JOINT_POSITION = auto()
    JOINT_VELOCITY = auto()
    CARTESIAN_POSITION = auto()


class DroidRldsDataset:
    def __init__(
        self,
        data_dir: str,
        batch_size: int,
        *,  # Force keyword-only arguments
        shuffle: bool = True,
        action_chunk_size: int = 16,
        # We default to joint position actions, since they allow policy evaluation in simulation.
        action_space: DroidActionSpace = DroidActionSpace.JOINT_POSITION,
        max_loaded_steps_per_episode: int = 100,
        # Reduce this if you are running out of memory, but careful -- below ~100k shuffling is not sufficiently random.
        shuffle_buffer_size: int = 250_000,
        num_parallel_reads: int = -1,  # -1 == tf.data.AUTOTUNE -- hack to not import tf at top level
        num_parallel_calls: int = -1,  # -1 == tf.data.AUTOTUNE -- hack to not import tf at top level
    ):
        # Import tensorflow here to not make it mandatory in case RLDS data loader is not used.
        import dlimp as dl
        import tensorflow as tf
        import tensorflow_datasets as tfds

        # Configure Tensorflow with *no GPU devices* (to prevent clobber with PyTorch / JAX)
        tf.config.set_visible_devices([], "GPU")

        builder = tfds.builder("droid", data_dir=data_dir)
        dataset = dl.DLataset.from_rlds(builder, split="train", shuffle=shuffle, num_parallel_reads=num_parallel_reads)

        # Filter out any unsuccessful trajectories -- we use the file name to check this
        dataset = dataset.filter(
            lambda traj: tf.strings.regex_full_match(
                traj["traj_metadata"]["episode_metadata"]["file_path"][0], ".*success.*"
            )
        )

        # Repeat dataset so we never run out of data.
        dataset = dataset.repeat()

        def restructure(traj):
            """Reformat observation and action keys, sample language instruction."""
            # Important: we use joint *position* action space -- easier to simulate!
            actions = tf.concat(
                (
                    (
                        traj["action_dict"]["joint_position"]
                        if action_space == DroidActionSpace.JOINT_POSITION
                        else traj["action_dict"]["joint_velocity"]
                    ),
                    traj["action_dict"]["gripper_position"],
                ),
                axis=-1,
            )
            # Randomly samples one of the two exterior images in DROID during training (we only train with one at a time).
            # Note: the "left" refers to the left camera in the stereo pair, we only train on the left camera.
            exterior_img = tf.cond(
                tf.random.uniform(shape=[]) > 0.5,
                lambda: traj["observation"]["exterior_image_1_left"],
                lambda: traj["observation"]["exterior_image_2_left"],
            )
            wrist_img = traj["observation"]["wrist_image_left"]
            # Randomly sample one of the three language instructions
            instruction = tf.random.shuffle(
                [traj["language_instruction"], traj["language_instruction_2"], traj["language_instruction_3"]]
            )[0]

            return {
                "actions": actions,
                "observation": {
                    "image": exterior_img,
                    "wrist_image": wrist_img,
                    "joint_position": traj["observation"]["joint_position"],
                    "gripper_position": traj["observation"]["gripper_position"],
                },
                "prompt": instruction,
            }

        dataset = dataset.traj_map(restructure, num_parallel_calls)

        def chunk_actions(traj):
            """Splits episode into action chunks."""
            traj_len = tf.shape(traj["actions"])[0]

            # For each step in the trajectory, construct indices for the next n actions
            action_chunk_indices = tf.broadcast_to(
                tf.range(action_chunk_size)[None],
                [traj_len, action_chunk_size],
            ) + tf.broadcast_to(
                tf.range(traj_len)[:, None],
                [traj_len, action_chunk_size],
            )

            # Cap to length of the sequence --> final chunks will repeat the last action
            # This makes sense, since we are using absolute joint + gripper position actions
            action_chunk_indices = tf.minimum(action_chunk_indices, traj_len - 1)

            # Gather the actions for each chunk
            traj["actions"] = tf.gather(traj["actions"], action_chunk_indices)
            return traj

        dataset = dataset.traj_map(chunk_actions, num_parallel_calls)

        def filter_idle(traj):
            """Filter out chunks with idle actions.
            --> we filter if at least first half of chunk does not move.
            """
            if action_space == DroidActionSpace.JOINT_POSITION:
                # Compute delta to first position in action chunk
                return tf.reduce_any(tf.abs(traj["actions"][: action_chunk_size // 2] - traj["actions"][:1]) > 1e-3)
            return tf.reduce_any(tf.abs(traj["actions"][: action_chunk_size // 2]) > 1e-3)

        dataset = dataset.filter(filter_idle)

        # Flatten: map from trajectory dataset to dataset of individual action chunks
        dataset = dataset.flatten(num_parallel_calls=num_parallel_calls)

        # Decode images: RLDS saves encoded images, only decode now for efficiency
        def decode_images(traj):
            traj["observation"]["image"] = tf.io.decode_image(
                traj["observation"]["image"], expand_animations=False, dtype=tf.uint8
            )
            traj["observation"]["wrist_image"] = tf.io.decode_image(
                traj["observation"]["wrist_image"], expand_animations=False, dtype=tf.uint8
            )
            return traj

        dataset = dataset.frame_map(decode_images, num_parallel_calls)

        # Shuffle, batch
        dataset = dataset.shuffle(shuffle_buffer_size)
        dataset = dataset.batch(batch_size)
        # Note =>> Seems to reduce memory usage without affecting speed?
        dataset = dataset.with_ram_budget(1)

        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        yield from self.dataset.as_numpy_iterator()

    def __len__(self):
        # This is the approximate number of samples in DROID after filtering.
        # Easier to hardcode than to iterate through the dataset and compute it.
        return 20_000_000


class DroidCoTRldsDataset:
    def __init__(
        self,
        data_dir: str,
        batch_size: int,
        language_action_dir: str,
        *,  # Force keyword-only arguments
        shuffle: bool = True,
        action_chunk_size: int = 16,
        # We default to joint position actions, since they allow policy evaluation in simulation.
        action_space: DroidActionSpace = DroidActionSpace.CARTESIAN_POSITION,
        max_loaded_steps_per_episode: int = 100,
        # Reduce this if you are running out of memory, but careful -- below ~100k shuffling is not sufficiently random.
        shuffle_buffer_size: int = 250_000,
        num_parallel_reads: int = -1,  # -1 == tf.data.AUTOTUNE -- hack to not import tf at top level
        num_parallel_calls: int = -1,  # -1 == tf.data.AUTOTUNE -- hack to not import tf at top level
    ):
        # Import tensorflow here to not make it mandatory in case RLDS data loader is not used.
        import dlimp as dl
        import tensorflow as tf
        import tensorflow_datasets as tfds

        assert action_space == DroidActionSpace.CARTESIAN_POSITION, "CoT only supports EEF actions for now"

        # Configure Tensorflow with *no GPU devices* (to prevent clobber with PyTorch / JAX)
        tf.config.set_visible_devices([], "GPU")

        # builder = tfds.builder("droid", data_dir=data_dir)
        builder = tfds.builder("droid", data_dir="gs://gresearch/robotics", split="train")

        dataset = dl.DLataset.from_rlds(builder, split="train", shuffle=shuffle, num_parallel_reads=num_parallel_reads)

        print_memory_usage("Before table building")

        # 1. build episode_id table
        lang_action_files = os.listdir(language_action_dir)
        valid_eids = [f.split("_language_action.json")[0] for f in lang_action_files if "_language_action.json" in f]
        keys = tf.constant(valid_eids, dtype=tf.string)
        values = tf.ones(  # all 1/True → “keep”
            shape=(len(valid_eids),),
            dtype=tf.bool,  # or tf.int32 / tf.int64
        )
        eid_table = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(keys, values),
            default_value=tf.constant(value=False, dtype=values.dtype),  # must match
        )

        print_memory_usage("After building eid_table")

        # 2. build language action table
        episodes, lang_serialized = [], []
        for f in Path(language_action_dir).glob("*_language_action.json"):
            eid = f.stem.replace("_language_action", "")
            with f.open("r") as fp:
                lst = json.load(fp)  # list of str (len == #steps)
                lst.append("stop")
            t = tf.constant(lst, dtype=tf.string)  # shape (T,)
            lang_serialized.append(tf.io.serialize_tensor(t).numpy())
            episodes.append(eid)
        keys = tf.constant(episodes, dtype=tf.string)
        values = tf.constant(lang_serialized, dtype=tf.string)
        lang_table = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(keys, values),
            default_value=tf.constant(b"", dtype=tf.string),
        )  # other options other than a lookup table: 1. use tf.numpy_function in restructure - runs under the Python GIL → one element at a time,
        # no vectorisation, no TPU support. 2. use tf.io.read_file -> per-element disk I/O, not as fast

        print_memory_usage("After building lang_table")

        # 3. build episode path table
        with open(f"{METADATA_PATH}/episode_id_to_path.json") as f:
            episode_id_to_path = json.load(f)
        episode_path_to_id = {v: k for k, v in episode_id_to_path.items()}
        keys = tf.constant(list(episode_path_to_id.keys()), dtype=tf.string)
        values = tf.constant(list(episode_path_to_id.values()), dtype=tf.string)
        default_value = tf.constant(value="", dtype=tf.string)
        ep_table = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(keys, values),
            default_value=default_value,
        )

        print_memory_usage("After building ep_table")

        # 4. build the camera idx table
        with open(f"{METADATA_PATH}/cam2base_extrinsics.json") as f:
            cam2base_extrinsics = json.load(f)
        with open(f"{METADATA_PATH}/camera_serials.json") as f:
            camera_serials = json.load(f)
        eid_to_cam_dict = {}
        for eid in cam2base_extrinsics:
            extr = cam2base_extrinsics[eid]
            cams = camera_serials[eid]
            for k in extr:
                if k.isdigit():
                    camera_serial = k
                    break
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
            calib_image_idx = IMAGE_LIST.index(calib_image_name)
            eid_to_cam_dict[eid] = calib_image_idx
        keys = tf.constant(list(eid_to_cam_dict.keys()), dtype=tf.string)
        values = tf.constant(list(eid_to_cam_dict.values()), dtype=tf.int32)
        cam_table = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(keys, values),
            default_value=-1,  # -1 ⇒ “use a fallback camera”
        )

        print_memory_usage("After building cam_table")

        # 5. build language instruction
        with open(f"{METADATA_PATH}/droid_language_annotations.json") as f:
            language_annotations = json.load(f)
        keys = tf.constant(list(language_annotations.keys()), dtype=tf.string)
        values_1 = tf.constant(
            [v["language_instruction1"] for v in list(language_annotations.values())], dtype=tf.string
        )
        instr_table_1 = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(keys, values_1),
            default_value="",  # "" ⇒ “use a fallback instruction”
        )
        values_2 = tf.constant(
            [v["language_instruction2"] for v in list(language_annotations.values())], dtype=tf.string
        )
        instr_table_2 = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(keys, values_2),
            default_value="",  # "" ⇒ “use a fallback instruction”
        )
        values_3 = tf.constant(
            [v["language_instruction3"] for v in list(language_annotations.values())], dtype=tf.string
        )
        instr_table_3 = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(keys, values_3),
            default_value="",  # "" ⇒ “use a fallback instruction”
        )
        fallback_instructions = tf.constant(
            [
                "Do something useful.",
                "Complete the task.",
                "Perform the task.",
                "Carry out the objective.",
                "Execute the current task.",
                "Accomplish the goal.",
                "Proceed with the task.",
                "Handle the task at hand.",
                "Continue the operation.",
                "Fulfill the task.",
                "Take meaningful steps.",
                "Demonstrate useful behavior.",
                "Act in a useful manner.",
                "Engage in productive actions.",
                "Make useful moves.",
                "Undertake useful actions.",
                "Behave purposefully.",
                "Start the activity.",
            ],
            dtype=tf.string,
        )

        print_memory_usage("After building instr_table")

        def _id_ok(traj):
            file_path = traj["traj_metadata"]["episode_metadata"]["file_path"][0]
            after_prefix = tf.strings.split(file_path, "r2d2-data-full/")[1]
            episode_path = tf.strings.split(after_prefix, "/trajectory")[0]
            episode_id = ep_table.lookup(episode_path)
            if tf.equal(episode_id, default_value):
                return tf.constant(value=False, dtype=tf.bool)
            return eid_table.lookup(episode_id)

        def _path_ok(traj):
            file_path = traj["traj_metadata"]["episode_metadata"]["file_path"][0]
            return tf.strings.regex_full_match(file_path, ".*success.*")

        # Filter out any unsuccessful trajectories -- we use the file name to check this
        dataset = (
            dataset.filter(_id_ok).filter(_path_ok)  # cheap O(1) hash lookup  # regex only on the survivors
            # .cache() .shuffle() .prefetch(...)  ↳ whatever else you need
        )

        # Repeat dataset so we never run out of data.
        dataset = dataset.repeat()

        def restructure(traj):
            """Reformat observation and action keys, sample language instruction."""
            # Important: we use joint *position* action space -- easier to simulate!
            actions = tf.concat(
                (
                    traj["action_dict"]["cartesian_position"],
                    traj["action_dict"]["gripper_position"],
                ),
                axis=-1,
            )
            file_path = traj["traj_metadata"]["episode_metadata"]["file_path"][0]
            after_prefix = tf.strings.split(file_path, "r2d2-data-full/")[1]
            episode_path = tf.strings.split(after_prefix, "/trajectory")[0]
            episode_id = ep_table.lookup(episode_path)
            lang_bytes = lang_table.lookup(episode_id)
            lang_tensor = tf.io.parse_tensor(lang_bytes, tf.string)
            instruction_1 = instr_table_1.lookup(episode_id)
            instruction_2 = instr_table_2.lookup(episode_id)
            instruction_3 = instr_table_3.lookup(episode_id)
            # Check which instruction is non-empty, and form a non-empty list of instructions.
            # If all instrcutions are empty, raise an error. Then sample one instruction from the list.
            instructions = tf.stack([instruction_1, instruction_2, instruction_3])
            mask = tf.strings.length(instructions) > 0
            non_empty = tf.boolean_mask(instructions, mask)
            num_valid = tf.shape(non_empty)[0]

            fallback_index = tf.random.uniform((), minval=0, maxval=tf.shape(fallback_instructions)[0], dtype=tf.int32)
            fallback_instruction = fallback_instructions[fallback_index]
            instruction = tf.cond(
                num_valid > 0,
                lambda: tf.random.shuffle(non_empty)[0],
                lambda: fallback_instruction,
            )

            instruction_vec = tf.fill([tf.shape(actions)[0]], instruction)

            cam_idx = cam_table.lookup(episode_path)
            cam_images = [
                traj["observation"]["exterior_image_1_left"],
                traj["observation"]["exterior_image_2_left"],
            ]
            cam_images = tf.stack(cam_images, axis=0)  # shape (3, H, W, C)
            cam_idx_clamped = tf.where(cam_idx < 0, 0, cam_idx)
            exterior_img = tf.gather(cam_images, cam_idx_clamped)

            # TODO: use wrist camera image or not
            # # Randomly samples one of the two exterior images in DROID during training (we only train with one at a time).
            # # Note: the "left" refers to the left camera in the stereo pair, we only train on the left camera.
            # exterior_img = tf.cond(
            #     tf.random.uniform(shape=[]) > 0.5,
            #     lambda: traj["observation"]["exterior_image_1_left"],
            #     lambda: traj["observation"]["exterior_image_2_left"],
            # )
            # wrist_img = traj["observation"]["wrist_image_left"]

            traj_len = tf.shape(actions)[0]
            episode_id_vec = tf.fill([traj_len], episode_id)

            return {
                "actions": actions,
                "observation": {
                    "image": exterior_img,
                    # "wrist_image": wrist_img,
                    "cartesian_position": traj["observation"]["cartesian_position"],
                    "gripper_position": traj["observation"]["gripper_position"],
                },
                "prompt": instruction_vec,
                "language_actions": lang_tensor,
                "episode_id": episode_id_vec,
            }

        dataset = dataset.traj_map(restructure, num_parallel_calls)

        # TODO: chunk reasoning as well depending on the frequency ratio between reasoning and actions
        def chunk_actions(traj):
            """Splits episode into action chunks."""
            traj_len = tf.shape(traj["actions"])[0]

            # For each step in the trajectory, construct indices for the next n actions
            action_chunk_indices = tf.broadcast_to(
                tf.range(action_chunk_size)[None],
                [traj_len, action_chunk_size],
            ) + tf.broadcast_to(
                tf.range(traj_len)[:, None],
                [traj_len, action_chunk_size],
            )

            # Cap to length of the sequence --> final chunks will repeat the last action
            # This makes sense, since we are using absolute joint + gripper position actions
            action_chunk_indices = tf.minimum(action_chunk_indices, traj_len - 1)

            # Gather the actions for each chunk
            traj["actions"] = tf.gather(traj["actions"], action_chunk_indices)
            return traj

        dataset = dataset.traj_map(chunk_actions, num_parallel_calls)

        def chunk_language_actions(traj):
            """Splits episode into action chunks."""
            traj_len = tf.shape(traj["language_actions"])[0]

            # For each step in the trajectory, construct indices for the next n actions
            action_chunk_indices = tf.broadcast_to(
                tf.range(action_chunk_size)[None],
                [traj_len, action_chunk_size],
            ) + tf.broadcast_to(
                tf.range(traj_len)[:, None],
                [traj_len, action_chunk_size],
            )

            # Cap to length of the sequence --> final chunks will repeat the last action
            # This makes sense, since we are using absolute joint + gripper position actions
            action_chunk_indices = tf.minimum(action_chunk_indices, traj_len - 1)

            # Gather the actions for each chunk
            traj["language_actions"] = tf.gather(traj["language_actions"], action_chunk_indices)
            return traj

        # TODO: chunk action or not
        # dataset = dataset.traj_map(chunk_language_actions, num_parallel_calls)

        def filter_idle(traj):
            """Filter out chunks with idle actions.
            --> we filter if at least first half of chunk does not move.
            """
            if action_space == DroidActionSpace.CARTESIAN_POSITION:
                # Compute delta to first position in action chunk
                return tf.reduce_any(tf.abs(traj["actions"][: action_chunk_size // 2] - traj["actions"][:1]) > 1e-3)
            return tf.reduce_any(tf.abs(traj["actions"][: action_chunk_size // 2]) > 1e-3)

        dataset = dataset.filter(filter_idle)

        # Flatten: map from trajectory dataset to dataset of individual action chunks
        dataset = dataset.flatten(num_parallel_calls=num_parallel_calls)

        # Decode images: RLDS saves encoded images, only decode now for efficiency
        def decode_images(traj):
            traj["observation"]["image"] = tf.io.decode_image(
                traj["observation"]["image"], expand_animations=False, dtype=tf.uint8
            )
            # traj["observation"]["wrist_image"] = tf.io.decode_image(
            #     traj["observation"]["wrist_image"], expand_animations=False, dtype=tf.uint8
            # )
            return traj

        dataset = dataset.shuffle(shuffle_buffer_size)

        dataset = dataset.frame_map(decode_images, num_parallel_calls)

        # Shuffle, batch
        dataset = dataset.batch(batch_size)
        # Note =>> Seems to reduce memory usage without affecting speed?
        dataset = dataset.with_ram_budget(1)

        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        yield from self.dataset.as_numpy_iterator()

    def __len__(self):
        # This is the approximate number of samples in DROID after filtering.
        # Easier to hardcode than to iterate through the dataset and compute it.
        return 20_000_000


if __name__ == "__main__":
    import numpy as np

    ds = DroidCoTRldsDataset(
        data_dir="/n/fs/vla-mi/datasets/OXE/",
        language_action_dir="/n/fs/robot-data/vlm-syn/posed_droid",
        batch_size=32,
        shuffle_buffer_size=200,
    )
    ds = iter(ds)
    all_eids = []
    for f in Path("/n/fs/robot-data/vlm-syn/posed_droid").glob("*_language_action.json"):
        eid = f.stem.replace("_language_action", "")
        all_eids.append(eid)

    with open(f"{METADATA_PATH}/droid_language_annotations.json") as f:
        language_annotations = json.load(f)
        all_lang_eids = list(language_annotations.keys())
    total_empty = 0
    for i, batch in enumerate(ds):
        if np.any(batch["prompt"] == b"Do something useful."):
            # count the number of "Do something useful." prompts
            total_empty += np.sum(batch["prompt"] == b"Do something useful.")
            propotion = total_empty / (i + 1) / 32
            print(f"Iter {i}, Total empty prompts: {total_empty}, Propotion: {propotion:.2f}")
        for raw_eid in batch["episode_id"]:
            eid = raw_eid.decode()
            assert eid in all_eids, f"Episode ID {eid} not found in the list of valid episode IDs."
            # assert eid in all_lang_eids, f"Episode ID {eid} not found in the language annotations."
            if eid not in all_lang_eids:
                print(f"Episode ID {eid} not found in the language annotations.")
