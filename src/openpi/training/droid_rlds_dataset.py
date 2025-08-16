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
import logging
import os
from pathlib import Path
import time

import jax
import numpy as np
import psutil

METADATA_PATH = "/n/fs/robot-data/vlm-syn/droid"
IMAGE_LIST = [
    "exterior_image_1_left",
    "exterior_image_2_left",
]

# Enable lightweight timing logs when set to "1"
DEBUG_TIMING = os.environ.get("OPENPI_TIMING", "0") == "1"


def print_memory_usage(label):
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / (1024**2)  # in MB
    logging.info(f"[{label}] Memory usage: {mem:.2f} MB")


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

        # Configure Tensorflow with no GPU/TPU devices to avoid clobbering JAX/TPU runtime
        tf.config.set_visible_devices([], "GPU")
        try:
            tf.config.set_visible_devices([], "TPU")
        except Exception:
            pass

        # Resolve autotune sentinels now that TF is imported
        if num_parallel_reads == -1:
            num_parallel_reads = tf.data.AUTOTUNE
        if num_parallel_calls == -1:
            num_parallel_calls = tf.data.AUTOTUNE

        builder = tfds.builder("droid", data_dir=data_dir)
        dataset = dl.DLataset.from_rlds(
            builder,
            split="train",
            shuffle=shuffle,
            num_parallel_reads=num_parallel_reads,
        )

        # Enable non-deterministic mapping and other tf.data optimizations for throughput
        opts = tf.data.Options()
        opts.experimental_deterministic = False
        dataset = dataset.with_options(opts)

        # Host-shard the dataset across JAX processes to avoid duplicated work and I/O on TPUs.
        dataset = dataset.shard(jax.process_count(), jax.process_index())

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

        if DEBUG_TIMING:
            def _wrap_timed_map(fn, name):
                def _inner(x):
                    t0 = tf.timestamp()
                    y = fn(x)
                    t1 = tf.timestamp()
                    ms = (t1 - t0) * 1000.0
                    def _log(ms_np):
                        try:
                            logging.info(f"[tf.data] {name} ms={float(ms_np):.1f}")
                        except Exception:
                            pass
                        return np.int64(0)
                    _ = tf.py_function(_log, [ms], Tout=tf.int64)
                    return y
                return _inner
            dataset = dataset.traj_map(_wrap_timed_map(restructure, "restructure"), num_parallel_calls)
        else:
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

        if DEBUG_TIMING:
            dataset = dataset.traj_map(_wrap_timed_map(chunk_actions, "chunk_actions"), num_parallel_calls)
        else:
            dataset = dataset.traj_map(chunk_actions, num_parallel_calls)

        def filter_idle(traj):
            """Filter out chunks with idle actions.
            --> we filter if at least first half of chunk does not move.
            """
            if action_space == DroidActionSpace.JOINT_POSITION:
                # Compute delta to first position in action chunk
                return tf.reduce_any(tf.abs(traj["actions"][: action_chunk_size // 2] - traj["actions"][:1]) > 1e-3)
            return tf.reduce_any(tf.abs(traj["actions"][: action_chunk_size // 2]) > 1e-3)

        if DEBUG_TIMING:
            def _timed_filter(x):
                t0 = tf.timestamp()
                out = filter_idle(x)
                t1 = tf.timestamp()
                ms = (t1 - t0) * 1000.0
                def _log(ms_np):
                    try:
                        logging.info(f"[tf.data] filter_idle ms={float(ms_np):.1f}")
                    except Exception:
                        pass
                    return np.int64(0)
                _ = tf.py_function(_log, [ms], Tout=tf.int64)
                return out
            dataset = dataset.filter(_timed_filter)
        else:
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

        if DEBUG_TIMING:
            dataset = dataset.frame_map(_wrap_timed_map(decode_images, "decode_images"), num_parallel_calls)
        else:
            dataset = dataset.frame_map(decode_images, num_parallel_calls)

        # Shuffle, batch
        dataset = dataset.shuffle(shuffle_buffer_size)
        dataset = dataset.batch(batch_size)
        # Overlap input pipeline with consumers; lets TF fill a small buffer per host.
        dataset = dataset.prefetch(2)
        # Note =>> Seems to reduce memory usage without affecting speed?
        dataset = dataset.with_ram_budget(1)

        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        it = self.dataset.as_numpy_iterator()
        while True:
            t0 = time.perf_counter() if DEBUG_TIMING else 0.0
            try:
                batch = next(it)
            except StopIteration:
                return
            if DEBUG_TIMING:
                dt = (time.perf_counter() - t0) * 1000.0
                logging.info("DroidRldsDataset as_numpy_iterator.next: %.1f ms", dt)
            yield batch

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
        summation_steps: int = 5,  # Number of future steps to sum over for language actions
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

        tf.config.set_visible_devices([], "TPU")

        # ⇨ point all data + metadata directories to the GCS bucket

        if "pi0-cot" in data_dir:  # for v4
            METADATA_PATH = language_action_dir.replace("droid-lang-actions", "metadata")
        else:  # for v6
            assert "droid-cot" in data_dir
            METADATA_PATH = language_action_dir.replace("posed_droid", "metadata")

        # ---------------------------------------------------------------------
        # 1. TF-DS builder + base dataset
        # ---------------------------------------------------------------------
        # Resolve autotune sentinels now that TF is imported
        if num_parallel_reads == -1:
            num_parallel_reads = tf.data.AUTOTUNE
        if num_parallel_calls == -1:
            num_parallel_calls = tf.data.AUTOTUNE

        t_build_ds_start = time.perf_counter() if DEBUG_TIMING else 0.0
        builder = tfds.builder("droid", data_dir=data_dir)
        dataset = dl.DLataset.from_rlds(
            builder,
            split="train",
            shuffle=shuffle,
            num_parallel_reads=num_parallel_reads,
        )
        t_build_ds = (time.perf_counter() - t_build_ds_start) if DEBUG_TIMING else 0.0
        
        # dataset = dataset.with_ram_budget(1)
        t_shard_start = time.perf_counter() if DEBUG_TIMING else 0.0
        dataset = dataset.shard(jax.process_count(), jax.process_index())
        t_shard = (time.perf_counter() - t_shard_start) if DEBUG_TIMING else 0.0

        # Enable non-deterministic mapping and other tf.data optimizations for throughput
        opts = tf.data.Options()
        opts.experimental_deterministic = False
        dataset = dataset.with_options(opts)

        # ---------------------------------------------------------------------
        # 2. Language-action table (episode_id → serialized tensor)
        # ---------------------------------------------------------------------
        FEATURES = {
            "episode_id": tf.io.FixedLenFeature([], tf.string),
            "lang_ser":   tf.io.FixedLenFeature([], tf.string),
        }
        def _parse(record):
            ex = tf.io.parse_single_example(record, FEATURES)
            lang = tf.io.parse_tensor(ex["lang_ser"], out_type=tf.string)  # shape: [T+1]
            return ex["episode_id"], lang

        t_lang_scan_start = time.perf_counter() if DEBUG_TIMING else 0.0
        files = tf.io.gfile.glob(f"{language_action_dir}/droid_language_actions-*.tfrecord.gz")
        ds = tf.data.TFRecordDataset(
            files, compression_type="GZIP", num_parallel_reads=tf.data.AUTOTUNE
        ).map(_parse, num_parallel_calls=tf.data.AUTOTUNE
        ).prefetch(tf.data.AUTOTUNE)
        episodes, lang_serialized = [], []
        for ep_id, lang in ds:
            episodes.append(ep_id.numpy().decode())
            lang_serialized.append(tf.io.serialize_tensor(lang).numpy())
        t_lang_scan = (time.perf_counter() - t_lang_scan_start) if DEBUG_TIMING else 0.0

        keys = tf.constant(episodes, dtype=tf.string)
        values = tf.constant(lang_serialized, dtype=tf.string)
        default_lang_value = tf.constant(b"", dtype=tf.string)
        lang_table = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(keys, values),
            default_value=default_lang_value,
        )

        if DEBUG_TIMING:
            logging.info(
                "DroidCoTRldsDataset: built lang table for %d episodes in %.1f ms (build_ds=%.1f ms, shard=%.1f ms)",
                len(episodes), t_lang_scan * 1000.0, t_build_ds * 1000.0, t_shard * 1000.0
            )
        print_memory_usage("After building lang_table")
        
        # ---------------------------------------------------------------------
        # 3. Episode-ID table  (valid_eids → True)
        # ---------------------------------------------------------------------
        t_eid_start = time.perf_counter() if DEBUG_TIMING else 0.0
        keys = tf.constant(episodes, dtype=tf.string)
        values = tf.ones(len(episodes), dtype=tf.bool)
        eid_table = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(keys, values),
            default_value=tf.constant(False, dtype=tf.bool),
        )
        t_eid = (time.perf_counter() - t_eid_start) if DEBUG_TIMING else 0.0

        if DEBUG_TIMING:
            logging.info("DroidCoTRldsDataset: built eid table in %.1f ms", t_eid * 1000.0)
        print_memory_usage("After building eid_table")

        # ---------------------------------------------------------------------
        # 4. Episode-path ↔ Episode-ID table
        # ---------------------------------------------------------------------
        t_epmap_start = time.perf_counter() if DEBUG_TIMING else 0.0
        with tf.io.gfile.GFile(f"{METADATA_PATH}/episode_id_to_path.json", "r") as fp:
            episode_id_to_path = json.load(fp)
        episode_path_to_id = {v: k for k, v in episode_id_to_path.items()}

        keys = tf.constant(list(episode_path_to_id.keys()), dtype=tf.string)
        values = tf.constant(list(episode_path_to_id.values()), dtype=tf.string)
        default_ep_value = tf.constant("", dtype=tf.string)
        ep_table = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(keys, values),
            default_value=default_ep_value,
        )
        t_epmap = (time.perf_counter() - t_epmap_start) if DEBUG_TIMING else 0.0

        if DEBUG_TIMING:
            logging.info("DroidCoTRldsDataset: built episode path↔id table in %.1f ms", t_epmap * 1000.0)
        print_memory_usage("After building ep_table")

        # ---------------------------------------------------------------------
        # 5. Camera-index table  (episode_id → ext-cam idx)
        # ---------------------------------------------------------------------
        t_cam_start = time.perf_counter() if DEBUG_TIMING else 0.0
        with tf.io.gfile.GFile(f"{METADATA_PATH}/cam2base_extrinsics.json", "r") as fp:
            cam2base_extrinsics = json.load(fp)
        with tf.io.gfile.GFile(f"{METADATA_PATH}/camera_serials.json", "r") as fp:
            camera_serials = json.load(fp)

        eid_to_cam_dict = {}
        for eid, extr in cam2base_extrinsics.items():
            cams = camera_serials[eid]
            camera_serial = next(k for k in extr if k.isdigit())
            serial_to_name = {v: k for k, v in cams.items()}
            if camera_serial not in serial_to_name:
                continue

            calib_camera_name = serial_to_name[camera_serial]
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
            default_value=-1,  # -1 ⇒ fallback camera
        )
        t_cam = (time.perf_counter() - t_cam_start) if DEBUG_TIMING else 0.0

        if DEBUG_TIMING:
            logging.info("DroidCoTRldsDataset: built camera index table in %.1f ms", t_cam * 1000.0)
        print_memory_usage("After building cam_table")

        # ---------------------------------------------------------------------
        # 6. Language-instruction tables (3 per episode_id)
        # ---------------------------------------------------------------------
        t_instr_start = time.perf_counter() if DEBUG_TIMING else 0.0
        with tf.io.gfile.GFile(f"{METADATA_PATH}/droid_language_annotations.json", "r") as fp:
            language_annotations = json.load(fp)

        keys = tf.constant(list(language_annotations.keys()), dtype=tf.string)
        values_1 = tf.constant([v["language_instruction1"] for v in language_annotations.values()], dtype=tf.string)
        values_2 = tf.constant([v["language_instruction2"] for v in language_annotations.values()], dtype=tf.string)
        values_3 = tf.constant([v["language_instruction3"] for v in language_annotations.values()], dtype=tf.string)

        instr_table_1 = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(keys, values_1),
            default_value="",
        )
        instr_table_2 = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(keys, values_2),
            default_value="",
        )
        instr_table_3 = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(keys, values_3),
            default_value="",
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

        t_instr = (time.perf_counter() - t_instr_start) if DEBUG_TIMING else 0.0
        if DEBUG_TIMING:
            logging.info("DroidCoTRldsDataset: built instruction tables in %.1f ms", t_instr * 1000.0)
        print_memory_usage("After building instr_table")

        def _id_ok(traj):
            file_path = traj["traj_metadata"]["episode_metadata"]["file_path"][0]
            after_prefix = tf.strings.split(file_path, "r2d2-data-full/")[1]
            episode_path = tf.strings.split(after_prefix, "/trajectory")[0]
            episode_id = ep_table.lookup(episode_path)
            if tf.equal(episode_id, default_ep_value):
                return tf.constant(value=False, dtype=tf.bool)
            # Look up by episode_id (NOT episode_path). Using episode_path here would filter everything out.
            lang = lang_table.lookup(episode_id)
            if tf.equal(lang, default_lang_value):
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

        # dataset = dataset.traj_map(restructure, num_parallel_calls)

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

        if DEBUG_TIMING:
            def _wrap_timed_map(fn, name):
                def _inner(x):
                    t0 = tf.timestamp()
                    y = fn(x)
                    t1 = tf.timestamp()
                    ms = (t1 - t0) * 1000.0
                    def _log(ms_np):
                        try:
                            logging.info(f"[tf.data] {name} ms={float(ms_np):.1f}")
                        except Exception:
                            pass
                        return np.int64(0)
                    _ = tf.py_function(_log, [ms], Tout=tf.int64)
                    return y
                return _inner
            dataset = dataset.traj_map(_wrap_timed_map(restructure, "restructure"), num_parallel_calls)
            dataset = dataset.traj_map(_wrap_timed_map(chunk_actions, "chunk_actions"), num_parallel_calls)
        else:
            dataset = dataset.traj_map(restructure, num_parallel_calls)
            dataset = dataset.traj_map(chunk_actions, num_parallel_calls)

        def _sum_language_actions(language_actions_batch):
            """Helper function to sum over a batch of language actions.
            
            Args:
                language_actions_batch: Tensor of shape [summation_steps] containing language action strings
                
            Returns:
                A single string representing the summed language action
            """
            # Use py_function to implement the complex string parsing and summation logic
            # This allows us to use Python string operations while keeping the function traceable
            
            def _python_sum_language_actions(actions_np):
                """Python implementation of language action summation."""
                import re
                
                # Dictionary to store summed movements by direction
                movement_sums = {}
                
                # Define opposite directions for cancellation
                opposite_directions = {
                    'left': 'right',
                    'right': 'left',
                    'forward': 'backward',
                    'backward': 'forward',
                    'up': 'down',
                    'down': 'up'
                }
                
                # Convert possible EagerTensor to numpy array of bytes
                try:
                    actions_arr = actions_np.numpy()
                except AttributeError:
                    actions_arr = actions_np
                
                for action_str in actions_arr:
                    # Ensure we operate on a Python string
                    if isinstance(action_str, (bytes, bytearray)):
                        s = action_str.decode("utf-8")
                    else:
                        s = str(action_str)
                    if not s:
                        continue
                    
                    # Split by " and " to get individual movements
                    movements = s.split(" and ")
                    
                    for movement in movements:
                        # Parse movement: "move direction value unit"
                        # Use regex to handle variations in spacing
                        match = re.match(r'move\s+(\w+)\s+([\d.]+)\s*(\w+)', movement.strip())
                        if match:
                            direction = match.group(1)
                            value = float(match.group(2))
                            unit = match.group(3)
                            
                            # Check if we have an opposite direction already
                            opposite = opposite_directions.get(direction)
                            if opposite in movement_sums:
                                # Cancel out opposite movements
                                opposite_value = movement_sums[opposite]['value']
                                if value > opposite_value:
                                    # Current direction wins
                                    movement_sums[direction] = {'value': value - opposite_value, 'unit': unit}
                                    del movement_sums[opposite]
                                elif value < opposite_value:
                                    # Opposite direction wins
                                    movement_sums[opposite]['value'] = opposite_value - value
                                else:
                                    # Equal values, cancel out completely
                                    del movement_sums[opposite]
                            else:
                                # Initialize if direction not seen before
                                if direction not in movement_sums:
                                    movement_sums[direction] = {'value': 0.0, 'unit': unit}
                                
                                # Add the value
                                movement_sums[direction]['value'] += value
                
                # Build the result string
                if not movement_sums:
                    return ""
                
                result_parts = []
                for direction, data in movement_sums.items():
                    if data['value'] > 0:  # Only include positive values
                        result_parts.append(f"move {direction} {data['value']:.2f} {data['unit']}")
                
                return " and ".join(result_parts)
            
            # Convert TensorFlow tensor to numpy and back
            result = tf.py_function(
                _python_sum_language_actions,
                [language_actions_batch],
                tf.string
            )
            
            return result
        
        def group_language_actions(traj):
            """Compute per-timestep summed language actions over future steps.

            For each timestep t, we sum the language actions from t to
            t + summation_steps - 1 (capped at trajectory end). We DO NOT
            chunk the language actions; after flattening, each sample will
            have a single language string aligned to its action chunk.
            """
            traj_len = tf.shape(traj["language_actions"])[0]
            
            # First, create indices for summation (current + future steps)
            summation_indices = tf.broadcast_to(
                tf.range(summation_steps)[None],
                [traj_len, summation_steps],
            ) + tf.broadcast_to(
                tf.range(traj_len)[:, None],
                [traj_len, summation_steps],
            )
            
            # Cap to length of the sequence (same as chunk_actions)
            summation_indices = tf.minimum(summation_indices, traj_len - 1)
            
            # Gather the language actions for summation
            language_actions_to_sum = tf.gather(traj["language_actions"], summation_indices)
            # Keep unsummed window for debugging: shape [traj_len, summation_steps]
            traj["language_actions_unsummed"] = language_actions_to_sum
            
            # Sum over the language actions
            summed_language_actions = tf.map_fn(
                _sum_language_actions,
                language_actions_to_sum,
                fn_output_signature=tf.string
            )
            # Keep a single summed language string per timestep (no chunking)
            traj["language_actions"] = summed_language_actions
            return traj

        # TODO: chunk action or not
        dataset = dataset.traj_map(group_language_actions, num_parallel_calls)

        def filter_idle(traj):
            """Filter out chunks with idle actions.
            --> we filter if at least first half of chunk does not move.
            """
            if action_space == DroidActionSpace.CARTESIAN_POSITION:
                # Compute delta to first position in action chunk
                return tf.reduce_any(tf.abs(traj["actions"][: action_chunk_size // 2] - traj["actions"][:1]) > 1e-3)
            return tf.reduce_any(tf.abs(traj["actions"][: action_chunk_size // 2]) > 1e-3)

        if DEBUG_TIMING:
            def _timed_filter(x):
                t0 = tf.timestamp()
                out = filter_idle(x)
                t1 = tf.timestamp()
                ms = (t1 - t0) * 1000.0
                def _log(ms_np):
                    try:
                        logging.info(f"[tf.data] filter_idle ms={float(ms_np):.1f}")
                    except Exception:
                        pass
                    return np.int64(0)
                _ = tf.py_function(_log, [ms], Tout=tf.int64)
                return out
            dataset = dataset.filter(_timed_filter)
        else:
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

        if DEBUG_TIMING:
            dataset = dataset.frame_map(_wrap_timed_map(decode_images, "decode_images"), num_parallel_calls)
        else:
            dataset = dataset.frame_map(decode_images, num_parallel_calls)

        # Shuffle, batch
        dataset = dataset.batch(batch_size)
        # Overlap input pipeline with consumers; lets TF fill a small buffer per host.
        dataset = dataset.prefetch(2)
        # Note =>> Seems to reduce memory usage without affecting speed?

        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.action_chunk_size = action_chunk_size
        self.summation_steps = summation_steps
        

    def __iter__(self):
        it = self.dataset.as_numpy_iterator()
        while True:
            t0 = time.perf_counter() if DEBUG_TIMING else 0.0
            try:
                batch = next(it)
            except StopIteration:
                logging.info("StopIteration")
                return
            if DEBUG_TIMING:
                dt = (time.perf_counter() - t0) * 1000.0
                logging.info("DroidCoTRldsDataset as_numpy_iterator.next: %.1f ms", dt)
            yield batch

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
