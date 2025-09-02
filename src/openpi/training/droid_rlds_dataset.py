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

import openpi.training.config as _config

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
        if shuffle:
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
        config: _config.DataConfig,
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
        # Validation support
        split_seed: int = 0,
        # Overfitting support: cap number of flattened samples (after shuffle)
        max_samples: int | None = None,
        # Global seed for all dataset-related randomness
        seed: int = 0,
        split: str = "train",
    ):
        # Import tensorflow here to not make it mandatory in case RLDS data loader is not used.
        import dlimp as dl
        import tensorflow as tf
        import tensorflow_datasets as tfds

        assert action_space == DroidActionSpace.CARTESIAN_POSITION, "CoT only supports EEF actions for now"
        validation_mode = getattr(config, "validation_mode", "easy")
        summation_steps = getattr(config, "summation_steps", 15)
        val_fraction = getattr(config, "val_fraction", 0.02)
        vis_dataset = getattr(config, "vis_dataset", False)
        use_wrist_image = getattr(config, "use_wrist_image", False)

        logging.info(
            f"validation_mode: {validation_mode}, val_fraction: {val_fraction}, vis_dataset: {vis_dataset}, \
                use_wrist_image: {use_wrist_image}, summation_steps: {summation_steps}, max_samples: {max_samples}, \
                    sum_decimal: {config.sum_decimal}, left_pad: {config.left_pad}, include_decimal_point: {config.include_decimal_point}, \
                        batch_size: {batch_size}"
        )

        # ------------------------------------------------------------------
        # Global seeding for reproducibility across dataset ops
        # ------------------------------------------------------------------
        tf.random.set_seed(seed)
        # try:
        #     # TF 2.12+: enable deterministic kernels where available
        #     tf.config.experimental.enable_op_determinism()
        # except Exception:
        #     pass

        # Configure Tensorflow with no GPU/TPU devices to avoid clobbering JAX/TPU runtime
        tf.config.set_visible_devices([], "GPU")
        try:
            tf.config.set_visible_devices([], "TPU")
        except Exception:
            pass

        METADATA_PATH = language_action_dir.replace("droid-lang-actions", "metadata")

        # ------------------------------------------------------------------
        # Validation difficulty levels
        #   - "easy": train/val do NOT share trajectories (split by episode_id); labs can overlap
        #   - "hard": train/val come from different labs (split by lab prefix in episode_id)
        # Aliases for backward compatibility: {"easier", "medium"} -> "easy"; {"harder"} -> "hard"
        # ------------------------------------------------------------------
        validation_mode = (validation_mode or "easy").lower()
        assert validation_mode in {"easy", "hard"}, (
            f"validation_mode must be one of 'easy', 'hard'; got: {validation_mode}"
        )

        # ---------------------------------------------------------------------
        # 1. TF-DS builder + base dataset
        # ---------------------------------------------------------------------
        # Resolve autotune sentinels now that TF is imported
        if num_parallel_reads == -1:
            num_parallel_reads = tf.data.AUTOTUNE
        if num_parallel_calls == -1:
            num_parallel_calls = tf.data.AUTOTUNE

        builder = tfds.builder("droid", data_dir=data_dir)
        dataset = dl.DLataset.from_rlds(
            builder,
            split="train",
            # shuffle=shuffle,
            num_parallel_reads=num_parallel_reads,
        )

        dataset = dataset.shard(jax.process_count(), jax.process_index())

        # # Enforce deterministic mapping/order for reproducibility
        # opts = tf.data.Options()
        # opts.experimental_deterministic = True
        # dataset = dataset.with_options(opts)

        # ---------------------------------------------------------------------
        # 2. Language-action table (episode_id → serialized tensor)
        # ---------------------------------------------------------------------
        FEATURES = {
            "episode_id": tf.io.FixedLenFeature([], tf.string),
            "lang_ser": tf.io.FixedLenFeature([], tf.string),
        }

        def _parse(record):
            ex = tf.io.parse_single_example(record, FEATURES)
            lang = tf.io.parse_tensor(ex["lang_ser"], out_type=tf.string)  # shape: [T+1]
            return ex["episode_id"], lang

        files = tf.io.gfile.glob(f"{language_action_dir}/tfds_language_actions-*.tfrecord.gz")
        ds = (
            tf.data.TFRecordDataset(files, compression_type="GZIP", num_parallel_reads=tf.data.AUTOTUNE)
            .map(_parse, num_parallel_calls=tf.data.AUTOTUNE)
            .prefetch(tf.data.AUTOTUNE)
        )
        episodes, lang_serialized = [], []
        for ep_id, lang in ds:
            episodes.append(ep_id.numpy().decode())
            lang_serialized.append(tf.io.serialize_tensor(lang).numpy())

        keys = tf.constant(episodes, dtype=tf.string)
        values = tf.constant(lang_serialized, dtype=tf.string)
        default_lang_value = tf.constant(b"", dtype=tf.string)
        lang_table = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(keys, values),
            default_value=default_lang_value,
        )

        print_memory_usage("After building lang_table")

        # ---------------------------------------------------------------------
        # 3. Episode-ID table  (valid_eids → True)
        # ---------------------------------------------------------------------
        keys = tf.constant(episodes, dtype=tf.string)
        values = tf.ones(len(episodes), dtype=tf.bool)
        eid_table = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(keys, values),
            default_value=tf.constant(False, dtype=tf.bool),
        )

        # ---------------------------------------------------------------------
        # 4. Episode-path ↔ Episode-ID table
        # ---------------------------------------------------------------------
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
        print_memory_usage("After building ep_table")

        # ---------------------------------------------------------------------
        # 5. Camera-index table  (episode_id → ext-cam idx)
        # ---------------------------------------------------------------------
        with tf.io.gfile.GFile(f"{METADATA_PATH}/cam2base_extrinsics.json", "r") as fp:
            cam2base_extrinsics = json.load(fp)
        with tf.io.gfile.GFile(f"{METADATA_PATH}/camera_serials.json", "r") as fp:
            camera_serials = json.load(fp)
        if vis_dataset:
            with tf.io.gfile.GFile(f"{METADATA_PATH}/intrinsics.json", "r") as fp:
                intrinsics_json = json.load(fp)
            eid_to_intr_vec = {}
            eid_to_extr_mat = {}

            def _euler_xyz_to_rot(rx, ry, rz):
                # Build rotation matrix from XYZ intrinsic rotations
                cx, sx = np.cos(rx), np.sin(rx)
                cy, sy = np.cos(ry), np.sin(ry)
                cz, sz = np.cos(rz), np.sin(rz)
                Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float32)
                Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float32)
                Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float32)
                return Rz @ Ry @ Rx

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

            if vis_dataset:
                # Camera intrinsics as [fx, fy, cx, cy]
                try:
                    fx, cx, fy, cy = intrinsics_json[eid][camera_serial]["cameraMatrix"]
                    eid_to_intr_vec[eid] = [fx, fy, cx, cy]
                except Exception:
                    # Fallback to zeros
                    eid_to_intr_vec[eid] = [0.0, 0.0, 0.0, 0.0]

                # Camera extrinsics 4x4 from [tx,ty,tz,rx,ry,rz]
                try:
                    tx, ty, tz, rx, ry, rz = extr[camera_serial]
                    R = _euler_xyz_to_rot(rx, ry, rz)
                    T = np.eye(4, dtype=np.float32)
                    T[:3, :3] = R
                    T[:3, 3] = [tx, ty, tz]
                    eid_to_extr_mat[eid] = T.reshape(-1)
                except Exception:
                    eid_to_extr_mat[eid] = np.zeros((16,), dtype=np.float32)

        keys = tf.constant(list(eid_to_cam_dict.keys()), dtype=tf.string)
        values = tf.constant(list(eid_to_cam_dict.values()), dtype=tf.int32)
        cam_table = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(keys, values),
            default_value=-1,  # -1 ⇒ fallback camera
        )
        print_memory_usage("After building cam_table")

        # Camera intrinsics/extrinsics lookup tables (serialize tensors to tf.string to avoid shape issues)
        if vis_dataset:
            calib_eids = list(eid_to_cam_dict.keys())
            intr_ser = []
            extr_ser = []
            for _eid in calib_eids:
                intr_ser.append(tf.io.serialize_tensor(tf.constant(eid_to_intr_vec[_eid], dtype=tf.float32)).numpy())
                extr_ser.append(tf.io.serialize_tensor(tf.constant(eid_to_extr_mat[_eid], dtype=tf.float32)).numpy())
            calib_keys = tf.constant(calib_eids, dtype=tf.string)
            intr_vals = tf.constant(intr_ser, dtype=tf.string)
            extr_vals = tf.constant(extr_ser, dtype=tf.string)
            intr_table = tf.lookup.StaticHashTable(
                tf.lookup.KeyValueTensorInitializer(calib_keys, intr_vals),
                default_value=tf.constant(b"", dtype=tf.string),
            )
            extr_table = tf.lookup.StaticHashTable(
                tf.lookup.KeyValueTensorInitializer(calib_keys, extr_vals),
                default_value=tf.constant(b"", dtype=tf.string),
            )

        # ---------------------------------------------------------------------
        # 6. Language-instruction tables (3 per episode_id)
        # ---------------------------------------------------------------------
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

        print_memory_usage("After building instr_table")

        # ------------------------------------------------------------------
        # Regex helpers for robust path/id extraction
        # ------------------------------------------------------------------
        def _extract_episode_path_from_file_path(file_path):
            """Extract episode path from a full file path using regex.

            Removes everything up to and including 'r2d2-data/' or
            'r2d2-data-full/', then trims anything from '/trajectory' onwards.
            """
            # Strip dataset prefix up to r2d2-data or r2d2-data-full
            rel = tf.strings.regex_replace(
                file_path,
                r"^.*r2d2-data(?:-full)?/",
                "",
            )
            # Remove trailing '/trajectory...' suffix
            episode_path = tf.strings.regex_replace(
                rel,
                r"/trajectory.*$",
                "",
            )
            return episode_path

        def _episode_id_from_traj(traj):
            """Lookup episode_id from trajectory metadata using regex extraction."""
            file_path = traj["traj_metadata"]["episode_metadata"]["file_path"][0]
            episode_path = _extract_episode_path_from_file_path(file_path)
            return ep_table.lookup(episode_path)

        def _lab_from_episode_id(episode_id):
            """Extract lab/environment name from an episode_id.

            Example episode_id: "AUTOLab+5d05c5aa+2023-07-07-10h-18m-41s" -> "AUTOLab".
            Uses regex to avoid RaggedTensor outputs from tf.strings.split.
            """
            return tf.strings.regex_replace(episode_id, r"\+.*$", "")

        def _id_ok(traj):
            episode_id = _episode_id_from_traj(traj)
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

        want_val = split == "val"

        def _split_filter(traj):
            episode_id = _episode_id_from_traj(traj)  # scalar tf.string

            # # --- Assertions: fail fast if any bad traj leaks through ---
            # # 1) episode_id must be non-empty (not the default "" from failed lookup)
            # a1 = tf.debugging.assert_greater(
            #     tf.strings.length(episode_id),
            #     0,
            #     message="[_split_filter] Empty/missing episode_id; expected pre-filtering.",
            # )
            # # 2) episode_id must exist in eid_table (has language actions)
            # a2 = tf.debugging.assert_equal(
            #     eid_table.lookup(episode_id),
            #     True,
            #     message="[_split_filter] episode_id not found in eid_table (no lang actions / bad metadata).",
            # )

            # # Make sure assertions run before we use episode_id.
            # with tf.control_dependencies([a1, a2]):
            #     episode_id = tf.identity(episode_id)

            # --- Deterministic hash split ---
            salt = tf.strings.as_string(split_seed)
            if validation_mode == "hard":
                # Environment-level split: hold out entire labs
                lab_name = _lab_from_episode_id(episode_id)
                key = tf.strings.join([salt, lab_name])
            else:  # "easy": per-trajectory split within seen labs
                key = tf.strings.join([salt, episode_id])
            bucket = tf.strings.to_hash_bucket_fast(key, 1000)
            thr = tf.cast(int(val_fraction * 1000), tf.int64)
            is_val = bucket < thr

            return is_val if want_val else tf.logical_not(is_val)

        dataset = dataset.filter(_split_filter)

        # Repeat dataset so we never run out of data.
        dataset = dataset.repeat()

        with tf.io.gfile.GFile(f"{METADATA_PATH}/keep_ranges_1_0_1.json", "r") as f:
            filter_dict = json.load(f)

        logging.info(f"Using filter dictionary with {len(filter_dict)} episodes")

        keys_tensor = []
        values_tensor = []

        for episode_key, ranges in filter_dict.items():
            for start, end in ranges:
                for t in range(start, end):
                    frame_key = f"{episode_key}--{t}"
                    keys_tensor.append(frame_key)
                    values_tensor.append(True)
        filter_table = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(keys_tensor, values_tensor), default_value=False
        )
        logging.info("Filter hash table initialized")

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
            # Align lengths across modalities
            traj_len = tf.shape(actions)[0]
            episode_id = _episode_id_from_traj(traj)
            lang_bytes = lang_table.lookup(episode_id)
            lang_tensor = tf.io.parse_tensor(lang_bytes, tf.string)
            # Language actions may include an extra terminal step; crop to match action length
            lang_tensor = lang_tensor[:traj_len]
            instruction_1 = instr_table_1.lookup(episode_id)
            instruction_2 = instr_table_2.lookup(episode_id)
            instruction_3 = instr_table_3.lookup(episode_id)
            # Check which instruction is non-empty, and form a non-empty list of instructions.
            # If all instrcutions are empty, raise an error. Then sample one instruction from the list.
            instructions = tf.stack([instruction_1, instruction_2, instruction_3])
            mask = tf.strings.length(instructions) > 0
            non_empty = tf.boolean_mask(instructions, mask)
            num_valid = tf.shape(non_empty)[0]

            fallback_index = tf.random.uniform(
                (), minval=0, maxval=tf.shape(fallback_instructions)[0], dtype=tf.int32, seed=seed
            )
            fallback_instruction = fallback_instructions[fallback_index]
            instruction = tf.cond(
                num_valid > 0,
                lambda: tf.random.shuffle(non_empty, seed=seed)[0],
                lambda: fallback_instruction,
            )

            instruction_vec = tf.fill([tf.shape(actions)[0]], instruction)

            cam_idx = cam_table.lookup(episode_id)
            cam_images = [
                traj["observation"]["exterior_image_1_left"],
                traj["observation"]["exterior_image_2_left"],
            ]
            cam_images = tf.stack(cam_images, axis=0)  # shape (2, H, W, C)
            cam_idx_clamped = tf.clip_by_value(cam_idx, 0, tf.shape(cam_images)[0] - 1)
            exterior_img = tf.gather(cam_images, cam_idx_clamped)

            # TODO: use wrist camera image or not
            # # Randomly samples one of the two exterior images in DROID during training (we only train with one at a time).
            # # Note: the "left" refers to the left camera in the stereo pair, we only train on the left camera.
            # exterior_img = tf.cond(
            #     tf.random.uniform(shape=[], seed=seed) > 0.5,
            #     lambda: traj["observation"]["exterior_image_1_left"],
            #     lambda: traj["observation"]["exterior_image_2_left"],
            # )
            episode_id_vec = tf.fill([traj_len], episode_id)

            # Deserialize intrinsics/extrinsics and broadcast across trajectory length
            if vis_dataset:
                intr = tf.io.parse_tensor(intr_table.lookup(episode_id), out_type=tf.float32)  # [4]
                extr = tf.reshape(
                    tf.io.parse_tensor(extr_table.lookup(episode_id), out_type=tf.float32), [4, 4]
                )  # [4,4]
                intr_b = tf.broadcast_to(intr[None, :], [traj_len, 4])
                extr_b = tf.broadcast_to(extr[None, :, :], [traj_len, 4, 4])

            step_id = (
                traj["traj_metadata"]["episode_metadata"]["recording_folderpath"]
                + "--"
                + traj["traj_metadata"]["episode_metadata"]["file_path"]
                + "--"
                + tf.as_string(tf.range(traj_len))
            )
            passes_filter = filter_table.lookup(step_id)

            _return_dict = {
                "actions": actions,
                "observation": {
                    "image": exterior_img,
                    "cartesian_position": traj["observation"]["cartesian_position"],
                    "gripper_position": traj["observation"]["gripper_position"],
                },
                "prompt": instruction_vec,
                "language_actions": lang_tensor,
                "episode_id": episode_id_vec,
                "passes_filter": passes_filter,
            }
            if vis_dataset:
                _return_dict["camera_intrinsics"] = intr_b  # [traj_len, 4]
                _return_dict["camera_extrinsics"] = extr_b  # [traj_len, 4, 4]
            if use_wrist_image:
                _return_dict["observation"]["wrist_image"] = traj["observation"]["wrist_image_left"]
            return _return_dict

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
            traj["language_actions"] = language_actions_to_sum

            if vis_dataset:
                grouped_images = tf.gather(traj["observation"]["image"], summation_indices)
                traj["observation"]["image"] = grouped_images

                if use_wrist_image:
                    grouped_wrist_images = tf.gather(traj["observation"]["wrist_image"], summation_indices)
                    traj["observation"]["wrist_image"] = grouped_wrist_images

                # Group cartesian positions for start/end projection
                grouped_cart = tf.gather(traj["observation"]["cartesian_position"], summation_indices)
                traj["observation"]["cartesian_position_window"] = grouped_cart

                # Also group broadcast calibration to align with the same windowed time dimension
                # camera_intrinsics: [traj_len, 4] -> [traj_len, summation_steps, 4]
                assert "camera_intrinsics" in traj, "camera_intrinsics not found in traj"
                traj["camera_intrinsics"] = tf.gather(traj["camera_intrinsics"], summation_indices)
                assert "camera_extrinsics" in traj, "camera_extrinsics not found in traj"
                # camera_extrinsics: [traj_len, 4,4] -> [traj_len, summation_steps, 4,4]
                idx = summation_indices
                T = traj["camera_extrinsics"]
                traj["camera_extrinsics"] = tf.gather(T, idx)

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

        def filter_from_dict(frame):
            return frame["passes_filter"]

        dataset = dataset.filter(filter_from_dict)

        # Remove "passes_filter" key from output
        def remove_passes_filter(frame):
            frame.pop("passes_filter")
            return frame

        dataset = dataset.map(remove_passes_filter)

        # Decode images: RLDS saves encoded images, only decode now for efficiency
        def decode_images(traj):
            def _decode_single(img_bytes):
                return tf.io.decode_image(img_bytes, expand_animations=False, dtype=tf.uint8)

            if vis_dataset:
                traj["observation"]["image"] = tf.map_fn(
                    _decode_single,
                    traj["observation"]["image"],
                    fn_output_signature=tf.uint8,
                )
                if use_wrist_image:
                    traj["observation"]["wrist_image"] = tf.map_fn(
                        _decode_single,
                        traj["observation"]["wrist_image"],
                        fn_output_signature=tf.uint8,
                    )
            else:
                traj["observation"]["image"] = _decode_single(traj["observation"]["image"])
                if use_wrist_image:
                    traj["observation"]["wrist_image"] = _decode_single(traj["observation"]["wrist_image"])
            return traj

        # Only shuffle during training; validation should be deterministic and cheaper
        if shuffle and max_samples is None:
            dataset = dataset.shuffle(shuffle_buffer_size, seed=seed)

        # If requested, cap the number of flattened samples for overfitting tests.
        # We cache the capped set so repeating yields the same fixed subset.
        if max_samples is not None:
            dataset = dataset.take(int(max_samples)).cache().repeat()

        if DEBUG_TIMING:
            dataset = dataset.frame_map(_wrap_timed_map(decode_images, "decode_images"), num_parallel_calls)
        else:
            dataset = dataset.frame_map(decode_images, num_parallel_calls)

        # Shuffle, batch
        dataset = dataset.batch(batch_size)
        # Overlap input pipeline with consumers; lets TF fill a small buffer per host.
        dataset = dataset.prefetch(2)
        # Note =>> Seems to reduce memory usage without affecting speed?
        dataset = dataset.with_ram_budget(1)

        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.action_chunk_size = action_chunk_size
        self.summation_steps = summation_steps
        self.max_samples = max_samples
        self.validation_mode = validation_mode

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
