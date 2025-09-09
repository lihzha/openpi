from collections.abc import Iterator, Sequence
import dataclasses
import multiprocessing
import os
import typing
from typing import Protocol, SupportsIndex, TypeVar

import jax
import jax.numpy as jnp

os.environ.pop("LEROBOT_HOME", None)  # Trick to ensure LEROBOT_HOME is not set.

try:
    import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
except:
    pass
import numpy as np

try:
    import torch
except:
    pass


import openpi.models.model as _model
import openpi.policies.droid_cot_policy as droid_cot_policy
import openpi.training.config as _config
from openpi.training.droid_rlds_dataset import DroidCoTRldsDataset
from openpi.training.droid_rlds_dataset import DroidRldsDataset
import openpi.transforms as _transforms

T_co = TypeVar("T_co", covariant=True)

# Enable lightweight timing logs when set to "1"
DEBUG_TIMING = os.environ.get("OPENPI_TIMING", "0") == "1"


class Dataset(Protocol[T_co]):
    """Interface for a dataset with random access."""

    def __getitem__(self, index: SupportsIndex) -> T_co:
        raise NotImplementedError("Subclasses of Dataset should implement __getitem__.")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")


class IterableDataset(Protocol[T_co]):
    """Interface for an iterable dataset."""

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError("Subclasses of IterableDataset should implement __iter__.")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")


class DataLoader(Protocol[T_co]):
    """Interface for a data loader."""

    def data_config(self) -> _config.DataConfig:
        """Get the data config for this data loader."""
        raise NotImplementedError("Subclasses of DataLoader should implement data_config.")

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError("Subclasses of DataLoader should implement __iter__.")


class TransformedDataset(Dataset[T_co]):
    def __init__(self, dataset: Dataset, transforms: Sequence[_transforms.DataTransformFn]):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)

    def __getitem__(self, index: SupportsIndex) -> T_co:
        return self._transform(self._dataset[index])

    def __len__(self) -> int:
        return len(self._dataset)


class IterableTransformedDataset(IterableDataset[T_co]):
    def __init__(
        self,
        dataset: IterableDataset,
        transforms: Sequence[_transforms.DataTransformFn],
        *,
        is_batched: bool = False,
    ):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)
        self._is_batched = is_batched

    def __iter__(self):
        upstream_iter = iter(self._dataset)
        while True:
            # Fetch from upstream iterable (e.g., RLDS/TF pipeline)
            try:
                sample = next(upstream_iter)
            except StopIteration:
                return

            if self._is_batched:
                # Transforms are designed to be applied to individual samples. So we need to split the batch into
                # individual samples and apply the transform to each sample individually.
                batch_size = next(v.shape[0] for v in sample.values())

                # Split -> Transform -> Stack
                individual_samples = [jax.tree.map(lambda x: x[i], sample) for i in range(batch_size)]  # noqa: B023

                transformed = [self._transform(s) for s in individual_samples]

                out = jax.tree.map(lambda *x: np.stack(x, axis=0), *transformed)

                yield out
            else:
                out = self._transform(sample)
                yield out

    def __len__(self) -> int:
        return len(self._dataset)


class FakeDataset(Dataset):
    def __init__(self, model_config: _model.BaseModelConfig, num_samples: int):
        self._num_samples = num_samples
        self._observation_spec, self._action_spec = model_config.inputs_spec()

    def __getitem__(self, index: SupportsIndex) -> dict:
        rng = jax.random.key(index.__index__())

        def make_from_spec(spec: jax.ShapeDtypeStruct):
            nonlocal rng
            rng, data_rng = jax.random.split(rng)
            # Remove the batch dimension.
            shape = spec.shape[1:]
            if spec.dtype == jnp.float32:
                return jax.random.uniform(data_rng, shape=shape, minval=-1.0, maxval=1.0)
            if spec.dtype == jnp.int32:
                return jax.random.randint(data_rng, shape=shape, minval=0, maxval=2048)
            return jnp.zeros(shape=shape, dtype=spec.dtype)

        observation = jax.tree.map(make_from_spec, self._observation_spec)
        action = jax.tree.map(make_from_spec, self._action_spec)

        return {
            **observation.to_dict(),
            "actions": action,
        }

    def __len__(self) -> int:
        return self._num_samples


class TestIterableDataset(IterableDataset[dict]):
    """Lightweight iterable dataset that yields synthetic, already-batched samples.

    This is intended for debugging multi-process data loading and sharding without
    pulling in RLDS/DROID dependencies. It produces batches matching the model's
    input/output specs.
    """

    def __init__(
        self,
        model_config: _model.BaseModelConfig,
        local_batch_size: int,
        *,
        num_batches: int | None = None,
        seed: int = 0,
    ) -> None:
        self._model_config = model_config
        self._local_batch_size = local_batch_size
        self._num_batches = num_batches
        self._rng = np.random.RandomState(seed)

        # Create shape/dtype specs with the desired per-host batch size.
        self._observation_spec, self._action_spec = model_config.inputs_spec(batch_size=local_batch_size)

    def __iter__(self):
        produced = 0

        def make_from_spec(spec: jax.ShapeDtypeStruct):
            shape = spec.shape
            if spec.dtype == jnp.float32:
                return self._rng.uniform(-1.0, 1.0, size=shape).astype(np.float32)
            if spec.dtype == jnp.int32:
                return self._rng.randint(0, 2048, size=shape, dtype=np.int32)
            return np.zeros(shape=shape, dtype=spec.dtype)

        while True:
            if self._num_batches is not None and produced >= self._num_batches:
                return
            observation = jax.tree.map(make_from_spec, self._observation_spec)
            action = jax.tree.map(make_from_spec, self._action_spec)
            produced += 1
            # Convert Observation to dict expected by DataLoaderImpl.
            yield {**observation.to_dict(), "actions": action}

    def __len__(self) -> int:
        return self._num_batches if self._num_batches is not None else 2**31 - 1


def create_torch_dataset(
    data_config: _config.DataConfig, action_horizon: int, model_config: _model.BaseModelConfig
) -> Dataset:
    """Create a dataset for training."""
    repo_id = data_config.repo_id
    if repo_id is None:
        raise ValueError("Repo ID is not set. Cannot create dataset.")
    if repo_id == "fake":
        return FakeDataset(model_config, num_samples=1024)

    dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id)
    dataset = lerobot_dataset.LeRobotDataset(
        data_config.repo_id,
        delta_timestamps={
            key: [t / dataset_meta.fps for t in range(action_horizon)] for key in data_config.action_sequence_keys
        },
    )

    if data_config.prompt_from_task:
        dataset = TransformedDataset(dataset, [_transforms.PromptFromLeRobotTask(dataset_meta.tasks)])

    return dataset


def create_rlds_dataset(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    *,
    shuffle: bool = False,
    split_seed: int = 0,
    seed: int = 0,
    max_samples: int | None = None,
    split: str = "train",
) -> Dataset:
    # At the moment, we only support DROID for RLDS datasets.
    # Use per-host batching to avoid duplicative slicing work in the loader
    # and reduce memory pressure when running multi-process (e.g., multi-host TPU).
    local_batch_size = max(1, batch_size // jax.process_count())
    if data_config.cot or data_config.use_memory:
        return DroidCoTRldsDataset(
            data_dir=data_config.rlds_data_dir,
            language_action_dir=data_config.language_action_dir,
            batch_size=local_batch_size,
            shuffle=shuffle,
            action_chunk_size=action_horizon,
            action_space=data_config.action_space,
            shuffle_buffer_size=data_config.shuffle_buffer_size,
            split_seed=split_seed,
            max_samples=max_samples,
            seed=seed,
            config=data_config,
            split=split,
        )
    return DroidRldsDataset(
        data_dir=data_config.rlds_data_dir,
        batch_size=local_batch_size,
        shuffle=shuffle,
        action_chunk_size=action_horizon,
        action_space=data_config.action_space,
        shuffle_buffer_size=data_config.shuffle_buffer_size,
    )


def transform_dataset(dataset: Dataset, data_config: _config.DataConfig, *, skip_norm_stats: bool = False) -> Dataset:
    """Transform the dataset by applying the data transforms."""
    norm_stats = {}
    if data_config.repo_id != "fake" and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError(
                "Normalization stats not found. "
                "Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`."
            )
        norm_stats = data_config.norm_stats

    return TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
    )


def transform_iterable_dataset(
    dataset: IterableDataset,
    data_config: _config.DataConfig,
    *,
    skip_norm_stats: bool = False,
    is_batched: bool = False,
    split: str | None = None,
) -> IterableDataset:
    """Transform the dataset by applying the data transforms."""
    norm_stats = {}
    if data_config.repo_id != "fake" and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError(
                "Normalization stats not found. "
                "Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`."
            )
        norm_stats = data_config.norm_stats

    # If caller passes split and config is RLDSDroidCoTDataConfig, force-disable dropout for non-train splits
    transforms_inputs = [
        *data_config.repack_transforms.inputs,
        *data_config.data_transforms.inputs,
        _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
        *data_config.model_transforms.inputs,
    ]

    if split is not None and hasattr(data_config, "wrist_image_dropout_prob"):
        if split != "train":
            # Create shallow copies of the final model transform if it's DroidCoTInputs to zero probs
            new_inputs = []
            for t in transforms_inputs:
                try:
                    if isinstance(t, droid_cot_policy.DroidCoTInputs):
                        new_inputs.append(
                            dataclasses.replace(
                                t,
                                wrist_image_dropout_prob=0.0,
                                text_state_dropout_prob=0.0,
                            )
                        )
                    else:
                        new_inputs.append(t)
                except Exception:
                    new_inputs.append(t)
            transforms_inputs = new_inputs

    return IterableTransformedDataset(
        dataset,
        transforms_inputs,
        is_batched=is_batched,
    )


def create_data_loader(
    config: _config.TrainConfig,
    *,
    sharding: jax.sharding.Sharding | None = None,
    shuffle: bool = False,
    num_batches: int | None = None,
    skip_norm_stats: bool = False,
    seed: int = 0,
    max_samples: int | None = None,
    split: str = "train",
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create a data loader for training."""
    data_config = config.data.create(config.assets_dirs, config.model)

    if data_config.rlds_data_dir is not None:
        return create_rlds_data_loader(
            data_config,
            action_horizon=config.model.action_horizon,
            batch_size=config.batch_size,
            sharding=sharding,
            shuffle=shuffle,
            num_batches=num_batches,
            skip_norm_stats=skip_norm_stats,
            split_seed=seed,
            seed=seed,
            max_samples=max_samples,
            split=split,
        )
    return create_torch_data_loader(
        data_config,
        model_config=config.model,
        action_horizon=config.model.action_horizon,
        batch_size=config.batch_size,
        sharding=sharding,
        shuffle=shuffle,
        num_batches=num_batches,
        num_workers=config.num_workers,
        seed=seed,
        skip_norm_stats=skip_norm_stats,
    )


def create_torch_data_loader(
    data_config: _config.DataConfig,
    model_config: _model.BaseModelConfig,
    action_horizon: int,
    batch_size: int,
    *,
    sharding: jax.sharding.Sharding | None = None,
    skip_norm_stats: bool = False,
    shuffle: bool = False,
    num_batches: int | None = None,
    num_workers: int = 0,
    seed: int = 0,
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create a data loader for training.

    Args:
        data_config: The data configuration.
        action_horizon: The action horizon.
        batch_size: The batch size.
        sharding: The sharding to use for the data loader. If None, the data loader will
            use a single device sharding.
        skip_norm_stats: Whether to skip data normalization.
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return. If the number exceeds the
            number of batches in the dataset, the data loader will loop over the dataset.
            If not provided, will iterate over the dataset indefinitely.
        num_workers: The number of worker processes to use. If zero, the data loader will
            execute in the main process.
        seed: The seed to use for shuffling the data.
    """
    dataset = create_torch_dataset(data_config, action_horizon, model_config)
    dataset = transform_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats)

    data_loader = TorchDataLoader(
        dataset,
        local_batch_size=batch_size // jax.process_count(),
        sharding=sharding,
        shuffle=shuffle,
        num_batches=num_batches,
        num_workers=num_workers,
        seed=seed,
    )

    return DataLoaderImpl(data_config, data_loader)


def create_rlds_data_loader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    *,
    sharding: jax.sharding.Sharding | None = None,
    skip_norm_stats: bool = False,
    shuffle: bool = False,
    num_batches: int | None = None,
    split_seed: int = 0,
    seed: int = 0,
    max_samples: int | None = None,
    split: str = "train",
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create an RLDS data loader for training.

    Note: This data loader requires some extra dependencies -- see examples/droid/README_train.md

    Args:
        data_config: The data configuration.
        action_horizon: The action horizon.
        batch_size: The batch size.
        sharding: The sharding to use for the data loader. If None, the data loader will
            use a single device sharding.
        skip_norm_stats: Whether to skip data normalization.
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return. If the number exceeds the
            number of batches in the dataset, the data loader will loop over the dataset.
            If not provided, will iterate over the dataset indefinitely.
    """
    dataset = create_rlds_dataset(
        data_config,
        action_horizon,
        batch_size,
        shuffle=shuffle,
        split_seed=split_seed,
        seed=seed,
        max_samples=(max_samples if max_samples is not None else getattr(data_config, "max_samples", None)),
        split=split,
    )
    dataset = transform_iterable_dataset(
        dataset,
        data_config,
        skip_norm_stats=skip_norm_stats,
        is_batched=True,
        split=split,
    )

    data_loader = RLDSDataLoader(
        dataset,
        sharding=sharding,
        num_batches=num_batches,
        # Dataset is already host-sharded and per-host batched; skip extra host slicing.
        auto_shard=False,
    )

    return DataLoaderImpl(data_config, data_loader)


def create_test_rlds_data_loader(
    data_config: _config.DataConfig,
    model_config: _model.BaseModelConfig,
    *,
    batch_size: int,
    sharding: jax.sharding.Sharding | None = None,
    num_batches: int | None = None,
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create a simple iterable data loader for multi-process sharding tests.

    Produces synthetic per-host batches that already match the model's
    Observation/Actions specs. This bypasses RLDS/DROID complexity while
    exercising the same sharding/multi-process pathway as RLDSDataLoader.
    """
    local_batch_size = max(1, batch_size // jax.process_count())
    dataset = TestIterableDataset(model_config, local_batch_size, num_batches=num_batches)

    data_loader = RLDSDataLoader(
        dataset,  # already per-host batched
        sharding=sharding,
        num_batches=num_batches,
        auto_shard=False,
    )

    # Provide a minimal DataConfig to satisfy interfaces; callers may pass their
    # own if desired.
    return DataLoaderImpl(data_config, data_loader)


class TorchDataLoader:
    def __init__(
        self,
        dataset,
        local_batch_size: int,
        *,
        sharding: jax.sharding.Sharding | None = None,
        shuffle: bool = False,
        num_batches: int | None = None,
        num_workers: int = 0,
        seed: int = 0,
    ):
        """Create a PyTorch data loader.

        Args:
            dataset: The dataset to load.
            local_batch_size: The local batch size for each process.
            sharding: The sharding to use for the data loader.
            shuffle: Whether to shuffle the data.
            num_batches: If provided, determines the number of returned batches. If the
                number is larger than the number of batches in the dataset, the data loader
                will loop over the dataset. If not provided, will iterate over the dataset
                indefinitely.
            num_workers: The number of worker processes to use. If zero, the data loader will
                execute in the main process.
            seed: The seed to use for shuffling the data.
        """
        if jax.process_count() > 1:
            raise NotImplementedError("Data loading with multiple processes is not supported.")

        if len(dataset) < local_batch_size:
            raise ValueError(f"Local batch size ({local_batch_size}) is larger than the dataset size ({len(dataset)}).")

        if sharding is None:
            # Use data parallel sharding by default.
            sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ("B",)),
                jax.sharding.PartitionSpec("B"),
            )

        self._sharding = sharding
        self._num_batches = num_batches

        mp_context = None
        if num_workers > 0:
            mp_context = multiprocessing.get_context("spawn")

        generator = torch.Generator()
        generator.manual_seed(seed)
        self._data_loader = torch.utils.data.DataLoader(
            typing.cast(torch.utils.data.Dataset, dataset),
            batch_size=local_batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            multiprocessing_context=mp_context,
            persistent_workers=num_workers > 0,
            collate_fn=_collate_fn,
            worker_init_fn=_worker_init_fn,
            drop_last=True,
            generator=generator,
        )

    @property
    def torch_loader(self):
        return self._data_loader

    def __iter__(self):
        num_items = 0
        while True:
            data_iter = iter(self._data_loader)
            while True:
                if self._num_batches is not None and num_items >= self._num_batches:
                    return
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break  # We've exhausted the dataset. Create a new iterator and start over.
                num_items += 1
                yield jax.tree.map(lambda x: jax.make_array_from_process_local_data(self._sharding, x), batch)


def _collate_fn(items):
    """Collate the batch elements into batched numpy arrays."""
    # Make sure to convert to numpy arrays before stacking since some of the incoming elements
    # may be JAX arrays.
    return jax.tree.map(lambda *x: np.stack(np.asarray(x), axis=0), *items)


def _worker_init_fn(worker_id: int) -> None:
    """Tell JAX inside the worker process not to preallocate the GPU memory."""
    # NOTE: This is called after jax is imported inside the worker process. This
    # means that this approach will not work for selecting the backend.
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"


class RLDSDataLoader:
    """Iterates an IterableTransformedDataset and returns sharded jax.Arrays.

    If you run on multiple JAX processes (e.g. multi-host TPU), each process
    automatically receives its 1/`process_count` share of every batch.
    """

    def __init__(
        self,
        dataset: IterableTransformedDataset,
        *,
        sharding: jax.sharding.Sharding | None = None,
        num_batches: int | None = None,
        auto_shard: bool = False,  # turn off if your dataset is already host-sharded
    ):
        self._dataset = dataset
        self._num_batches = num_batches
        self._auto_shard = auto_shard

        # Default to data-parallel sharding across local devices.
        # if sharding is None:
        #     sharding = jax.sharding.NamedSharding(
        #         jax.sharding.Mesh(jax.devices(), ("B",)),
        #         jax.sharding.PartitionSpec("B"),
        #     )
        self._n_proc = jax.process_count()
        self._proc_idx = jax.process_index()

        if sharding is None:
            if self._n_proc > 1:
                ps = jax.sharding.PositionalSharding(jax.devices())
                ps = ps.reshape(self._n_proc, jax.local_device_count())[self._proc_idx]
                sharding = ps
            else:
                sharding = jax.sharding.PositionalSharding(jax.devices())
        self._sharding = sharding

    def _to_device(self, batch):
        def put(x):
            if not (hasattr(x, "shape") and x.shape):
                return x
            if isinstance(self._sharding, jax.sharding.NamedSharding):
                # Assemble a global jax.Array across processes.
                return jax.make_array_from_process_local_data(self._sharding, x)
            # Per-host sharding (PositionalSharding etc.).
            return jax.device_put(x, self._sharding)

        return jax.tree_util.tree_map(put, batch)

    def _assert_divisible(self, batch):
        sizes = [x.shape[0] for x in jax.tree_util.tree_leaves(batch) if hasattr(x, "shape") and x.shape]
        if not sizes:
            return
        b = max(sizes)  # this is per-host if dataset was shard(...)’ed

        if isinstance(self._sharding, jax.sharding.NamedSharding):
            mesh = self._sharding.mesh
            # DATA axis size across the whole mesh:
            data_axis_size = mesh.shape.get("data", None)  # or use your DATA_AXIS constant
            if data_axis_size is None:
                return  # no data axis; nothing to check

            # Special case: for cross-host FSDP when data_axis_size == 1,
            # we don't need data parallelism across hosts - each host gets the same data
            # and works together on FSDP sharding
            if data_axis_size == 1 and self._n_proc > 1:
                # Cross-host FSDP: validate against local device count instead
                ldc = jax.local_device_count()
                if b % ldc != 0:
                    raise ValueError(
                        f"Per-host batch {b} must be divisible by local_device_count {ldc} for cross-host FSDP"
                    )
                return

            # Standard data parallelism validation
            dp_per_host = data_axis_size // self._n_proc
            if dp_per_host == 0 or data_axis_size % self._n_proc != 0:
                raise ValueError("Mesh/data axis inconsistent with process_count.")
            if b % dp_per_host != 0:
                raise ValueError(f"Per-host batch {b} must be divisible by dp_per_host {dp_per_host}")
        else:
            # PositionalSharding fallback shards leading axis across local devices
            ldc = jax.local_device_count()
            if b % ldc != 0:
                raise ValueError(f"Per-host batch {b} must be divisible by local_device_count {ldc}")

    # ──────────────────────────────────────────────────────────────────────────
    # Helper: split a full batch so that each host keeps only its own rows
    # ──────────────────────────────────────────────────────────────────────────
    def _local_slice(self, batch):
        if not self._auto_shard or self._n_proc == 1:
            return batch

        # infer global batch size
        sizes = [x.shape[0] for x in jax.tree_util.tree_leaves(batch) if hasattr(x, "shape") and x.shape]
        if not sizes:
            return batch
        B = max(sizes)
        per_host = B // self._n_proc
        start = self._proc_idx * per_host
        end = start + per_host

        def _slice(x):
            if hasattr(x, "shape") and x.shape and x.shape[0] == B:
                return x[start:end]
            return x  # leave non-batch leaves alone

        return jax.tree_util.tree_map(_slice, batch)

    # ──────────────────────────────────────────────────────────────────────────
    def __iter__(self):
        seen = 0
        data_iter = iter(self._dataset)
        while True:
            if self._num_batches is not None and seen >= self._num_batches:
                return

            # Pull next preprocessed batch (may block on upstream I/O/TF)
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(self._dataset)
                continue

            self._assert_divisible(batch)
            batch = self._local_slice(batch)
            out = self._to_device(batch)
            seen += 1

            yield out


# class RLDSDataLoader:
#     """Shallow wrapper around the DROID data loader to make it compatible with openpi.

#     All batching already happens in the DROID dataset, so we don't need to do anything here.
#     """

#     def __init__(
#         self,
#         dataset: DroidRldsDataset | DroidCoTRldsDataset,
#         *,
#         sharding: jax.sharding.Sharding | None = None,
#         num_batches: int | None = None,
#     ):
#         self._dataset = dataset
#         self._num_batches = num_batches

#         if jax.process_count() > 1:
#             raise NotImplementedError("Data loading with multiple processes is not supported.")

#         if sharding is None:
#             # Use data parallel sharding by default.
#             sharding = jax.sharding.NamedSharding(
#                 jax.sharding.Mesh(jax.devices(), ("B",)),
#                 jax.sharding.PartitionSpec("B"),
#             )

#         self._sharding = sharding
#         self._num_batches = num_batches

#     def __iter__(self):
#         num_items = 0
#         while True:
#             data_iter = iter(self._dataset)
#             while True:
#                 if self._num_batches is not None and num_items >= self._num_batches:
#                     return
#                 try:
#                     batch = next(data_iter)
#                 except StopIteration:
#                     break  # We've exhausted the dataset. Create a new iterator and start over.
#                 num_items += 1
#                 yield jax.tree.map(lambda x: jax.make_array_from_process_local_data(self._sharding, x), batch)


class DataLoaderImpl(DataLoader):
    def __init__(self, data_config: _config.DataConfig, data_loader: TorchDataLoader | RLDSDataLoader):
        self._data_config = data_config
        self._data_loader = data_loader

    def data_config(self) -> _config.DataConfig:
        return self._data_config

    def __iter__(self):
        for batch in self._data_loader:
            yield _model.Observation.from_dict(batch), batch["actions"]
