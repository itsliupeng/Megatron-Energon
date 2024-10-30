from abc import ABC
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Iterator,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
)

from megatron.energon.epathlib import EPath
from megatron.energon.flavors.base_dataset import (
    BaseCoreDataset,
    Sample,
    SavableDataset,
    set_sample_restore_key,
)
from megatron.energon.flavors.webdataset.base_webdataset import (
    BaseWebdataset,
    WebdatasetMergedState,
    WebdatasetState,
)
from megatron.energon.flavors.webdataset.error_handler import ErrorHandler
from megatron.energon.flavors.webdataset.sample_loader import WebdatasetSampleLoaderDataset
from megatron.energon.flavors.webdataset.sharder import Sharder
from megatron.energon.flavors.webdataset.structs import FilteredSample, ShardInfo, reraise_exception
from megatron.energon.worker import WorkerConfig
from megatron.energon.wrappers.map_dataset import MapDataset

T_sample = TypeVar("T_sample", covariant=True)


class MergedWebdataset(
    BaseCoreDataset[T_sample], Sharder, ErrorHandler[T_sample], Generic[T_sample], ABC
):
    """
    Base class for all webdataset loaders. Applies proper sharding across workers.
    """

    _sample_merger: Callable[[Callable[..., T_sample], Tuple[Optional[Sample], ...]], T_sample]

    training: bool
    worker_config: WorkerConfig

    shards: List[Sequence[ShardInfo]]
    part_datasets: SavableDataset[T_sample]

    inner_datasets: List[BaseWebdataset]
    inner_dataset_keys: Optional[List[str]]

    def __init__(
        self,
        inner_datasets: Union[List[BaseWebdataset], Dict[str, BaseWebdataset]],
        *,
        training: bool,
        worker_config: WorkerConfig,
        shuffle_over_epochs: int = 1,
        parallel_shard_iters: Optional[int] = None,
        max_samples_per_sequence: Optional[int] = None,
        part_filter: Optional[Callable[[str], bool]] = None,
        join_method: Literal["inner_match"] = "inner_match",
        join_type: Type[T_sample],
        handler: Callable[[Exception, Optional[str]], None] = reraise_exception,
    ):
        """
        Constructs the webdataset loader.

        Args:
            inner_dataset: The inner datasets. Must be loaded internally with `_is_composed=True`.
            training: If true, apply shuffling and loop the dataset.
            worker_config: Configuration for the workers.
            shuffle_over_epochs: Only effective if training=True.
                How many epochs to shuffle over if training.
                If = 1, every sample is seen exactly once per epoch.
                If > 1, samples (or rather shard slices) are shuffled within this number of epochs
                (i.e. randomly selected without replacement).
                If -1, the shards are effectively shuffle over infinite epochs (i.e. shard slices
                are drawn with replacement).
            parallel_shard_iters: Number of parallel opened shards per worker, shuffling between.
            max_samples_per_sequence: Maximum number of samples per sequence (=how many samples
                    will be sequentially iterated).
            part_filter: (internal) Function for filtering tar files by dict keys
            join_method: How to join the samples.
                inner_match: All samples must match 1:1 of the merged datasets.
                This might be extended to further modes in the future, but those will require a new index, which
            join_type: Type of the joined samples.
            handler: Exception handler. Args: (exception, key).
        """
        self.__sample_type__ = join_type
        assert self.__sample_type__ is not None, f"Class {type(self)} must define __sample_type__"
        if isinstance(inner_datasets, dict):
            inner_keys = list(inner_datasets.keys())
            self.inner_dataset_keys = inner_keys
            self._sample_merger = lambda composer, samples: composer(
                **dict(zip(inner_keys, samples))
            )
            inner_datasets = list(inner_datasets.values())
        else:
            self._sample_merger = lambda composer, samples: composer(*samples)
            self.inner_dataset_keys = None
        assert all(
            not hasattr(d, "dataset") for d in inner_datasets
        ), "Inner dataset was not instantiated with _is_composed=True"
        self.inner_datasets = inner_datasets
        self.training = training
        self.worker_config = worker_config
        self.handler = handler
        assert all(
            len(dataset.shards) == len(inner_datasets[0].shards) for dataset in inner_datasets[1:]
        ), f"Dataset structures do not match, shards differ"
        sample_exclude = inner_datasets[0].sample_excludes
        assert all(
            sample_exclude == dataset.sample_excludes for dataset in inner_datasets[1:]
        ), f"Sample excludes must be the same for all paths"

        if join_method == "inner_match":
            assert all(
                shard1.count == shard2.count
                for dataset in inner_datasets[1:]
                for shard1, shard2 in zip(dataset.shards, inner_datasets[0].shards)
            ), "For inner_match, all shards must have the same count"
        else:
            assert False, f"Invalid join method {join_method}"

        self.shards = list(zip(*(dataset.shards for dataset in inner_datasets)))

        if parallel_shard_iters is None:
            if training:
                # 16 seems to be a good choice since we don't want too many file handles open
                parallel_shard_iters = 16
            else:
                parallel_shard_iters = 1

        self.rank_shards = self.shard_workers(
            self.shards,
            self.worker_config,
            max_samples_per_sequence=max_samples_per_sequence,
        )

        self.rank_total = sum(
            subshard[0].count for shards in self.rank_shards for subshard in shards
        )
        for rank_idx, shards in enumerate(self.rank_shards):
            shards_text = ", ".join(
                f"{subshard[0].name}[{subshard[0].offset}, {subshard[0].offset+subshard[0].count})"
                for subshard in shards[:3]
            )
            if len(shards) > 6:
                shards_text += f", ...<{len(shards) - 6}>, " + ", ".join(
                    f"{subshards[0].name}[{subshards[0].offset}, {subshards[0].offset+subshards[0].count})"
                    for subshards in shards[-3:]
                )
            elif len(shards) > 3:
                shards_text += ", " + ", ".join(
                    f"{subshards[0].name}[{subshards[0].offset}, {subshards[0].offset+subshards[0].count})"
                    for subshards in shards[3:]
                )
            print(
                f"rank={self.worker_config.rank}, worker={rank_idx}: shard_range="
                f"[{shards_text}] "
                f"sum(count)={sum(subshards[0].count for subshards in shards)}"
            )

        dataset = WebdatasetSampleLoaderDataset(
            rank_shards=self.rank_shards,
            worker_config=self.worker_config,
            part_filter=part_filter,
            exclude=sample_exclude,
            loop=training,
            shuffle_over_epochs=shuffle_over_epochs if training else None,
            parallel_shard_iters=parallel_shard_iters,
            dataset_join_method=join_method,
            handler=self.sample_error_handler,
        )
        self.dataset = self._process_samples(dataset)

    @property
    def paths(self) -> List[EPath]:
        return [dataset.path for dataset in self.inner_datasets]

    def _process_samples(
        self, dataset: SavableDataset[Tuple[Optional[FilteredSample], ...]]
    ) -> SavableDataset[T_sample]:
        """Internally loads the sample."""
        return MapDataset(
            dataset,
            self.load_sample,
            error_handler=self.error_handler,
            stateless_map_fn=True,
            worker_config=self.worker_config,
        )

    def load_sample(self, samples: Tuple[Optional[FilteredSample], ...]) -> T_sample:
        assert len(samples) > 0 and samples[0] is not None, "Always need primary sample"
        # Combine the restore key. This must be in accordance to the ShardReader's restore unpacking
        restore_key = [
            *samples[0]["__restore_key__"],
        ]
        for sample in samples[1:]:
            if sample is None:
                restore_key.append("")
                restore_key.append(-1)
            else:
                restore_key.extend(sample["__restore_key__"][1:3])

        # First call the loaders of all inner datasets
        loaded_samples = tuple(
            None if sample is None else dataset.load_sample(sample)
            for dataset, sample in zip(self.inner_datasets, samples)
        )
        # Then combine the loaded smaples into the final type
        assert issubclass(self.__sample_type__, Sample), "Merged dataset must be of type Sample"
        return set_sample_restore_key(
            self._sample_merger(self.__sample_type__.from_joined, loaded_samples),
            *restore_key,
            src=self,
            fail_otherwise=True,
        )

    def __len__(self):
        # In the training case, the result is an approximation (i.e. number of different samples)
        return self.rank_total

    def __iter__(self) -> Iterator[T_sample]:
        yield from self.dataset

    def worker_has_samples(self) -> bool:
        return self.dataset.worker_has_samples()

    def save_state(self) -> WebdatasetState:
        return WebdatasetState(
            dataset_state=self.dataset.save_state(),
        )

    def merge_states(self, states: List[WebdatasetState]) -> WebdatasetMergedState:
        assert all(s is None or isinstance(s, WebdatasetState) for s in states)
        return WebdatasetMergedState(
            dataset_state=self.dataset.merge_states(
                [None if s is None else s.dataset_state for s in states]
            ),
        )

    def restore_state(self, state: Optional[WebdatasetMergedState]) -> None:
        if state is None:
            self.dataset.restore_state(None)
        else:
            assert isinstance(state, WebdatasetMergedState)
            self.dataset.restore_state(state.dataset_state)

    def can_restore_sample(self) -> bool:
        return self.dataset.can_restore_sample()

    def assert_can_restore(self):
        self.dataset.assert_can_restore()

    def restore_sample(self, key: Tuple[Union[str, int, tuple], ...]) -> T_sample:
        return self.dataset.restore_sample(key)

    def config(self) -> Dict[str, Any]:
        return {
            "type": type(self).__qualname__,
            "training": self.training,
            "inner_datasets": [dataset.config() for dataset in self.inner_datasets],
        }

    def __str__(self):
        return f"{type(self).__name__}(paths={self.paths}, dataset={self.dataset})"
