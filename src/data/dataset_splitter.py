import logging
from abc import ABC, abstractmethod
from collections import defaultdict
from functools import cached_property
from typing import Any, Dict, Generator, Generic, Iterable, List, Literal, Tuple

import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold
from torch.functional import Tensor

from src.data.auxiliary import NetworkDatasetCollectionWrapper
from src.data.datasets import (
    BaseLabeledDataset,
    LabeledDataset,
    NetworkDataset,
    NoLabelDataset,
    NoLabelSubset,
    TextSequenceDataset,
)
from src.data.vocabulary import Vocabulary
from src.typing import CrossFoldConfiguration, T

logger = logging.getLogger(__name__)


class SubsetSplitter(Generic[T]):
    """
    A dataset splitter that subsets a dataset by the number required.

    Args:
        dataset (DatasetLike[T]): a big dataset that is to subset.
        n_elements (int): the number of elements to be randomly sampled from the dataset. This cannot be larger than the number of elements in the dataset.
        test_size (int): the size of the test partition. This is taken from n_elements so test_size cannot be larger than n_elements when unique_test is False. If unique_test is True, then a partition of the dataset of size test_size is reserved and returned every call. The train partition is disjoint from this test partition. If unique_test is True then test_size does not take elements from n_elements.
        seed (Any): the seed to use for splitting the dataset.
        unique_test (bool, optional): if True pregenerate the split for the test partition and returns it every call. If False randomly chooses the test partition on call and reduce the train partition by test_size. Defaults to True.


    if unique_test == True:
        len(train) = n_elements-test_size
        len(test) = test_size
        both partitions randomly choosen
    if unique_test == False:
        len(train) = n_elements
        len(test) = test_size
        train is randomly choosen on each call. test is the same over all calls (only generated once). train will not contain any element from test as long the original dataset does not contain duplicates.
    """

    def __init__(
        self,
        dataset: NoLabelDataset[T],
        n_elements: int,
        test_size: int,
        seed: Any,
        unique_test: bool = True,
    ):

        if len(dataset) < n_elements:
            raise ValueError(
                "The sample number cannot be smaller than the number of "
                "elements to sample from: dataset has "
                f"{len(dataset)} < {n_elements} elements"
            )

        self.dataset = dataset
        self.sample = n_elements
        self.test = test_size
        self.rand = np.random.default_rng(seed)

        self.indices = list(range(len(self.dataset)))
        self.rand.shuffle(self.indices)

        self.test_partition = None
        if unique_test:
            if test_size > n_elements:
                raise ValueError(
                    "Cannot sample more elements for the test set than "
                    f"the total sampled elements, {test_size} > {n_elements}"
                )

            # generate the unique test partition
            self.test_partition = NoLabelSubset(self.dataset, self.indices[:test_size])
            # remove the selected indices from the available indices
            self.indices = self.indices[test_size:]

            assert len(self.test_partition) == test_size
            assert len(self.indices) == len(self.dataset) - test_size

    def __call__(self):
        ind = self.rand.choice(self.indices, size=self.sample, replace=False)

        if self.test_partition is None:
            test_idx = ind[: self.test]
            train_idx = ind[self.test :]

            train_set = NoLabelSubset(self.dataset, train_idx)
            test_set = NoLabelSubset(self.dataset, test_idx)

        else:
            test_set = self.test_partition
            train_set = NoLabelSubset(self.dataset, ind)

        return train_set, test_set


class PreloadedDataSplitter(SubsetSplitter[T]):
    def __init__(
        self,
        train_dataset: Generator[Tuple[Tuple[float, ...], T], None, None],
        test_dataset: NoLabelDataset[T],
        n_elements_per_distribution: int,
        seed: Any,
    ):

        self.train_dataset_mapping, self.train_dataset = self._regroup_distributions(
            train_dataset
        )

        if (
            len(next(iter(self.train_dataset_mapping.values())))
            < n_elements_per_distribution
        ):
            existing_elements = len(next(iter(self.train_dataset_mapping.values())))
            logger.error(
                f"Requested {n_elements_per_distribution} elements, only {existing_elements} available"
            )
            raise ValueError("Not enough elements for each distribution")

        logger.info(
            "Each sample iteration will yield "
            f"{len(self.train_dataset_mapping) * n_elements_per_distribution} training elements"
        )

        self.sample = n_elements_per_distribution
        self.rand = np.random.default_rng(seed)

        self.test_dataset = test_dataset

    def _regroup_distributions(self, dataset):
        distribution_mapping = defaultdict(list)
        cleaned_dataset = []
        for i, (distribution, data) in enumerate(dataset):
            distribution_mapping[distribution].append(i)
            cleaned_dataset.append(data)
        return distribution_mapping, NoLabelDataset(dataset=cleaned_dataset)

    def __call__(self):
        indices = []
        for distribution, indices_available in self.train_dataset_mapping.items():
            ind = self.rand.choice(indices_available, size=self.sample, replace=False)
            indices.append(ind)

        train_indices = np.concatenate(indices)
        return NoLabelSubset(self.train_dataset, train_indices), self.test_dataset


class PreloadedDataSplitterWithBalancer(SubsetSplitter[T]):
    def __init__(
        self,
        train_dataset: Generator[Tuple[Tuple[float, ...], T], None, None],
        test_dataset: NoLabelDataset[T],
        train_size: int,
        positive_graph_threshold: float,
        positive_distribution_threshold: float,
        seed: Any,
    ):

        (
            positive_distributions,
            negative_distributions,
            self.distribution_graph_mapping,
            self.train_dataset,
        ) = self._calculate_distributions(
            dataset=train_dataset,
            positive_graph_threshold=positive_graph_threshold,
            positive_distribution_threshold=positive_distribution_threshold,
        )

        self.per_iteration_distribution_graphs = {}

        self._calculate_sizes(
            expected_train_size=train_size,
            positive_distributions=positive_distributions,
            negative_distributions=negative_distributions,
        )

        self.rand = np.random.default_rng(seed)

        self.test_dataset = test_dataset

    def _calculate_distributions(
        self,
        dataset,
        positive_graph_threshold: float,
        positive_distribution_threshold: float,
    ):
        cleaned_dataset = []
        distribution_mapping = defaultdict(list)
        distribution_positive_graph_count = defaultdict(int)
        for i, (distribution, data) in enumerate(dataset):
            if data.y.float().mean() >= positive_graph_threshold:
                distribution_positive_graph_count[distribution] += 1
            distribution_mapping[distribution].append(i)
            cleaned_dataset.append(data)

        positive_distributions = {}
        negative_distributions = {}
        for distribution, distribution_indices in distribution_mapping.items():
            positive_count = distribution_positive_graph_count[distribution]
            if (
                float(positive_count) / len(distribution_indices)
                >= positive_distribution_threshold
            ):
                positive_distributions[distribution] = distribution_indices
            else:
                negative_distributions[distribution] = distribution_indices
        return (
            positive_distributions,
            negative_distributions,
            distribution_mapping,
            NoLabelDataset(dataset=cleaned_dataset),
        )

    @staticmethod
    def __sum_dictionary_sizes(list_dict: Dict[Any, List]) -> int:
        return sum(len(value) for value in list_dict.values())

    def _add_to_per_iteration_distribution_graphs(
        self, distributions: Iterable[Any], graph_number: int
    ):
        for distribution in distributions:
            self.per_iteration_distribution_graphs[distribution] = graph_number

    def _calculate_sizes(
        self,
        expected_train_size: int,
        positive_distributions: Dict[Any, List[int]],
        negative_distributions: Dict[Any, List[int]],
    ):
        balancing_size = expected_train_size // 2

        positive_distribution_number = len(positive_distributions)
        negative_distribution_number = len(negative_distributions)

        positive_distribution_graphs = self.__sum_dictionary_sizes(
            positive_distributions
        )
        negative_distribution_graphs = self.__sum_dictionary_sizes(
            negative_distributions
        )

        if (
            positive_distribution_graphs >= balancing_size
            and negative_distribution_graphs >= balancing_size
        ):
            # normal distribute
            # at least 1 graph per distribution
            positive_per_distribution = balancing_size // positive_distribution_number
            negative_per_distribution = balancing_size // negative_distribution_number

            self._add_to_per_iteration_distribution_graphs(
                positive_distributions.keys(),
                graph_number=positive_per_distribution,
            )
            self._add_to_per_iteration_distribution_graphs(
                negative_distributions.keys(),
                graph_number=negative_per_distribution,
            )

            logger.debug(
                "Balanced training set. "
                f"{positive_per_distribution} graphs sampled from "
                f"{positive_distribution_number} positive distributions. "
                f"{negative_per_distribution} graphs sampled from "
                f"{negative_distribution_number} negative distributions"
            )
        elif (
            positive_distribution_graphs < balancing_size
            and negative_distribution_graphs >= balancing_size
        ):
            # all positive, distribute negative
            for distribution, graph_indices in positive_distributions.items():
                self._add_to_per_iteration_distribution_graphs(
                    [distribution],
                    graph_number=len(graph_indices),
                )
            positive_per_distribution = 1  # just to by-pass check

            # chooce from (TOTAL - positives) // number_of_negative_distributions
            negative_per_distribution = (
                expected_train_size - positive_distribution_graphs
            ) // negative_distribution_number
            self._add_to_per_iteration_distribution_graphs(
                negative_distributions.keys(),
                graph_number=negative_per_distribution,
            )

            logger.warning(
                "Unbalanced training set. "
                f"All {positive_distribution_graphs} graphs "
                f"from all {positive_distribution_number} positive distributions. "
                f"{negative_per_distribution} graphs sampled from "
                f"{negative_distribution_number} negative distributions"
            )

        elif (
            positive_distribution_graphs >= balancing_size
            and negative_distribution_graphs < balancing_size
        ):
            # all negative, distribute positive
            for distribution, graph_indices in negative_distributions.items():
                self._add_to_per_iteration_distribution_graphs(
                    [distribution],
                    graph_number=len(graph_indices),
                )
            negative_per_distribution = 1  # just to by-pass check

            # chooce from (TOTAL - negative) // number_of_positive_distributions
            positive_per_distribution = (
                expected_train_size - negative_distribution_graphs
            ) // positive_distribution_number
            self._add_to_per_iteration_distribution_graphs(
                positive_distributions.keys(),
                graph_number=positive_per_distribution,
            )

            logger.warning(
                "Unbalanced training set. "
                f"All {negative_distribution_graphs} graphs "
                f"from all {negative_distribution_number} negative distributions. "
                f"{positive_per_distribution} graphs sampled from "
                f"{positive_distribution_number} positive distributions"
            )

        else:
            raise ValueError(
                "Not enough graphs to distribute. "
                f"Total number of {positive_distribution_graphs + negative_distribution_graphs} "
                f"graphs, but {expected_train_size} were requested por iteration."
            )

        if positive_per_distribution <= 0 or negative_per_distribution <= 0:
            raise ValueError(
                "Per iteration graph count per distribution is less than 1"
            )

    def __call__(self):
        indices = []
        for (
            distribution,
            take,
        ) in self.per_iteration_distribution_graphs.items():
            available_indices = self.distribution_graph_mapping[distribution]
            ind = self.rand.choice(available_indices, size=take, replace=False)
            indices.append(ind)

        train_indices = np.concatenate(indices)
        return NoLabelSubset(self.train_dataset, train_indices), self.test_dataset


class BaseNetworkDatasetCrossFoldSplitter(ABC, Generic[T]):
    def __init__(
        self,
        datasets: List[NetworkDataset[T]],
        crossfold_config: CrossFoldConfiguration,
        use_stratified_kfold: bool,
    ):
        use_stratified = crossfold_config.pop("use_stratified", None)

        if (use_stratified is None and use_stratified_kfold) or use_stratified:
            logger.debug("Using stratified KFold")
            kfold_strategy = StratifiedKFold
        else:
            logger.debug("Using regular KFold")
            kfold_strategy = KFold

        self.waiting_loading = crossfold_config["defer_loading"]
        self.on_demand = not crossfold_config.pop("defer_loading", False)

        self.force_train_hashes = set(crossfold_config.pop("required_train_hashes"))

        self.kfold = kfold_strategy(**crossfold_config)

        self.datasets = datasets
        self.labels = [dataset.labels[0] for dataset in datasets]
        self.hashes_data = {
            dataset.formula_hash: (dataset, labels)
            for (dataset, labels) in zip(self.datasets, self.labels)
        }

        self.folds = {}

        # train|test -> hash -> network dataset
        self.loaded_fold_mappings: List[Dict[str, Dict[str, NetworkDataset[T]]]] = []

    def load_precalculated_folds(
        self,
        fold_dict: Dict[str, Dict[str, List[Dict[str, str]]]],
    ):
        formula_hash_to_network_data = {
            network_data._formula_hash: network_data for network_data in self.datasets
        }

        folds = list(map(str, sorted(map(int, fold_dict.keys()))))

        # fold format:
        # fold_index -> train|test -> List[hash->hash, formula->formula]
        for fold_index in folds:
            fold_data = fold_dict[fold_index]

            loaded_fold: Dict[str, Dict[str, NetworkDataset[T]]] = {}

            train_fold: Dict[str, NetworkDataset[T]] = {}
            for formula_data in fold_data["train"]:
                formula_hash = formula_data["hash"]
                train_fold[formula_hash] = formula_hash_to_network_data[formula_hash]

            test_fold: Dict[str, NetworkDataset[T]] = {}
            for formula_data in fold_data["test"]:
                formula_hash = formula_data["hash"]
                test_fold[formula_hash] = formula_hash_to_network_data[formula_hash]

            loaded_fold["train"] = train_fold
            loaded_fold["test"] = test_fold

            self.loaded_fold_mappings.append(loaded_fold)

        assert len(self.loaded_fold_mappings) > 0, "Loaded fold must be more than 0"
        self.waiting_loading = False
        self.folds = {
            int(fold_index): fold_data for fold_index, fold_data in fold_dict.items()
        }

    @abstractmethod
    def build_output_dataset(
        self, dataset_data: List[NetworkDataset[T]]
    ) -> BaseLabeledDataset[Tensor, T]:
        ...

    def __on_demand_split(self):
        datasets: List[NetworkDataset[T]] = []
        labels = []

        force_train_datasets: List[NetworkDataset[T]] = []

        for formula_hash, (dataset, label) in self.hashes_data.items():
            if formula_hash in self.force_train_hashes:
                force_train_datasets.append(dataset)
            else:
                datasets.append(dataset)
                labels.append(label)

        for i, (train_index, test_index) in enumerate(
            self.kfold.split(X=datasets, y=labels), start=1
        ):
            train_datasets = [datasets[i] for i in train_index] + force_train_datasets
            test_datasets = [datasets[i] for i in test_index]

            train_dataset = self.build_output_dataset(dataset_data=train_datasets)
            test_dataset = self.build_output_dataset(dataset_data=test_datasets)

            reconstruction_mapping = NetworkDatasetCollectionWrapper(test_datasets)

            self.folds[i] = {
                "train": [
                    {
                        "hash": dataset.formula_hash,
                        "formula": repr(dataset.formula),
                    }
                    for dataset in train_datasets
                ],
                "test": [
                    {
                        "hash": dataset.formula_hash,
                        "formula": repr(dataset.formula),
                    }
                    for dataset in test_datasets
                ],
            }

            yield train_dataset, test_dataset, reconstruction_mapping

    def __preloaded_splits(self):
        for fold in self.loaded_fold_mappings:
            train_datasets = list(fold["train"].values())
            test_datasets = list(fold["test"].values())

            train_dataset = self.build_output_dataset(dataset_data=train_datasets)
            test_dataset = self.build_output_dataset(dataset_data=test_datasets)

            reconstruction_mapping = NetworkDatasetCollectionWrapper(test_datasets)

            yield train_dataset, test_dataset, reconstruction_mapping

    def __iter__(self):
        if self.on_demand:
            yield from self.__on_demand_split()
        else:
            if self.waiting_loading:
                raise RuntimeError(
                    "Defer loading is active, waiting for concrete settings"
                )
            yield from self.__preloaded_splits()

    @property
    def n_splits(self) -> int:
        if self.on_demand:
            return self.kfold.n_splits
        else:
            return len(self.loaded_fold_mappings)

    @cached_property
    def dataset_size(self):
        size = 0
        for dataset in self.datasets:
            size += len(dataset)
        return size

    def get_folds(self) -> Dict[int, Dict[str, List[Dict[str, str]]]]:
        return self.folds


class NetworkDatasetCrossFoldSplitter(BaseNetworkDatasetCrossFoldSplitter[T]):
    def __init__(
        self,
        datasets: List[NetworkDataset[T]],
        crossfold_config: CrossFoldConfiguration,
    ):
        self.multilabel = datasets[0].multilabel

        super().__init__(
            datasets=datasets,
            crossfold_config=crossfold_config,
            use_stratified_kfold=not self.multilabel,
        )

    def build_output_dataset(
        self, dataset_data: List[NetworkDataset[T]]
    ) -> LabeledDataset[Tensor, T]:
        return LabeledDataset.from_iterable(dataset_data, multilabel=self.multilabel)


class TextNetworkDatasetCrossFoldSplitter(BaseNetworkDatasetCrossFoldSplitter[Tensor]):
    def __init__(
        self,
        datasets: List[NetworkDataset[Tensor]],
        crossfold_config: CrossFoldConfiguration,
        vocabulary: Vocabulary,
    ):
        super().__init__(
            datasets=datasets,
            crossfold_config=crossfold_config,
            use_stratified_kfold=False,
        )
        self.vocabulary = vocabulary

    def build_output_dataset(
        self, dataset_data: List[NetworkDataset[Tensor]]
    ) -> TextSequenceDataset[Tensor]:
        return TextSequenceDataset.from_iterable(
            dataset_data, vocabulary=self.vocabulary
        )

    def _grouped_formulas(
        self, cv_fold, dataset_name: Literal["train", "test"]
    ) -> Generator[Tuple[str, TextSequenceDataset[Tensor]], None, None]:
        for formula_hash, network_dataset in cv_fold[dataset_name].items():
            yield formula_hash, self.build_output_dataset(
                dataset_data=[network_dataset]
            )

    def group_formulas_per_dataset_name(self, dataset_name: Literal["train", "test"]):
        for fold in self.loaded_fold_mappings:
            yield self._grouped_formulas(fold, dataset_name=dataset_name)
