import logging
from collections import defaultdict
from typing import Any, Dict, Generator, Generic, List, Tuple

import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold
from torch.functional import Tensor
from src.data.datasets import (
    LabeledDataset,
    NetworkDataset,
    NoLabelDataset,
    NoLabelSubset,
)
from src.typing import CrossFoldConfiguration, T

logger = logging.getLogger(__name__)


class SubsetSampler(Generic[T]):
    """
    A dataset sampler that subsets a dataset by the number required.

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


class PreloadedDataSampler(SubsetSampler[T]):
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


class NetworkDatasetCrossFoldSampler(Generic[T]):
    def __init__(
        self,
        datasets: List[NetworkDataset[Tensor]],
        crossfold_config: CrossFoldConfiguration,
    ):
        self.multilabel = datasets[0].multilabel

        if self.multilabel:
            logger.debug("Using regular KFold")
            kfold_strategy = KFold
        else:
            logger.debug("Using stratified KFold")
            kfold_strategy = StratifiedKFold

        self.kfold = kfold_strategy(**crossfold_config)

        self.datasets = datasets
        self.labels = [dataset.labels[0] for dataset in datasets]

        self.folds = {}

    def __iter__(self):
        for i, (train_index, test_index) in enumerate(
            self.kfold.split(X=self.datasets, y=self.labels), start=1
        ):
            train_datasets = [self.datasets[i] for i in train_index]
            test_datasets = [self.datasets[i] for i in test_index]

            train_dataset = LabeledDataset.from_iterable(
                train_datasets, multilabel=self.multilabel
            )
            test_dataset = LabeledDataset.from_iterable(
                test_datasets, multilabel=self.multilabel
            )

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

            yield train_dataset, test_dataset

    @property
    def n_splits(self) -> int:
        return self.kfold.n_splits

    def get_folds(self) -> Dict[int, Dict[str, List[Dict[str, str]]]]:
        return self.folds
