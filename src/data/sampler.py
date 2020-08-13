
from typing import Any, Generic

import numpy as np

from src.typing import DatasetLike, T

from .datasets import Subset


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
            dataset: DatasetLike[T],
            n_elements: int,
            test_size: int,
            seed: Any,
            unique_test: bool = True):

        if len(dataset) < n_elements:
            raise ValueError(
                "The sample number cannot be smaller than the number of "
                "elements to sample from: dataset has "
                f"{len(dataset)} < {n_elements} elements")

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
                    f"the total sampled elements, {test_size} > {n_elements}")

            # generate the unique test partition
            self.test_partition = Subset(
                self.dataset, self.indices[:test_size])
            # remove the selected indices from the available indices
            self.indices = self.indices[test_size:]

            assert len(self.test_partition) == test_size
            assert len(self.indices) == len(self.dataset) - test_size

    def __call__(self):
        ind = self.rand.choice(
            self.indices,
            size=self.sample,
            replace=False)

        if self.test_partition is None:
            test_idx = ind[:self.test]
            train_idx = ind[self.test:]

            train_set = Subset(self.dataset, train_idx)
            test_set = Subset(self.dataset, test_idx)

        else:
            test_set = self.test_partition
            train_set = Subset(self.dataset, ind)

        return train_set, test_set
