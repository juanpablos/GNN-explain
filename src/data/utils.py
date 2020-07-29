import logging
import os
from collections import defaultdict
from typing import Any, Callable, Dict, Generic, Hashable, List

import numpy as np
import torch
from sklearn.model_selection import train_test_split as sk_split

from src.typing import DatasetType, FormulaHash, Indexable, T

from .datasets import MergedDataset, NetworkDataset, Subset


def clean_state(model_dict: Dict[str, Any]):
    """Removes the weights associated with batchnorm"""
    return {k: v for k, v in model_dict.items() if "batch" not in k}


def load_gnn_files(root: str, model_hash: str,
                   formula_hashes: FormulaHash, load_all: bool):
    """
    formula hashes has the following format
    hash: {
        limit: number,
        label: any
    }
    """

    def _prepare_files(path: str):
        files: Dict[str, str] = {}
        # reproducibility, always sorted files
        for file in sorted(os.listdir(path)):
            if file.endswith(".pt"):
                _hash = file.split(".")[0].split("-")[-1]
                files[_hash] = file
        return files

    if model_hash not in os.listdir(root):
        raise FileExistsError(
            f"No directory for the current model hash: {root}")

    model_path = os.path.join(root, model_hash)
    dir_formulas = _prepare_files(model_path)

    if load_all:
        formula_configs: FormulaHash = {}
        for label, f_hash in enumerate(dir_formulas):
            formula_configs[f_hash] = {
                "label": label,
                "limit": None
            }
    else:
        if not all(f in dir_formulas for f in formula_hashes):
            _not = [f for f in formula_hashes if f not in dir_formulas]
            raise ValueError(
                f"Not all requested formula hashes are present in the directory: {_not}")

        formula_configs = formula_hashes

    datasets: List[NetworkDataset] = []
    for formula_hash, config in formula_configs.items():
        logging.info(f"\tLoading {formula_hash}")

        file_path = os.path.join(model_path, dir_formulas[formula_hash])
        dataset = NetworkDataset(file=file_path, **config)

        datasets.append(dataset)

    return MergedDataset(datasets)


def train_test_dataset(
        dataset: DatasetType[T],
        test_size: float = 0.25,
        random_state: int = None,
        shuffle: bool = True,
        stratify: bool = True):

    classes = None
    if stratify:
        _item = next(iter(dataset))
        if not isinstance(_item, Indexable):
            raise TypeError("Elements of the dataset must be tuple-like")
        if len(_item) < 2:
            raise ValueError(
                "The return type of an item from the dataset must be at least of length 2")

        classes = [data[-1] for data in dataset]

    train_idx, test_idx = sk_split(list(range(len(dataset))),
                                   test_size=test_size,
                                   random_state=random_state,
                                   shuffle=shuffle,
                                   stratify=classes)

    return Subset(dataset, train_idx), Subset(dataset, test_idx)


def get_input_dim(data):
    datapoint = next(iter(data))
    if isinstance(datapoint, Indexable):
        x = datapoint[0]
    else:
        x = datapoint

    if not isinstance(x, torch.Tensor):
        raise TypeError(f"Dataset elements are not tensors: {type(x)}")

    return x.shape


def get_label_distribution(
        dataset: DatasetType[T], getter: Callable[[T], Any] = None):
    _item = next(iter(dataset))
    if not isinstance(_item, Indexable):
        raise TypeError("Elements of the dataset must be tuple-like")
    if len(_item) < 2:
        raise ValueError(
            "The return type of an item from the dataset must be at least of length 2")

    if getter is None:
        getter = lambda x: x[-1]

    if not isinstance(getter(_item), Hashable):
        raise TypeError(
            f"Label elements must be hashable, type {type(_item[-1])} is not")

    counter: Dict[Any, float] = defaultdict(float)
    elements = 0
    for item in dataset:
        counter[getter(item)] += 1
        elements += 1

    for k in counter:
        counter[k] /= elements

    return dict(counter)


class SubsetSampler(Generic[T]):
    """
    A dataset sampler that subsets a dataset by the number required.

    Args:
        dataset (DatasetType[T]): a big dataset that is to subset.
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
            dataset: DatasetType[T],
            n_elements: int,
            test_size: int,
            seed: Any,
            unique_test: bool = True):

        if len(dataset) < n_elements:
            raise ValueError(
                f"The sample number cannot be smaller than the number of elements to sample from: dataset has {len(dataset)} < {n_elements}")

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
                    f"Cannot sample more elements for the test set than the total sampled elements, {test_size} > {n_elements}")

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
