import numpy as np
import logging
import os
from collections import defaultdict
from typing import Any, Callable, Dict, Generic, Hashable, List

import torch
from sklearn.model_selection import train_test_split as sk_split

from src.typing import DatasetType, FormulaHash, Indexable, T

from .datasets import MergedDataset, NetworkDataset, Subset


def clean_state(model_dict: Dict[str, Any]):
    """Removes the weights associated with batchnorm"""
    return {k: v for k, v in model_dict.items() if "batch" not in k}


def load_gnn_files(root: str, model_hash: str,
                   formula_hashes: FormulaHash):
    """
    formula hashes has the following format
    hash: {
        limit: number,
        label: any
    }
    """

    def _prepare_files(path: str):
        files: Dict[str, str] = {}
        for file in os.listdir(path):
            if file.endswith(".pt"):
                _hash = file.split(".")[0].split("-")[-1]
                files[_hash] = file
        return files

    if model_hash not in os.listdir(root):
        raise FileExistsError(
            f"No directory for the current model hash: {root}")

    model_path = os.path.join(root, model_hash)
    dir_formulas = _prepare_files(model_path)

    if not all(f in dir_formulas for f in formula_hashes):
        _not = [f for f in formula_hashes if f not in dir_formulas]
        raise ValueError(
            f"Not all requested formula hashes are present in the directory: {_not}")

    datasets: List[NetworkDataset] = []
    for formula_hash, config in formula_hashes.items():
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
    def __init__(
            self,
            dataset: DatasetType[T],
            n_graphs: int,
            test_size: int,
            seed: Any):
        self.dataset = dataset
        self.sample = n_graphs
        self.test = test_size

        if len(dataset) < n_graphs:
            raise ValueError(
                f"The sample number cannot be smaller than the number of elements to sample from: dataset has {len(dataset)} < {n_graphs}")
        if test_size > n_graphs:
            raise ValueError(
                f"Cannot sample more elements for the test set than the total sampled elements, {test_size} > {n_graphs}")

        self.indices = list(range(len(self.dataset)))
        self.rand = np.random.default_rng(seed)

    def __call__(self):
        ind = self.rand.choice(self.indices, size=self.sample, replace=False)
        test_idx = ind[:self.test]
        train_idx = ind[self.test:]

        return Subset(self.dataset, train_idx), Subset(self.dataset, test_idx)
