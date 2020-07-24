import logging
import os
from typing import Any, Dict, List, Sequence

import torch
from sklearn.model_selection import train_test_split as sk_split

from src.typing import DatasetType, FormulaHash, T

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
        if not isinstance(_item, Sequence):
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
    if isinstance(datapoint, Sequence):
        x = datapoint[0]
    else:
        x = datapoint

    if not isinstance(x, torch.Tensor):
        raise TypeError(f"Dataset elements are not tensors: {type(x)}")

    return x.shape
