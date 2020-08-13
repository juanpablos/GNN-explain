import logging
from typing import Any, Dict, Tuple, Union, overload

import torch
from sklearn.model_selection import train_test_split as sk_split

from src.typing import DatasetLike, Indexable, LabeledDatasetLike, S, T

from .datasets import LabeledSubset, Subset

logger = logging.getLogger(__name__)


def clean_state(model_dict: Dict[str, Any]):
    """Removes the weights associated with batchnorm"""
    return {k: v for k, v in model_dict.items() if "batch" not in k}


@overload
def train_test_dataset(dataset: LabeledDatasetLike[T, S],
                       test_size: float = ...,
                       random_state: int = ...,
                       shuffle: bool = ...,
                       stratify: bool = ...) -> Tuple[LabeledSubset[T, S],
                                                      LabeledSubset[T, S]]: ...


@overload
def train_test_dataset(dataset: DatasetLike[T],
                       test_size: float = ...,
                       random_state: int = ...,
                       shuffle: bool = ...,
                       stratify: bool = ...) -> Tuple[Subset[T],
                                                      Subset[T]]: ...


def train_test_dataset(
        dataset: Union[LabeledDatasetLike[T, S], DatasetLike[T]],
        test_size: float = 0.25,
        random_state: int = None,
        shuffle: bool = True,
        stratify: bool = True):

    classes = None
    if stratify:
        if not isinstance(dataset, LabeledDatasetLike):
            raise ValueError("`dataset` is not a labeled dataset")

        classes = dataset.labels

    train_idx, test_idx = sk_split(list(range(len(dataset))),
                                   test_size=test_size,
                                   random_state=random_state,
                                   shuffle=shuffle,
                                   stratify=classes)

    if isinstance(dataset, LabeledDatasetLike):
        return LabeledSubset(dataset, train_idx),\
            LabeledSubset(dataset, test_idx)
    else:
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


def get_label_distribution(dataset: LabeledDatasetLike[T, S]):
    label_info = dataset.label_info
    n_elements = len(dataset)

    return {k: float(v) / n_elements for k, v in label_info.items()}
