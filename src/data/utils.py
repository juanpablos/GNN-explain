import logging
import warnings
from typing import Counter, Union

import torch
from sklearn.model_selection import train_test_split as sk_split

from src.data.datasets import (
    LabeledDataset,
    LabeledSubset,
    NoLabelDataset,
    NoLabelSubset
)
from src.typing import Indexable, S, T

logger = logging.getLogger(__name__)


def train_test_dataset(
        dataset: Union[LabeledDataset[T, S], NoLabelDataset[T]],
        test_size: float = 0.25,
        random_state: int = None,
        shuffle: bool = True,
        stratify: bool = True,
        # !! remove multilabel warning
        multilabel: bool = False):

    classes = None
    if stratify:
        if multilabel:
            warnings.warn(
                "Cannot use stratified data splitting with multilabels. "
                "Ignoring...", UserWarning, stacklevel=2)
        else:
            if not isinstance(dataset, LabeledDataset):
                raise ValueError("`dataset` is not a labeled dataset")

            classes = dataset.labels

    train_idx, test_idx = sk_split(list(range(len(dataset))),
                                   test_size=test_size,
                                   random_state=random_state,
                                   shuffle=shuffle,
                                   stratify=classes)

    if isinstance(dataset, LabeledDataset):
        return LabeledSubset(dataset, train_idx),\
            LabeledSubset(dataset, test_idx)
    else:
        return NoLabelSubset(
            dataset, train_idx), NoLabelSubset(
            dataset, test_idx)


def get_input_dim(data):
    datapoint = next(iter(data))
    if isinstance(datapoint, Indexable):
        x = datapoint[0]
    else:
        x = datapoint

    if not isinstance(x, torch.Tensor):
        raise TypeError(f"Dataset elements are not tensors: {type(x)}")

    return x.shape


def get_label_distribution(dataset: LabeledDataset[T, S]):
    label_info = Counter()

    if dataset.multilabel:
        for label in dataset.labels:
            label_info.update(label)
    else:
        label_info.update(dataset.labels)
    n_elements = len(dataset)

    return label_info, {k: float(v) /
                        n_elements for k, v in label_info.items()}
