import logging
import warnings
from collections import defaultdict
from typing import Union

import torch
from sklearn.model_selection import train_test_split as sk_split

from src.data.datasets import LabeledSubset, Subset
from src.typing import DatasetLike, Indexable, LabeledDatasetLike, S, T

logger = logging.getLogger(__name__)


def train_test_dataset(
        dataset: Union[LabeledDatasetLike[T, S], DatasetLike[T]],
        test_size: float = 0.25,
        random_state: int = None,
        shuffle: bool = True,
        stratify: bool = True,
        multilabel: bool = False):

    classes = None
    if stratify:
        if multilabel:
            warnings.warn(
                "Cannot use the stratified data splitting with multilabels. "
                "Ignoring...", UserWarning, stacklevel=2)
        else:
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


def get_label_distribution(dataset: LabeledDatasetLike[T, S],
                           multilabel: bool = False):
    label_info = dataset.label_info
    n_elements = len(dataset)
    if multilabel:
        classes = defaultdict(int)
        for labels, n_items in label_info.items():
            for label in labels:
                classes[label] += n_items
    else:
        classes = label_info

    return {k: float(v) / n_elements for k, v in classes.items()}
