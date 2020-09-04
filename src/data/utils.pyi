from typing import Dict, Tuple, overload

import torch

from src.data.datasets import LabeledSubset, Subset
from src.typing import DatasetLike, LabeledDatasetLike, S, T


@overload
def train_test_dataset(dataset: LabeledDatasetLike[T, S],
                       test_size: float = ...,
                       random_state: int = ...,
                       shuffle: bool = ...,
                       stratify: bool = ...,
                       multilabel: bool = ...) -> Tuple[LabeledSubset[T, S],
                                                        LabeledSubset[T, S]]: ...


@overload
def train_test_dataset(dataset: DatasetLike[T],
                       test_size: float = ...,
                       random_state: int = ...,
                       shuffle: bool = ...,
                       stratify: bool = ...,
                       multilabel: bool = ...) -> Tuple[Subset[T],
                                                        Subset[T]]: ...


def get_input_dim(data) -> torch.Size: ...


def get_label_distribution(
    dataset: LabeledDatasetLike[T, S]) -> Dict[S, float]: ...
