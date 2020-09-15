from typing import Any, Dict, List, Tuple, overload

import torch

from src.data.datasets import (
    LabeledDataset,
    LabeledSubset,
    NoLabelDataset,
    NoLabelSubset
)
from src.typing import S, T


@overload
def train_test_dataset(dataset: LabeledDataset[T, S],
                       test_size: float = ...,
                       random_state: int = ...,
                       shuffle: bool = ...,
                       stratify: bool = ...,
                       multilabel: bool = ...) -> Tuple[LabeledSubset[T, S],
                                                        LabeledSubset[T, S]]: ...


@overload
def train_test_dataset(dataset: NoLabelDataset[T],
                       test_size: float = ...,
                       random_state: int = ...,
                       shuffle: bool = ...,
                       stratify: bool = ...,
                       multilabel: bool = ...) -> Tuple[NoLabelSubset[T],
                                                        NoLabelSubset[T]]: ...


def get_input_dim(data) -> torch.Size: ...


def get_label_distribution(
    dataset: LabeledDataset[T, S]) -> Tuple[Dict[S, int], Dict[S, float]]: ...


def label_idx2tensor(label: List[Any], n_labels: int) -> torch.Tensor: ...


def label_tensor2idx(label: torch.Tensor) -> List[Any]: ...
