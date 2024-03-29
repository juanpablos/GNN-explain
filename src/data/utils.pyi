from typing import Any, Dict, List, Tuple, Union, overload

import torch

from src.data.datasets import (
    BaseLabeledDataset,
    LabeledDataset,
    LabeledSubset,
    NoLabelDataset,
    NoLabelSubset,
)
from src.typing import S, T

@overload
def train_test_dataset(
    dataset: BaseLabeledDataset[T, S],
    test_size: float = ...,
    random_state: int = ...,
    shuffle: bool = ...,
    stratify: bool = ...,
    multilabel: bool = ...,
) -> Tuple[LabeledSubset[T, S], LabeledSubset[T, S]]: ...
@overload
def train_test_dataset(
    dataset: NoLabelDataset[T],
    test_size: float = ...,
    random_state: int = ...,
    shuffle: bool = ...,
    stratify: bool = ...,
    multilabel: bool = ...,
) -> Tuple[NoLabelSubset[T], NoLabelSubset[T]]: ...
def get_input_dim(data) -> torch.Size: ...
def get_label_distribution(
    dataset: Union[LabeledDataset[T, S], LabeledSubset[T, S]]
) -> Tuple[Dict[S, int], Dict[S, float]]: ...
def label_idx2tensor(label: List[Any], n_labels: int) -> torch.Tensor: ...
def label_tensor2idx(label: torch.Tensor) -> List[int]: ...
def get_tensor_dict_input_layer_dim(data) -> int: ...
def get_tensor_dict_output_layer_dim(data) -> int: ...
