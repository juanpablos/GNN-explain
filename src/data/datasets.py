from abc import ABC
from collections import Counter
from typing import Dict, Generic, Iterator, List, Mapping, Sequence, Tuple

import torch
from torch.utils.data import Dataset

from src.typing import (DatasetLike, Indexable, IndexableIterable,
                        LabeledDatasetLike, S_co, T_co)


class DummyIterable(Generic[T_co]):
    def __init__(self, value: T_co, length: int):
        self.value: T_co = value
        self.length: int = length

    def __len__(self):
        return self.length

    def __getitem__(self, index: int):
        return self.value

    def __iter__(self):
        for _ in range(self.length):
            yield self.value


class SimpleDataset(Dataset, Generic[T_co]):
    def __init__(self, dataset: IndexableIterable[T_co]):
        self._dataset: IndexableIterable[T_co] = dataset

    def __getitem__(self, index: int) -> T_co:
        return self.dataset[index]

    def __len__(self) -> int:
        return len(self.dataset)

    def __iter__(self) -> Iterator[T_co]:
        return iter(self.dataset)

    @property
    def dataset(self) -> IndexableIterable[T_co]:
        return self._dataset


class RandomGraphDataset(SimpleDataset[T_co]):
    """A Pytorch dataset that takes a (infinite) generator and stores 'limit' elements of it."""

    def __init__(self, generator: Iterator[T_co], limit: int):
        super().__init__(dataset=[next(generator) for _ in range(limit)])


class LabeledDatasetBase(ABC, Generic[T_co, S_co]):
    def __init__(self):
        # only declaration
        self._dataset: IndexableIterable[T_co]
        self._labels: IndexableIterable[S_co]
        self._label_info: Dict[S_co, int]
        self._label_info_loaded: bool = False

    def __getitem__(self, index: int) -> Tuple[T_co, S_co]:
        return self.dataset[index], self.labels[index]

    def __len__(self) -> int:
        return len(self.dataset)

    def __iter__(self) -> Iterator[Tuple[T_co, S_co]]:
        for x, y in zip(self.dataset, self.labels):
            yield x, y

    @property
    def dataset(self) -> IndexableIterable[T_co]:
        return self._dataset

    @property
    def labels(self) -> IndexableIterable[S_co]:
        return self._labels

    @property
    def label_info(self) -> Mapping[S_co, int]:
        self._load()
        return dict(self._label_info)

    def _load(self):
        if not self._label_info_loaded:
            self._label_info = Counter(self.labels)
            self._label_info_loaded = True

    @staticmethod
    def _check_if_element_cond(data):
        data_element = data[0]
        if not isinstance(data_element, Indexable):
            raise TypeError(
                "The elements in the sequence are not "
                f"indexable: {type(data_element)}")
        if len(data_element) < 2:
            raise TypeError(
                "The elements in the sequence have less than 2 items. "
                f"Not enough elements to unpack: {len(data_element)}")


class NetworkDataset(LabeledDatasetBase[torch.Tensor, S_co], Dataset):
    """
    A Pytorch dataset that loads a pickle file storing a list of the outputs of torch.nn.Module.state_dict(), that is basically a Dict[str, Tensor]. This dataset loads that file, for each network it flattens the tensors into a single vector and stores a tuple (flattened vector, label). Stores exactly the first 'limit' elements of the list.
    """

    def __init__(self, file: str, label: S_co, limit: int = None):
        super().__init__()
        # the weights in a vector
        self.__load(file, label, limit)

    def __load(self, file_name, label: S_co, limit):
        networks = torch.load(file_name)

        dataset = []

        if limit is not None:
            if not isinstance(limit, int):
                raise ValueError("Limit must be an integer")
            if len(networks) < limit:
                raise ValueError(
                    "Limit is larger than the size of the dataset")

        for i, weights in enumerate(networks, start=1):
            concat_weights = torch.cat([w.flatten() for w in weights.values()])
            dataset.append(concat_weights)

            if i == limit:
                break

        self._dataset = torch.stack(dataset)
        self._labels = DummyIterable(label, length=len(dataset))


class LabeledDataset(LabeledDatasetBase[T_co, S_co], Dataset):
    """A simple dataset that supports labeled data."""

    def __init__(self,
                 dataset: IndexableIterable[T_co],
                 labels: IndexableIterable[S_co]):
        super().__init__()

        self._dataset = dataset
        self._labels = labels

    @classmethod
    def from_tuple_dataset(cls, dataset: IndexableIterable[Tuple[T_co, S_co]]):
        cls._check_if_element_cond(dataset)
        data_elements: List[T_co] = []
        labels: List[S_co] = []
        for x, y in dataset:
            data_elements.append(x)
            labels.append(y)

        data_instance = cls(data_elements, labels)

        return data_instance

    @classmethod
    def from_iterable(
            cls, datasets: Sequence[IndexableIterable[Tuple[T_co, S_co]]]):

        dataset: List[T_co] = []
        labels: List[S_co] = []
        for d in datasets:
            if not isinstance(d, LabeledDatasetLike):
                d = cls.from_tuple_dataset(d)

            dataset.extend(d.dataset)
            labels.extend(d.labels)

        return cls(dataset=dataset, labels=labels)


class Subset(Dataset, Generic[T_co]):
    def __init__(
            self,
            dataset: DatasetLike[T_co],
            indices: Sequence[int]):
        self._dataset = dataset
        self._indices = indices

    def __getitem__(self, idx: int):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)

    def __iter__(self):
        for ind in self.indices:
            yield self._dataset[ind]

    @property
    def dataset(self):
        return self._dataset.dataset

    @property
    def indices(self):
        return self._indices

    def apply_subset(self):
        return SimpleDataset(dataset=list(self))


class LabeledSubset(Dataset, Generic[T_co, S_co]):
    def __init__(self,
                 dataset: LabeledDatasetLike[T_co, S_co],
                 indices=Sequence[int]):
        self._dataset = dataset
        self._indices = indices

    def __getitem__(self, idx: int):
        return self._dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)

    def __iter__(self):
        for ind in self.indices:
            yield self._dataset[ind]

    @property
    def dataset(self):
        return self._dataset.dataset

    @property
    def labels(self):
        return self._dataset.labels

    @property
    def label_info(self):
        return self._dataset.label_info

    @property
    def indices(self):
        return self._indices

    def apply_subset(self):
        dataset: List[T_co] = []
        labels: List[S_co] = []
        for x, y in self:
            dataset.append(x)
            labels.append(y)
        return LabeledDataset(dataset=dataset, labels=labels)
