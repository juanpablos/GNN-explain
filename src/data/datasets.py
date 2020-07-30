import warnings
from abc import ABC
from collections import defaultdict
from itertools import chain
from typing import (
    Dict, Generic, Hashable, Iterator, List, Mapping, Sequence, Tuple)

import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset

from src.typing import DatasetLike, Indexable, IndexableIterable, T_co


class DatasetBase(ABC, Generic[T_co]):
    """
    Base class for the datasets used. Defines basic functionality to be usable and label information support.
    """

    def __init__(self, labeled: bool = False):
        self._labeled: bool = labeled
        self._label_info: Dict[Hashable, int] = defaultdict(int)
        self._label_info_loaded: bool = False
        # only declaration
        self._dataset: IndexableIterable[T_co]

    def __getitem__(self, index: int) -> T_co:
        return self.dataset[index]

    def __len__(self) -> int:
        return len(self.dataset)

    def __iter__(self) -> Iterator[T_co]:
        return iter(self.dataset)

    @property
    def dataset(self) -> IndexableIterable[T_co]:
        return self._dataset

    @property
    def label_info(self) -> Mapping[Hashable, int]:
        self._check()
        return dict(self._label_info)

    @property
    def labeled(self) -> bool:
        return self._labeled

    def _check(self):
        if self.labeled:
            if not self._label_info_loaded:
                for el in self:
                    self._label_info[el[1]] += 1
                self._label_info_loaded = True
        else:
            warnings.warn(
                "The argument `labeled` was not set, this method is not supported for `labeled=False`",
                RuntimeWarning)

    @staticmethod
    def _check_if_element_cond(element):
        data_element = element[0]
        if not isinstance(data_element, Indexable):
            raise TypeError(
                f"Labeled is passed but the sequence of datasets do not have indexable elements: {type(data_element)}")
        if len(data_element) < 2:
            raise TypeError(
                f"Labeled is passed but the sequence of datasets do not have indexable elements with enough elements to unpack: {len(data_element)}")
        if not isinstance(data_element[1], Hashable):
            raise TypeError(
                f"The labels are not hashable: {type(data_element[1])}")


class LimitedStreamDataset(IterableDataset):
    def __init__(self, data, limit: int, store: bool = False, seed=None):
        self.data = data
        self.limit = limit
        self.store = store
        self.dataset = []
        self.rand = np.random.default_rng(seed)

    def __iter__(self):
        if self.dataset:
            self.rand.shuffle(self.dataset)
            yield from self.dataset
        else:
            for _ in range(self.limit):
                datapoint = next(self.data)

                if self.store:
                    self.dataset.append(datapoint)

                yield datapoint


class RandomGraphDataset(DatasetBase[T_co], Dataset):
    """A Pytorch dataset that takes a (infinite) generator and stores 'limit' elements of it.
    """

    def __init__(self, generator: Iterator[T_co], limit: int):
        super().__init__(labeled=False)
        self._dataset = [next(generator) for _ in range(limit)]


class NetworkDataset(DatasetBase[Tuple[torch.Tensor, Hashable]], Dataset):
    """A Pytorch dataset that loads a pickle file storing a list of the outputs of torch.nn.Module.state_dict(), that is basically a Dict[str, Tensor]. This dataset loads that file, for each network it flattens the tensors into a single vector and stores a tuple (flattened vector, label). Stores exactly the first 'limit' elements of the list.
    """

    def __init__(self, file: str, label: Hashable, limit: int = None):
        super().__init__(labeled=True)
        # the weights in a vector
        self._dataset: List[Tuple[torch.Tensor, Hashable]] = []
        self.__load(file, label, limit)

    def __load(self, file_name, label, limit):
        networks = torch.load(file_name)

        if limit is not None:
            if not isinstance(limit, int):
                raise ValueError("Limit must be an integer")
            if len(networks) < limit:
                raise ValueError(
                    "Limit is larger than the size of the dataset")

        for i, weights in enumerate(networks, start=1):
            concat_weights = torch.cat([w.flatten() for w in weights.values()])
            self._dataset.append((concat_weights, label))

            if i == limit:
                break


class SingleDataset(DatasetBase[T_co], Dataset):
    """A simple dataset that supports labeled data.
    """

    def __init__(self,
                 dataset: IndexableIterable[T_co],
                 labeled: bool = False):
        super().__init__(labeled=labeled)

        if labeled:
            self._check_if_element_cond(dataset)

        self._dataset = dataset

    @classmethod
    def from_iterable(cls,
                      datasets: Sequence[IndexableIterable[T_co]],
                      labeled: bool = False):

        if labeled:
            _el = datasets[0]
            cls._check_if_element_cond(_el)
        data = list(chain.from_iterable(datasets))
        dataset = cls(dataset=data, labeled=labeled)

        return dataset


class Subset(DatasetBase[T_co], Dataset):

    def __init__(self, dataset: DatasetLike[T_co], indices: Sequence[int]):
        super().__init__(labeled=dataset.labeled)
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
    def indices(self):
        return self._indices
