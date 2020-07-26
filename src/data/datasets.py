from itertools import chain
from typing import Any, Iterable, Iterator, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import IterableDataset

from src.typing import DatasetType, T_co


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


class RandomGraphDataset(DatasetType[T_co]):
    """A Pytorch dataset that takes a (infinite) generator and stores 'limit' elements of it.
    """

    def __init__(self, generator: Iterator[T_co], limit: int):
        self.dataset = [next(generator) for _ in range(limit)]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int):
        return self.dataset[idx]

    def __iter__(self):
        return iter(self.dataset)


class NetworkDataset(DatasetType[Tuple[torch.Tensor, Any]]):
    """A Pytorch dataset that loads a pickle file storing a list of the outputs of torch.nn.Module.state_dict(), that is basically a Dict[str, Tensor]. This dataset loads that file, for each network it flattens the tensors into a single vector and stores a tuple (flattened vector, label). Stores exactly the first 'limit' elements of the list.
    """

    def __init__(self, file: str, label: Any, limit: int = None):
        # the weights in a vector
        self.dataset: List[Tuple[torch.Tensor, Any]] = []
        self.__load(file, limit, label)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int):
        # ??: improve memory by only storing the label once
        return self.dataset[idx]

    def __iter__(self):
        return iter(self.dataset)

    def __load(self, file_name, limit, label):
        networks = torch.load(file_name)

        if limit is not None:
            if not isinstance(limit, int):
                raise ValueError("Limit must be an integer")
            if len(networks) < limit:
                raise ValueError(
                    "Limit is larger than the size of the dataset")

        for i, weights in enumerate(networks, start=1):

            self.dataset.append(
                (torch.cat([w.flatten() for w in weights.values()]),
                 label)
            )

            if i == limit:
                break


class MergedDataset(DatasetType[T_co]):
    """A Pytorch dataset that merges a sequence of iterables, or other Datasets physically.
    """

    def __init__(self, datasets: Sequence[Iterable[T_co]]):
        self.dataset = list(chain.from_iterable(datasets))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int):
        return self.dataset[idx]

    def __iter__(self):
        return iter(self.dataset)


class Subset(DatasetType[T_co]):

    def __init__(self, dataset: DatasetType[T_co], indices: Sequence[int]):
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx: int):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)

    def __iter__(self):
        return iter(self.dataset)
