from itertools import chain

import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset


class LimitedStreamDataset(IterableDataset):
    def __init__(self, data, limit: int, store: bool = False):
        self.data = data
        self.limit = limit
        self.store = store
        self.dataset = []

    def __iter__(self):
        if self.dataset:
            np.random.shuffle(self.dataset)
            yield from self.dataset
        else:
            for _ in range(self.limit):
                datapoint = next(self.data)

                if self.store:
                    self.dataset.append(datapoint)

                yield datapoint


class RandomGraphDataset(Dataset):
    def __init__(self, generator, limit: int):
        self.generator = generator
        self.dataset = [next(self.generator) for _ in range(limit)]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


class NetworkDataset(Dataset):
    def __init__(self, file, label, limit: int = None):
        # the weights in a vector
        self.dataset = []
        self.__load(file, limit, label)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # REV: improve memory by only storing the label once
        return self.dataset[idx]

    def __load(self, file_name, limit, label):
        networks = torch.load(file_name)

        if limit is not None:
            assert isinstance(limit, int), "Limit is not an integer"
            assert len(
                networks) >= limit, "Limit is larger than the size of the dataset"

        for i, weights in enumerate(networks, start=1):

            self.dataset.append(
                (torch.cat([w.flatten() for w in weights.values()]),
                 label)
            )

            if i == limit:
                break


class MergedDataset(Dataset):
    def __init__(self, datasets):
        self.data = list(chain.from_iterable(datasets))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
