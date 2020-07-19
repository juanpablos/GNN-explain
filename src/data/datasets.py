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
    def __init__(self, file, label):
        # the weights in a vector
        self.dataset = []
        # TODO: the pathm
        self.__load(file)

        self.label = label
        pass

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx], self.label

    def __load(self, file_name):
        networks = torch.load(file_name)
        for weights in networks:
            self.dataset.append(
                torch.cat([w.flatten() for w in weights.values()]))
