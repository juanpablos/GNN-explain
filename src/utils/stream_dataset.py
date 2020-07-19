import numpy as np
from torch.utils.data import IterableDataset


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
