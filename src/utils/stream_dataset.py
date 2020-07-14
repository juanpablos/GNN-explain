from torch.utils.data import IterableDataset


class LimitedStreamDataset(IterableDataset):
    def __init__(self, data, limit: int):
        self.data = data
        self.limit = limit

    def __iter__(self):
        for _ in range(self.limit):
            yield next(self.data)
