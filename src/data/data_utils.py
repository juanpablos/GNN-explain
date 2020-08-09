from typing import Any


class DummyIterable:
    def __init__(self, value: Any, length: int):
        self.value = value
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, index: int):
        return self.value

    def __iter__(self):
        for _ in range(self.length):
            yield self.value
