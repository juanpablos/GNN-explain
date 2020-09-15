import bisect
from typing import Generic, Iterator, List, Sequence, Tuple

import torch
from torch.utils.data import Dataset

from src.graphs.foc import Element
from src.typing import Indexable, IndexableIterable, S_co, T_co
import logging

logger = logging.getLogger(__name__)


class DummyIterable(Generic[S_co]):
    def __init__(self, value: S_co, length: int):
        self.value: T_co = value
        self.length: int = length

    def __len__(self):
        return self.length

    def __getitem__(self, index: int):
        return self.value

    def __iter__(self):
        for _ in range(self.length):
            yield self.value


class NoLabelDataset(Dataset, Generic[T_co]):
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


class GraphDataset(NoLabelDataset[T_co]):
    """A Pytorch dataset that takes a (infinite) generator and stores 'limit' elements of it."""

    def __init__(self, generator: Iterator[T_co], limit: int):
        super().__init__(dataset=[next(generator) for _ in range(limit)])


class NoLabelSubset(NoLabelDataset[T_co]):
    def __init__(
            self,
            dataset: NoLabelDataset[T_co],
            indices: Sequence[int]):
        super().__init__(dataset=dataset)
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
        # ?? change this to only compute this once?
        return self.apply_subset().dataset

    @property
    def indices(self):
        return self._indices

    def apply_subset(self):
        return NoLabelDataset(dataset=list(self))


class LabeledDataset(Dataset, Generic[T_co, S_co]):
    def __init__(
            self,
            dataset: IndexableIterable[T_co],
            labels: IndexableIterable[S_co],
            multilabel: bool):

        self._dataset: IndexableIterable[T_co] = dataset
        self._labels: IndexableIterable[S_co] = labels
        self._multilabel = multilabel
        self._unique_labels = set()
        self._process_labels()

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
    def multilabel(self):
        return self._multilabel

    @property
    def unique_labels(self):
        return self._unique_labels

    def _process_labels(self):
        if self.multilabel:
            for label in self.labels:
                self._unique_labels.update(label)
        else:
            self._unique_labels.update(self.labels)

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

    @classmethod
    def from_tuple_sequence(
            cls,
            dataset: IndexableIterable[Tuple[T_co, S_co]],
            multilabel: bool):
        cls._check_if_element_cond(dataset)

        data_elements: List[T_co] = []
        labels: List[S_co] = []
        for x, y in dataset:
            data_elements.append(x)
            labels.append(y)

        return cls(data_elements, labels, multilabel=multilabel)

    @classmethod
    def from_iterable(
            cls,
            datasets: Sequence[IndexableIterable[Tuple[T_co, S_co]]], multilabel: bool):

        dataset: List[T_co] = []
        labels: List[S_co] = []
        for d in datasets:
            if not isinstance(d, LabeledDataset):
                logger.debug("Getting dataset from tuples")
                d = cls.from_tuple_sequence(d, multilabel=multilabel)
            dataset.extend(d.dataset)
            labels.extend(d.labels)

        return cls(dataset=dataset, labels=labels, multilabel=multilabel)


class NetworkDataset(LabeledDataset[torch.Tensor, S_co]):
    """
    A Pytorch dataset that loads a pickle file storing a list of the outputs of torch.nn.Module.state_dict(), that is basically a Dict[str, Tensor]. This dataset loads that file, for each network it flattens the tensors into a single vector and stores a tuple (flattened vector, label). Stores exactly the first 'limit' elements of the list.
    """

    def __init__(
            self,
            file: str,
            label: S_co,
            formula: Element,
            limit: int = None,
            multilabel: bool = False,
            _legacy_load_without_batch: bool = False):

        dataset, labels = self.__load(
            file, label, limit, _legacy_load_without_batch)

        super().__init__(dataset=dataset, labels=labels, multilabel=multilabel)
        self.formula = formula

    def __load(self, filename, label: S_co, limit, no_batch):
        networks = torch.load(filename)

        dataset = []

        if limit is not None:
            if not isinstance(limit, int):
                raise ValueError("Limit must be an integer")
            if len(networks) < limit:
                raise ValueError(
                    "Limit is larger than the size of the dataset")

        for i, weights in enumerate(networks, start=1):

            # legacy
            if no_batch:
                weights = self.clean_state(weights)
            # /legacy

            concat_weights = torch.cat([w.flatten() for w in weights.values()])
            dataset.append(concat_weights)

            if i == limit:
                break

        return dataset, DummyIterable(label, length=len(dataset))

    @staticmethod
    def clean_state(model_dict):
        """Removes the weights associated with batchnorm"""
        return {k: v for k, v in model_dict.items() if "batch" not in k}


class LabeledSubset(LabeledDataset[T_co, S_co]):
    def __init__(self,
                 dataset: LabeledDataset[T_co, S_co],
                 indices: Sequence[int]):
        self._dataset: LabeledDataset[T_co, S_co] = dataset
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
        # ?? change this to only compute this once?
        return self.apply_subset().dataset

    @property
    def labels(self):
        # ?? change this to only compute this once?
        return self.apply_subset().labels

    @property
    def indices(self):
        return self._indices

    @property
    def multilabel(self):
        return self._dataset.multilabel

    @property
    def unique_labels(self):
        return self.apply_subset().unique_labels

    def apply_subset(self):
        dataset: List[T_co] = []
        labels: List[S_co] = []
        for x, y in self:
            dataset.append(x)
            labels.append(y)
        return LabeledDataset(
            dataset=dataset,
            labels=labels,
            multilabel=self._dataset.multilabel)


class NetworkDatasetCollectionWrapper(Dataset, Generic[S_co]):

    def __init__(self, datasets: Sequence[NetworkDataset[S_co]]):
        if len(datasets) < 1:
            raise ValueError("datasets cannot be an empty sequence")

        self.formulas: List[Element] = []
        for d in datasets:
            assert isinstance(
                d, NetworkDataset), "elements should be NetworkDatasets"
            self.formulas.append(d.formula)

        self.cumulative_sizes = self.cumsum(datasets)

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, index: int) -> Element:
        dataset_index = bisect.bisect_right(self.cumulative_sizes, index)
        return self.formulas[dataset_index]

    @staticmethod
    def cumsum(sequence: Sequence):
        seq: List[int] = []
        curr = 0
        for s in sequence:
            l = len(s)
            seq.append(curr + l)
            curr += l
        return seq
