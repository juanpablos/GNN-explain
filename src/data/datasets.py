import bisect
import logging
from typing import Dict, Generic, Iterator, List, Sequence, Tuple

import torch
from torch.utils.data import Dataset

from src.graphs.foc import Element
from src.typing import Indexable, IndexableIterable, S_co, T_co

logger = logging.getLogger(__name__)


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
        return self.apply_subset().dataset

    @property
    def indices(self):
        return self._indices

    def apply_subset(self):
        return NoLabelDataset(dataset=list(self))


class BaseLabeledDataset(Generic[T_co, S_co]):
    def __init__(
            self,
            dataset: IndexableIterable[T_co],
            labels: IndexableIterable[S_co]):
        self._dataset: IndexableIterable[T_co] = dataset
        self._labels: IndexableIterable[S_co] = labels

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
            **kwargs):
        cls._check_if_element_cond(dataset)

        data_elements: List[T_co] = []
        labels: List[S_co] = []
        for x, y in dataset:
            data_elements.append(x)
            labels.append(y)

        return cls(data_elements, labels, **kwargs)

    @classmethod
    def from_iterable(
            cls,
            datasets: Sequence[IndexableIterable[Tuple[T_co, S_co]]],
            **kwargs):

        dataset: List[T_co] = []
        labels: List[S_co] = []
        for d in datasets:
            if not isinstance(d, LabeledDataset):
                logger.debug("Getting dataset from tuples")
                d = cls.from_tuple_sequence(d, **kwargs)
            dataset.extend(d.dataset)
            labels.extend(d.labels)

        return cls(dataset=dataset, labels=labels, **kwargs)


class LabeledDataset(BaseLabeledDataset[T_co, S_co], Dataset):
    def __init__(
            self,
            dataset: IndexableIterable[T_co],
            labels: IndexableIterable[S_co],
            multilabel: bool):
        super().__init__(dataset=dataset, labels=labels)

        self._multilabel = multilabel
        self._unique_labels = set()
        self._process_labels()

    @property
    def multilabel(self):
        return self._multilabel

    @property
    def unique_labels(self):
        return self._unique_labels

    def _process_labels(self):
        if self.multilabel:
            self._unique_labels.update(range(len(self.labels[0])))
        else:
            self._unique_labels.update(self.labels)

    @classmethod
    def from_tuple_sequence(
            cls,
            dataset: IndexableIterable[Tuple[T_co, S_co]],
            multilabel: bool):
        return super().from_tuple_sequence(dataset=dataset, multilabel=multilabel)

    @classmethod
    def from_iterable(
            cls,
            datasets: Sequence[IndexableIterable[Tuple[T_co, S_co]]],
            multilabel: bool):
        return super().from_iterable(datasets=datasets, multilabel=multilabel)


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
        return self.apply_subset().dataset

    @property
    def labels(self):
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


class DummyIterable(Generic[S_co]):
    def __init__(self, value: S_co, length: int):
        self.value: S_co = value
        self.length: int = length

    def __len__(self):
        return self.length

    def __getitem__(self, index: int):
        return self.value

    def __iter__(self):
        for _ in range(self.length):
            yield self.value


class NetworkDataset(Dataset, Generic[S_co]):
    """
    A Pytorch dataset that loads a pickle file storing a list of the outputs of torch.nn.Module.state_dict(), that is basically a Dict[str, Tensor]. This dataset loads that file, for each network it flattens the tensors into a single vector and stores a tuple (flattened vector, label). Stores exactly the first 'limit' elements of the list.
    """

    def __init__(
            self,
            label: S_co,
            formula: Element,
            file: str = "",
            limit: int = None,
            multilabel: bool = False,
            text: bool = False,
            vocabulary: Dict[str, int] = None,
            preloaded: IndexableIterable[torch.Tensor] = None,
            _legacy_load_without_batch: bool = False):

        if file == "" and preloaded is None:
            raise ValueError("Cannot have `file` and `preloaded` unset")

        if preloaded is None:
            dataset = self.__load(file, limit, _legacy_load_without_batch)
        else:
            dataset = preloaded

        self._dataset = dataset
        self._labels = DummyIterable(label, length=len(dataset))
        self._multilabel = multilabel
        self._formula = formula
        self._text = text
        self._vocabulary = vocabulary if vocabulary is not None else {}

    @classmethod
    def categorical(cls,
                    label: S_co,
                    formula: Element,
                    file: str = "",
                    limit: int = None,
                    multilabel: bool = False,
                    preloaded: IndexableIterable[torch.Tensor] = None,
                    _legacy_load_without_batch: bool = False):

        dataset = cls(
            label=label,
            formula=formula,
            file=file,
            limit=limit,
            multilabel=multilabel,
            text=False,
            vocabulary=None,
            preloaded=preloaded,
            _legacy_load_without_batch=_legacy_load_without_batch,
        )
        setattr(dataset, "is_categorical", True)
        return dataset

    @classmethod
    def text_sequence(cls,
                      label: S_co,
                      formula: Element,
                      file: str = "",
                      limit: int = None,
                      vocabulary: Dict[str, int] = None,
                      preloaded: IndexableIterable[torch.Tensor] = None,
                      _legacy_load_without_batch: bool = False):

        dataset = cls(
            label=label,
            formula=formula,
            file=file,
            limit=limit,
            multilabel=False,
            text=True,
            vocabulary=vocabulary,
            preloaded=preloaded,
            _legacy_load_without_batch=_legacy_load_without_batch,
        )
        setattr(dataset, "is_text", True)
        return dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index: int):
        return self.dataset[index], self.labels[index]

    def __iter__(self):
        for x, y in zip(self.dataset, self.labels):
            yield x, y

    @property
    def dataset(self):
        return self._dataset

    @property
    def labels(self):
        return self._labels

    @property
    def multilabel(self):
        return self._multilabel

    @property
    def formula(self):
        return self._formula

    @property
    def text(self):
        return self._text

    @property
    def vocab(self):
        return self._vocabulary

    @property
    def inverse_vocab(self):
        return {v: k for k, v in self.vocab.items()}

    def __load(self, filename, limit, no_batch):
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

        return dataset

    @staticmethod
    def clean_state(model_dict):
        """Removes the weights associated with batchnorm"""
        return {k: v for k, v in model_dict.items() if "batch" not in k}


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


class AggregatedNetworkDataset:
    def __init__(self, file_path: str):
        """
        format:
        {
            formula_hash: {
                file: file_path,
                data: tensor_data
            }
        }
        """
        logger.debug("Loading formulas")
        self.formulas = torch.load(file_path)

    def __getitem__(self, formula: str) -> torch.Tensor:
        return self.formulas[formula]["data"]

    def available_formulas(self) -> Dict[str, str]:
        return {k: v["file"] for k, v in self.formulas.items()}


class TextSequenceDataset(
        BaseLabeledDataset[torch.Tensor, List[int]], Dataset):
    def __init__(
            self,
            dataset: IndexableIterable[T_co],
            labels: IndexableIterable[List[int]],
            vocabulary: Dict[str, int]):

        self._dataset: IndexableIterable[T_co] = dataset
        self._labels: IndexableIterable[List[int]] = labels

        self._vocabulary = vocabulary
        self._inverse_vocabulary = {v: k for k, v in vocabulary.items()}

    def __getitem__(self, index: int):
        return self.dataset[index], self.labels[index]

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        for x, y in zip(self.dataset, self.labels):
            yield x, y

    @property
    def dataset(self):
        return self._dataset

    @property
    def labels(self):
        return self._labels

    @property
    def vocabulary(self):
        return self._vocabulary

    @property
    def inverse_vocabulary(self):
        return self._inverse_vocabulary

    @classmethod
    def from_tuple_sequence(
            cls,
            dataset: IndexableIterable[Tuple[T_co, S_co]],
            vocabulary: Dict[str, int]):
        return super().from_tuple_sequence(dataset=dataset, vocabulary=vocabulary)

    @classmethod
    def from_iterable(
            cls,
            datasets: Sequence[IndexableIterable[Tuple[T_co, S_co]]], vocabulary: Dict[str, int]):
        return super().from_iterable(datasets=datasets, vocabulary=vocabulary)
