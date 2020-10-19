from __future__ import annotations

import bisect
import logging
import warnings
from abc import ABC
from typing import (
    Dict,
    Generic,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeVar
)

import torch
from torch.utils.data import Dataset

from src.data.vocabulary import Vocabulary
from src.graphs.foc import Element
from src.typing import DatasetLike, Indexable, IndexableIterable, S, S_co, T_co

logger = logging.getLogger(__name__)

LabeledDatasetType = TypeVar("LabeledDatasetType", bound="BaseLabeledDataset")


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
        # FIX: here until
        # https://github.com/microsoft/pylance-release/issues/409
        dataset = [next(generator) for _ in range(limit)]
        super().__init__(dataset=dataset)


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


class BaseLabeledDataset(ABC, Generic[T_co, S_co]):
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
            cls: Type[LabeledDatasetType],
            dataset: IndexableIterable[Tuple[T_co, S_co]],
            **kwargs) -> LabeledDatasetType:
        cls._check_if_element_cond(dataset)

        data_elements: List[T_co] = []
        labels: List[S_co] = []
        for x, y in dataset:
            data_elements.append(x)
            labels.append(y)

        return cls(data_elements, labels, **kwargs)

    @classmethod
    def from_iterable(
            cls: Type[LabeledDatasetType],
            datasets: Sequence[IndexableIterable[Tuple[T_co, S_co]]],
            **kwargs) -> LabeledDatasetType:

        dataset: List[T_co] = []
        labels: List[S_co] = []
        for d in datasets:
            if not isinstance(d, DatasetLike):
                logger.debug("Getting dataset from tuples")
                d = cls.from_tuple_sequence(d, **kwargs)
            dataset.extend(d.dataset)
            labels.extend(d.labels)

        return cls(dataset=dataset, labels=labels, **kwargs)

    @classmethod
    def from_subset(cls: Type[LabeledDatasetType],
                    subset: LabeledSubset[T_co,
                                          S_co]) -> LabeledDatasetType:
        raise NotImplementedError(
            f"from_iterable is not implemented for {cls}")


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
        return super(LabeledDataset, cls).from_tuple_sequence(
            dataset=dataset,
            multilabel=multilabel)

    @classmethod
    def from_iterable(
            cls,
            datasets: Sequence[IndexableIterable[Tuple[T_co, S_co]]],
            multilabel: bool):
        return super(LabeledDataset, cls).from_iterable(
            datasets=datasets,
            multilabel=multilabel)

    @classmethod
    def from_subset(cls, subset: LabeledSubset[T_co, S_co]):
        if isinstance(subset._dataset, LabeledDataset):
            dataset: List[T_co] = []
            labels: List[S_co] = []
            for x, y in subset:
                dataset.append(x)
                labels.append(y)

            return cls(
                dataset=dataset,
                labels=labels,
                multilabel=subset._dataset.multilabel)
        else:
            raise TypeError("The subset is not a LabeledDatset subset")


class LabeledSubset(BaseLabeledDataset[T_co, S_co], Dataset):
    def __init__(self,
                 dataset: BaseLabeledDataset[T_co, S_co],
                 indices: Sequence[int]):
        self._dataset: BaseLabeledDataset[T_co, S_co] = dataset
        self._indices = indices

    def __getitem__(self, idx: int):
        return self._dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)

    def __iter__(self):
        for ind in self.indices:
            yield self._dataset[ind]

    @property
    def indices(self):
        return self._indices

    def __getattr__(self, name: str):
        if name in ["dataset", "labels"]:
            warnings.warn(
                "Creating concrete dataset from subset. "
                "To avoid computation overhead call `apply_subset` "
                "and store that object",
                stacklevel=3)
            return getattr(self.apply_subset(), name)
        elif name in ["multilabel"]:
            return getattr(self._dataset, name)

        raise AttributeError(
            "Do not call Concrete Dataset methods "
            f"on {self.__class__}. Call `apply_subset` and then the method.")

    def apply_subset(self):
        return self._dataset.from_subset(subset=self)

    @classmethod
    def from_tuple_sequence(cls, *args, **kwargs):
        raise NotImplementedError(
            f"from_tuple_sequence is not implemented for {cls}")

    @classmethod
    def from_iterable(cls, *args, **kwargs):
        raise NotImplementedError(
            f"from_iterable is not implemented for {cls}")


class DummyIterable(Generic[S]):
    def __init__(self, value: S, length: int):
        self.value: S = value
        self.length: int = length

    def __len__(self):
        return self.length

    def __getitem__(self, index: int):
        return self.value

    def __iter__(self):
        for _ in range(self.length):
            yield self.value


class NetworkDataset(Dataset, Generic[S]):
    """
    A Pytorch dataset that loads a pickle file storing a list of the outputs of torch.nn.Module.state_dict(), that is basically a Dict[str, Tensor]. This dataset loads that file, for each network it flattens the tensors into a single vector and stores a tuple (flattened vector, label). Stores exactly the first 'limit' elements of the list.
    """

    def __init__(
            self,
            label: S,
            formula: Element,
            file: str = "",
            limit: int = None,
            multilabel: bool = False,
            text: bool = False,
            vocabulary: Vocabulary = None,
            preloaded: IndexableIterable[torch.Tensor] = None,
            _legacy_load_without_batch: bool = False):

        if file == "" and preloaded is None:
            raise ValueError("Cannot have `file` and `preloaded` unset")

        if preloaded is None:
            dataset = self.__load(file, limit, _legacy_load_without_batch)
        else:
            dataset = preloaded

        self._dataset: IndexableIterable[torch.Tensor] = dataset
        self._labels: IndexableIterable[S] = DummyIterable(
            label, length=len(dataset))
        self._multilabel = multilabel
        self._formula = formula
        self._text = text
        self._vocabulary = vocabulary

    @classmethod
    def categorical(cls,
                    label: S,
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
                      label: S,
                      formula: Element,
                      file: str = "",
                      limit: int = None,
                      vocabulary: Vocabulary = None,
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
    def vocabulary(self):
        return self._vocabulary

    def __load(self, filename: str, limit: Optional[int], no_batch: bool):
        networks = torch.load(filename)

        dataset: List[torch.Tensor] = []

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


class NetworkDatasetCollectionWrapper(Dataset):
    def __init__(self, datasets: Sequence[NetworkDataset]):
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
        BaseLabeledDataset[T_co, torch.Tensor], Dataset):
    def __init__(
            self,
            dataset: IndexableIterable[T_co],
            labels: IndexableIterable[torch.Tensor],
            vocabulary: Vocabulary):

        self._dataset: IndexableIterable[T_co] = dataset
        self._labels: IndexableIterable[torch.Tensor] = labels

        self._vocabulary = vocabulary

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

    @classmethod
    def from_tuple_sequence(
            cls,
            dataset: IndexableIterable[Tuple[T_co, torch.Tensor]],
            vocabulary: Vocabulary):
        return super(TextSequenceDataset, cls).from_tuple_sequence(
            dataset=dataset, vocabulary=vocabulary)

    @classmethod
    def from_iterable(cls,
                      datasets: Sequence[IndexableIterable[Tuple[
                          T_co,
                          torch.Tensor]]],
                      vocabulary: Vocabulary):
        return super(TextSequenceDataset, cls).from_iterable(
            datasets=datasets, vocabulary=vocabulary)

    @classmethod
    def from_subset(cls, subset: LabeledSubset[T_co, torch.Tensor]):
        if isinstance(subset._dataset, TextSequenceDataset):
            dataset: List[T_co] = []
            labels: List[torch.Tensor] = []
            for x, y in subset:
                dataset.append(x)
                labels.append(y)

            return cls(
                dataset=dataset,
                labels=labels,
                vocabulary=subset._dataset.vocabulary)
        else:
            raise TypeError("The subset is not a TextSequenceDataset subset")
