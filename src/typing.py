from abc import abstractmethod
from typing import (
    Dict,
    Iterator,
    List,
    Literal,
    Mapping,
    Optional,
    Protocol,
    Tuple,
    TypedDict,
    TypeVar,
    Union,
    runtime_checkable
)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader as torch_loader
from torch_geometric.data import DataLoader as torch_geometric_loader

from src.data.formula_index import FormulaMapping
from src.data.formulas.filter import Filter
from src.data.formulas.labeler import LabelerApply

T = TypeVar("T")
S = TypeVar("S")
T_co = TypeVar("T_co", covariant=True)
S_co = TypeVar("S_co", covariant=True)
TNum = TypeVar("TNum", int, float)


class NetworkDataConfig(TypedDict):
    root: str
    model_hash: str
    selector: Filter
    labeler: LabelerApply
    formula_mapping: FormulaMapping
    test_selector: Filter


class StopFormat(TypedDict):
    operation: str
    stay: int
    conditions: Dict[str, Union[int, float]]


class MinModelConfig(TypedDict):
    num_layers: int
    input_dim: Optional[int]
    hidden_dim: int
    output_dim: Optional[int]
    hidden_layers: Optional[List[int]]
    use_batch_norm: bool


class GNNModelConfig(MinModelConfig):
    name: str
    aggregate_type: Literal["add", "mean", "max"]
    combine_type: Literal["identity", "linear", "mlp"]
    mlp_layers: int
    combine_layers: int
    task: Literal["node", "graph"]


class MetricHistory(Protocol):
    def __getitem__(self, key: str) -> TNum: ...
    def keys(self, select: bool = ...) -> Iterator[str]: ...

    def items(self,
              select: bool = ...) -> Iterator[Tuple[str,
                                                    List[TNum]]]: ...

    def log(self) -> str: ...
    def update(self, **kwargs: TNum) -> None: ...
    def get_history(self, key: str) -> List[TNum]: ...


class Trainer(Protocol):
    @abstractmethod
    def get_model(self, **kwargs) -> nn.Module: ...
    @abstractmethod
    def get_loss(self) -> nn.Module: ...

    @abstractmethod
    def get_optim(
        self,
        model: nn.Module,
        lr: float) -> optim.Optimizer: ...

    # @abstractmethod
    # def get_scheduler(
    #     self,
    #     optimizer: optim.Optimizer,
    #     step: int = ...) -> Optional[optim.lr_scheduler.StepLR]: ...

    @abstractmethod
    def train(self,
              model: nn.Module,
              training_data: Union[torch_loader, torch_geometric_loader],
              criterion: nn.Module,
              device: torch.device,
              optimizer: optim.Optimizer,
              **kwargs) -> float: ...

    @abstractmethod
    def evaluate(self,
                 model: nn.Module,
                 test_data: Union[torch_loader, torch_geometric_loader],
                 criterion: nn.Module,
                 device: torch.device,
                 using_train_data: bool,
                 **kwargs) -> Tuple[float, ...]: ...

    @abstractmethod
    def log(self) -> str: ...

    @abstractmethod
    def get_metric_logger(self) -> MetricHistory: ...


@runtime_checkable
class Indexable(Protocol[T_co]):
    def __getitem__(self, index: int) -> T_co: ...
    def __len__(self) -> int: ...


class IndexableIterable(Indexable[T_co], Protocol[T_co]):
    def __iter__(self) -> Iterator[T_co]: ...


class DatasetLike(IndexableIterable[T_co], Protocol[T_co]):
    @property
    def dataset(self) -> IndexableIterable[T_co]: ...


@runtime_checkable
class LabeledDatasetLike(Protocol[T_co, S_co]):
    def __getitem__(self, index: int) -> Tuple[T_co, S_co]: ...
    def __len__(self) -> int: ...
    def __iter__(self) -> Iterator[Tuple[T_co, S_co]]: ...
    @property
    def dataset(self) -> IndexableIterable[T_co]: ...
    @property
    def labels(self) -> IndexableIterable[S_co]: ...
    @property
    def label_info(self) -> Mapping[S_co, int]: ...
