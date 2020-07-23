
from abc import abstractmethod
from typing import (Any, Dict, Generic, Iterator, List, Literal, Optional,
                    Protocol, Tuple, TypedDict, TypeVar, Union)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader as torch_loader
from torch_geometric.data import DataLoader as torch_geometric_loader

T = TypeVar("T")


class FormulaHashInfo(TypedDict):
    limit: Optional[int]
    label: Any


FormulaHash = Dict[str, FormulaHashInfo]


class StopFormat(TypedDict):
    operation: str
    stay: int
    conditions: Dict[str, Union[int, float]]


class MinModelConfig(TypedDict):
    num_layers: int
    input_dim: Optional[int]
    hidden_dim: int
    output_dim: int
    hidden_layers: Optional[List[int]]


class GNNModelConfig(MinModelConfig):
    name: str
    aggregate_type: Literal["add", "mean", "max"]
    combine_type: Literal["identity", "linear", "mlp"]
    mlp_layers: int
    combine_layers: int
    task: Literal["node", "graph"]


class NetworkDataConfig(TypedDict):
    root: str
    model_hash: str
    formula_hashes: FormulaHash


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

    @abstractmethod
    def get_scheduler(
        self,
        optimizer: optim.Optimizer,
        step: int = ...) -> Optional[optim.lr_scheduler.StepLR]: ...

    @abstractmethod
    def train(self,
              model: nn.Module,
              training_data: Union[torch_loader, torch_geometric_loader],
              criterion: nn.Module,
              device: torch.device,
              optimizer: optim.Optimizer,
              collector: Dict[str, Any],
              **kwargs) -> float: ...

    @abstractmethod
    def evaluate(self,
                 model: nn.Module,
                 test_data: Union[torch_loader, torch_geometric_loader],
                 criterion: nn.Module,
                 device: torch.device,
                 collector: Dict[str, Any],
                 **kwargs) -> Tuple[float, ...]: ...

    @abstractmethod
    def log(self, info: Dict[str, Any]) -> str: ...


class DatasetType(Dataset, Generic[T]):
    @abstractmethod
    def __getitem__(self, idx: int) -> T: ...
    @abstractmethod
    def __len__(self) -> int: ...
    @abstractmethod
    def __iter__(self) -> Iterator[T]: ...
