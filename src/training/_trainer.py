from abc import ABC, abstractmethod
from typing import List, Literal, Tuple, TypeVar, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.training.utils import MetricLogger


class Trainer(ABC):
    loss: nn.Module
    model: nn.Module
    optim: torch.optim.Optimizer
    device: torch.device
    train_loader: DataLoader
    test_loader: DataLoader

    available_metrics: List[str]
    metric_logger: MetricLogger

    def __init__(self,
                 logging_variables: Union[Literal["all"],
                                          List[str]] = "all"):

        if logging_variables != "all" and not all(
                var in self.available_metrics for var in logging_variables):
            raise ValueError(
                "Encountered not supported metric. "
                f"Supported are: {self.available_metrics}")
        self.metric_logger = MetricLogger(logging_variables)

    def init_device(self, device: torch.device):
        self.device = device

    @abstractmethod
    def init_model(self, **kwargs) -> nn.Module: ...
    @abstractmethod
    def init_loss(self) -> nn.Module: ...

    @abstractmethod
    def init_optim(self, lr: float) -> torch.optim.Optimizer: ...

    @abstractmethod
    def init_dataloader(
        self,
        data,
        mode: Union[Literal["train"], Literal["test"]],
        *,
        batch_size: int,
        shuffle: bool,
        num_workers: int,
        **kwargs) -> DataLoader: ...

    @abstractmethod
    def train(self, **kwargs) -> float: ...

    @abstractmethod
    def evaluate(self,
                 use_train_data: bool,
                 **kwargs) -> Tuple[float, ...]: ...

    @abstractmethod
    def log(self) -> str: ...


TrainType = TypeVar("TrainType", bound=Trainer)


class TrainerBuilder:
    def __init__(self, trainer: TrainType):
        self.__trainer: TrainType = trainer

    def init_device(self, device):
        return self.__trainer.init_device(device=device)

    def init_model(self, **kwargs) -> nn.Module:
        if not hasattr(self.__trainer, "device"):
            raise ValueError("Must call `init_device` before `init_model`")
        return self.__trainer.init_model(**kwargs)

    def init_loss(self) -> nn.Module:
        return self.__trainer.init_loss()

    def init_optim(self, lr: float) -> torch.optim.Optimizer:
        if not hasattr(self.__trainer, "model"):
            raise ValueError("Should call `init_model` before `init_optim`")
        return self.__trainer.init_optim(lr=lr)

    def init_dataloader(
            self,
            data,
            mode: Union[Literal["train"], Literal["test"]],
            batch_size: int,
            shuffle: bool,
            num_workers: int,
            **kwargs) -> DataLoader:
        return self.__trainer.init_dataloader(
            data=data,
            mode=mode,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            **kwargs)

    def validate_trainer(self):
        variables = ["loss",
                     "model",
                     "optim",
                     "device",
                     "metric_logger",
                     "train_loader",
                     "test_loader"]

        for var in variables:
            if not hasattr(self.__trainer, var):
                raise ValueError(
                    f"Trainer is not completely initialized: {var} missing")

        return self.__trainer
