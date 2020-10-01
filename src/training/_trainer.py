from abc import ABC, abstractmethod
from typing import List, Literal, Tuple, Union

import torch
import torch.nn as nn

from src.training.utils import MetricLogger


class Trainer(ABC):
    device: torch.device

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

    def set_device(self, device: torch.device):
        self.device = device

    @abstractmethod
    def get_models(self) -> List[nn.Module]: ...

    @abstractmethod
    def init_trainer(self, **optim_params) -> None: ...

    @abstractmethod
    def train(self, **kwargs) -> float: ...

    @abstractmethod
    def evaluate(self,
                 use_train_data: bool,
                 **kwargs) -> Tuple[float, ...]: ...

    @abstractmethod
    def log(self) -> str: ...
