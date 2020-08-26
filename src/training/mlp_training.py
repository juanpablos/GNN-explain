import logging
from typing import Any, Dict, List, Literal, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from src.gnn import MLP
from src.training.utils import MetricLogger
from src.typing import Trainer

logger = logging.getLogger(__name__)


class Metric:
    def __init__(self, average: str = "macro"):
        if average not in ["binary", "micro", "macro"]:
            raise ValueError(
                "Argument `average` must be one of `binary`, `micro`, `macro`")
        self.y_true = []
        self.y_pred = []
        self.average = average

    def __call__(self, y_true: torch.Tensor, y_pred: torch.Tensor):
        self.y_true.extend(y_true.tolist())
        self.y_pred.extend(y_pred.tolist())

    def precision_recall_fscore(self) -> Dict[str, Any]:
        precision, recall, f1score, _ = precision_recall_fscore_support(
            self.y_true, self.y_pred, average=self.average, beta=1.0)

        return {"precision": precision, "recall": recall, "f1": f1score}

    def accuracy(self):
        return {"acc": accuracy_score(self.y_true, self.y_pred)}

    def clear(self):
        self.y_true.clear()
        self.y_pred.clear()


class Training(Trainer):
    available_metrics = [
        "all",
        "train_loss",
        "test_loss",
        "train_precision",
        "train_recall",
        "train_f1",
        "train_acc",
        "test_precision",
        "test_recall",
        "test_f1",
        "test_acc"
    ]

    def __init__(self,
                 n_classes: int = 2,
                 metrics_average: str = "macro",
                 logging_variables: Union[Literal["all"],
                                          List[str],
                                          None] = "all"):

        if logging_variables is None:
            logging_variables = []

        if logging_variables != "all" and not all(
                var in self.available_metrics for var in logging_variables):
            raise ValueError(
                "Encountered not supported metric. "
                f"Supported are: {self.available_metrics}")
        self.metric_logger = MetricLogger(logging_variables)

        self.n_classes = n_classes
        self.metrics = Metric(average=metrics_average)

    def get_metric_logger(self):
        return self.metric_logger

    def transform_y(self, y):
        if self.n_classes == 2:
            return F.one_hot(y, 2).float()
        else:
            return y

    def activation(self, output):
        if self.n_classes == 2:
            return torch.sigmoid(output)
        else:
            return torch.log_softmax(output, dim=1)

    def get_loss(self):
        if self.n_classes > 2:
            logger.debug("Using CrossEntropyLoss")
            return nn.CrossEntropyLoss(reduction="mean")
        elif self.n_classes == 2:
            logger.debug("Using BCEWithLogitsLoss")
            return nn.BCEWithLogitsLoss(reduction="mean")
        else:
            raise ValueError("Number of classes cannot be less than 2")

    def get_model(self,
                  *,
                  num_layers: int,
                  input_dim: int,
                  hidden_dim: int,
                  output_dim: int,
                  use_batch_norm: bool = True,
                  hidden_layers: List[int] = None,
                  **kwargs):
        return MLP(num_layers=num_layers,
                   input_dim=input_dim,
                   hidden_dim=hidden_dim,
                   output_dim=output_dim,
                   use_batch_norm=use_batch_norm,
                   hidden_layers=hidden_layers,
                   **kwargs)

    def get_optim(self, model, lr):
        # return optim.Rprop(model.parameters(), lr=lr)
        # return optim.RMSprop(model.parameters(), lr=lr)
        # return optim.SGD(model.parameters(), lr=lr)
        # return optim.AdamW(model.parameters(), lr=lr)
        return optim.Adam(model.parameters(), lr=lr)

    def get_scheduler(self, **kwargs):
        pass

    def train(self,
              model,
              training_data,
              criterion,
              device,
              optimizer,
              **kwargs):

        #!########
        model.train()
        #!########

        accum_loss = []

        for x, y in training_data:
            x = x.to(device)
            y = y.to(device)

            output = model(x)
            loss = criterion(output, self.transform_y(y))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            accum_loss.append(loss.detach().cpu().numpy())

        average_loss = np.mean(accum_loss)

        self.metric_logger.update(train_loss=average_loss)

        return average_loss

    def evaluate(self,
                 model,
                 test_data,
                 criterion,
                 device,
                 using_train_data,
                 **kwargs):

        #!########
        model.eval()
        self.metrics.clear()
        #!########

        accum_loss = []

        for x, y in test_data:
            x = x.to(device)
            y = y.to(device)

            with torch.no_grad():
                output = model(x)

            loss = criterion(output, self.transform_y(y))
            accum_loss.append(loss.detach().cpu().numpy())

            output = self.activation(output)
            _, y_pred = output.max(dim=1)

            self.metrics(y, y_pred)

        average_loss = np.mean(accum_loss)
        metrics = self.metrics.precision_recall_fscore()
        acc = self.metrics.accuracy()["acc"]

        if using_train_data:
            metrics = {
                f"train_{name}": value for name,
                value in metrics.items()}

            self.metric_logger.update(
                train_acc=acc,
                **metrics)
        else:
            metrics = {
                f"test_{name}": value for name,
                value in metrics.items()}

            self.metric_logger.update(
                test_loss=average_loss,
                test_acc=acc,
                **metrics)

        return average_loss

    def log(self):
        return self.metric_logger.log()
