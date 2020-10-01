import logging
from typing import Any, Dict, List, Literal, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import (
    accuracy_score,
    hamming_loss,
    jaccard_score,
    precision_recall_fscore_support
)
from torch.utils.data import DataLoader

from src.models import MLP

from . import Trainer

logger = logging.getLogger(__name__)


class Metric:
    def __init__(self, average: str = "macro", multilabel: bool = False):
        if average not in ["binary", "micro", "macro"]:
            raise ValueError(
                "Argument `average` must be one of `binary`, `micro`, `macro`")
        self.y_true: List[torch.Tensor] = []
        self.y_pred: List[torch.Tensor] = []
        self.average = average
        self.multilabel = multilabel
        self.converted = False

        self.use_y_true = np.array([])
        self.use_y_pred = np.array([])

    def __call__(self, y_true: torch.Tensor, y_pred: torch.Tensor):
        if not self.converted:
            self.y_true.append(y_true.detach())
            self.y_pred.append(y_pred.detach())
        else:
            raise RuntimeError(
                "Cannot add more test samples because already numpy converted")

    def clear(self):
        self.y_true.clear()
        self.y_pred.clear()
        self.use_y_true = np.array([])
        self.use_y_pred = np.array([])
        self.converted = False

    def convert(self):
        if not self.converted:
            self.use_y_true = torch.cat(self.y_true, dim=0).cpu().numpy()
            self.use_y_pred = torch.cat(self.y_pred, dim=0).cpu().numpy()
            self.converted = True

    def precision_recall_fscore(self) -> Dict[str, Any]:
        self.convert()
        precision, recall, f1score, _ = precision_recall_fscore_support(
            self.use_y_true, self.use_y_pred, average=self.average, beta=1.0)

        return {"precision": precision, "recall": recall, "f1": f1score}

    def accuracy(self):
        self.convert()
        return {"acc": accuracy_score(self.use_y_true, self.use_y_pred)}

    def multilabel_metrics(self):
        self.convert()
        return {
            "jaccard": jaccard_score(
                self.use_y_true,
                self.use_y_pred,
                average=self.average),
            "hamming": hamming_loss(self.use_y_true, self.use_y_pred)
        }

    def get_all(self):
        self.convert()
        res = {**self.precision_recall_fscore(), **self.accuracy()}

        if self.multilabel:
            res.update(self.multilabel_metrics())

        return res

    def report(self):
        self.convert()
        metrics = {}

        precision_avg, recall_avg, f1score_avg, _ = \
            precision_recall_fscore_support(
                self.use_y_true, self.use_y_pred, average=self.average, beta=1.0)
        precision_single, recall_single, f1score_single, _ = \
            precision_recall_fscore_support(
                self.use_y_true, self.use_y_pred, average=None, beta=1.0)

        precision = {
            "average": precision_avg,
            "single": precision_single
        }
        recall = {
            "average": recall_avg,
            "single": recall_single
        }
        f1 = {
            "average": f1score_avg,
            "single": f1score_single
        }

        acc = {"total": accuracy_score(self.use_y_true, self.use_y_pred)}

        metrics["precision"] = precision
        metrics["recall"] = recall
        metrics["f1"] = f1
        metrics["acc"] = acc

        if self.multilabel:
            jaccard_avg = jaccard_score(
                self.use_y_true, self.use_y_pred, average=self.average)
            jaccard_single = jaccard_score(
                self.use_y_true, self.use_y_pred, average=None)

            jaccard = {
                "average": jaccard_avg,
                "single": jaccard_single
            }
            hamming = {"total": hamming_loss(self.use_y_true, self.use_y_pred)}

            metrics["jaccard"] = jaccard
            metrics["hamming"] = hamming

        return metrics


class MLPTrainer(Trainer):
    loss: nn.Module
    model: MLP
    optim: torch.optim.Optimizer
    train_loader: DataLoader
    test_loader: DataLoader

    available_metrics = [
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
    multilabel_metrics = [
        "train_jaccard",
        "test_jaccard",
        "train_hamming",
        "test_hamming"
    ]

    def __init__(self,
                 logging_variables: Union[Literal["all"], List[str]] = "all",
                 n_classes: int = 2,
                 metrics_average: str = "macro",
                 multilabel: bool = False):

        if multilabel:
            self.available_metrics.extend(self.multilabel_metrics)

        super().__init__(logging_variables=logging_variables)
        self.n_classes = n_classes
        self.multilabel = multilabel
        self.metrics = Metric(average=metrics_average, multilabel=multilabel)

    def transform_y(self, y):
        if self.n_classes == 2 and not self.multilabel:
            return F.one_hot(y, 2).float()
        else:
            return y

    def activation(self, output):
        if self.n_classes == 2 or self.multilabel:
            return torch.sigmoid(output)
        else:
            return torch.log_softmax(output, dim=1)

    def inference(self, output):
        if self.multilabel:
            return (output >= 0.5).int()
        else:
            _, y_pred = output.max(dim=1)
            return y_pred

    def init_model(self,
                   *,
                   num_layers: int,
                   input_dim: int,
                   hidden_dim: int,
                   output_dim: int,
                   use_batch_norm: bool = True,
                   hidden_layers: List[int] = None,
                   **kwargs):
        self.model = MLP(num_layers=num_layers,
                         input_dim=input_dim,
                         hidden_dim=hidden_dim,
                         output_dim=output_dim,
                         use_batch_norm=use_batch_norm,
                         hidden_layers=hidden_layers,
                         **kwargs)

        return self.model

    def init_trainer(self, **optim_params):
        self.init_loss()
        self.model = self.model.to(self.device)
        self.init_optim(**optim_params)

    def init_loss(self):
        if self.n_classes > 2 and not self.multilabel:
            logger.debug("Using CrossEntropyLoss")
            self.loss = nn.CrossEntropyLoss(reduction="mean")
        elif self.n_classes == 2 or self.multilabel:
            logger.debug("Using BCEWithLogitsLoss")
            self.loss = nn.BCEWithLogitsLoss(reduction="mean")
        else:
            raise ValueError("Number of classes cannot be less than 2")

        return self.loss

    def init_optim(self, lr):
        self.optim = optim.Adam(self.model.parameters(), lr=lr)
        return self.optim

    def init_dataloader(self,
                        data,
                        mode: Union[Literal["train"], Literal["test"]],
                        **kwargs):

        if mode not in ["train", "test"]:
            raise ValueError("Supported modes are only `train` and `test`")

        loader = DataLoader(data, **kwargs)
        if mode == "train":
            self.train_loader = loader
        elif mode == "test":
            self.test_loader = loader

        return loader

    def train(self, **kwargs):

        #!########
        self.model.train()
        #!########

        accum_loss = []

        for x, y in self.train_loader:
            x = x.to(self.device)
            y = y.to(self.device)

            output = self.model(x)
            loss = self.loss(output, self.transform_y(y))

            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            accum_loss.append(loss.detach().cpu().numpy())

        average_loss = np.mean(accum_loss)

        self.metric_logger.update(train_loss=average_loss)

        return average_loss

    def evaluate(self,
                 use_train_data,
                 **kwargs):

        #!########
        self.model.eval()
        self.metrics.clear()
        #!########

        loader = self.train_loader if use_train_data else self.test_loader

        accum_loss = []

        for x, y in loader:
            x = x.to(self.device)
            y = y.to(self.device)

            with torch.no_grad():
                output = self.model(x)

            loss = self.loss(output, self.transform_y(y))
            accum_loss.append(loss.detach().cpu().numpy())

            output = self.activation(output)
            y_pred = self.inference(output)

            self.metrics(y, y_pred)

        average_loss = np.mean(accum_loss)
        metrics = self.metrics.get_all()

        return_metrics = {"loss": average_loss, **metrics}

        if use_train_data:
            metrics = {
                f"train_{name}": value for name,
                value in metrics.items()}

            self.metric_logger.update(**metrics)
        else:
            metrics = {
                f"test_{name}": value for name,
                value in metrics.items()}

            self.metric_logger.update(test_loss=average_loss, **metrics)

        return return_metrics

    def log(self):
        return self.metric_logger.log()

    def get_models(self):
        return [self.model]
