from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from src.gnn import MLP
from src.typing import TNum, Trainer


class Metric:
    def __init__(self, average: str = "micro"):
        if average not in ["binary", "micro", "macro"]:
            raise ValueError(
                "Argument `average` must be one of `binary`, `micro`, `macro`")
        self.acc_y = []
        self.acc_y_pred = []
        self.average = average

    def __call__(self, y_true: torch.Tensor, y_pred: torch.Tensor):
        self.acc_y.extend(y_true.tolist())
        self.acc_y_pred.extend(y_pred.tolist())

    def precision_recall_fscore(self) -> Dict[str, TNum]:
        precision, recall, f1score, _ = precision_recall_fscore_support(
            self.acc_y, self.acc_y_pred, average=self.average, beta=1.0)

        return {"precision": precision, "recall": recall, "f1score": f1score}

    def accuracy(self):
        return {"acc": accuracy_score(self.acc_y, self.acc_y_pred)}

    def clear(self):
        self.acc_y.clear()
        self.acc_y_pred.clear()


class Training(Trainer):

    def __init__(self, n_classes: int = 2, metrics_average: str = "micro"):
        self.n_classes = n_classes
        self.metrics = Metric(average=metrics_average)

    def transform_y(self, y):
        if self.n_classes == 2:
            return F.one_hot(y, 2).float()
        else:
            return y

    def activation(self, output):
        if self.n_classes == 2:
            return F.sigmoid(output)
        else:
            return F.log_softmax(output, dim=1)

    def get_loss(self):
        if self.n_classes > 2:
            return nn.CrossEntropyLoss(reduction="mean")
        elif self.n_classes == 2:
            return nn.BCEWithLogitsLoss(reduction="mean")
        else:
            raise ValueError("Number of classes cannot be less than 2")

    def get_model(self,
                  num_layers: int,
                  input_dim: int,
                  hidden_dim: int,
                  output_dim: int,
                  hidden_layers: List[int] = None,
                  **kwargs):
        return MLP(num_layers=num_layers,
                   input_dim=input_dim,
                   hidden_dim=hidden_dim,
                   hidden_layers=hidden_layers,
                   output_dim=output_dim)

    def get_optim(self, model, lr):
        return optim.Adam(model.parameters(), lr=lr)

    def get_scheduler(self, **kwargs):
        pass

    def train(self,
              model,
              training_data,
              criterion,
              device,
              optimizer,
              collector: Dict[str, TNum],
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

        collector["train_loss"] = average_loss

        return average_loss

    def evaluate(self,
                 model,
                 test_data,
                 criterion,
                 device,
                 collector: Dict[str, TNum],
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
        collector["test_loss"] = average_loss
        collector.update(self.metrics.precision_recall_fscore())
        collector.update(self.metrics.accuracy())

        return average_loss

    def log(self, info: Dict[str, TNum]):
        return """loss {train_loss: <10.6f} \
                test_loss {test_loss: <10.6f} \
                precision {precision: <10.4f} \
                recall {recall: <10.4f} \
                f1score {f1score: <10.4f} \
                accuracy {acc:.4f}""".format(**info)
