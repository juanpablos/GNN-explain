from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from src.gnn import MLP
from src.typing import Trainer


class Metric:
    def __init__(self):
        self.tp = 0.0
        self.tn = 0.0
        self.fp = 0.0
        self.fn = 0.0

    def __call__(self, y_true, y_pred):
        self.tp = ((y_true == 1) & (y_pred == 1)).sum().cpu().item()
        self.tn = ((y_true == 0) & (y_pred == 0)).sum().cpu().item()
        self.fp = ((y_true == 0) & (y_pred == 1)).sum().cpu().item()
        self.fn = ((y_true == 1) & (y_pred == 0)).sum().cpu().item()

    def precision(self):
        return self.tp / (self.tp + self.fp)

    def recall(self):
        return self.tp / (self.tp + self.fn)

    def f1(self):
        precision = self.precision()
        recall = self.recall()

        return 2 * (precision * recall) / (precision + recall)

    def accuracy(self):
        return (self.tp + self.tn) / (self.tp + self.tn + self.fp + self.fn)


class Training(Trainer):
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

    def get_loss(self):
        return nn.BCEWithLogitsLoss(reduction="mean")

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
              collector,
              **kwargs):

        #!########
        model.train()
        #!########

        accum_loss = []

        for x, y in training_data:
            x = x.to(device)
            y = y.to(device)

            output = model(x)
            loss = criterion(output, F.one_hot(y, 2).float())

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
                 collector,
                 **kwargs):

        #!########
        model.eval()
        #!########

        accum_loss = []

        metric = Metric()

        for x, y in test_data:
            x = x.to(device)
            y = y.to(device)

            with torch.no_grad():
                output = model(x)

            loss = criterion(output, F.one_hot(y, 2).float())
            accum_loss.append(loss.detach().cpu().numpy())

            output = torch.sigmoid(output)
            _, y_pred = output.max(dim=1)

            metric(y, y_pred)

        average_loss = np.mean(accum_loss)
        collector["test_loss"] = average_loss
        collector["precision"] = metric.precision()
        collector["recall"] = metric.recall()
        collector["f1score"] = metric.f1()
        collector["acc"] = metric.accuracy()

        return average_loss, metric.precision(), metric.recall()

    def log(self, info):
        return "loss {train_loss: <10.6f} test_loss {test_loss: <10.6f} precision {precision: <10.4f} recall {recall: <10.4f} f1score {f1score: <10.4f} accuracy {acc:.4f}".format(
            **info)
