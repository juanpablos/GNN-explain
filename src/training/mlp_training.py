import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from src.gnn import MLP


class Metric:
    def __init__(self):
        self.correct_true = 0.0
        self.predicted_true = 0.0
        self.true_true = 0.0

    def __call__(self, y_true, y_pred):
        self.correct_true += torch.sum(y_pred * y_true).item()
        self.predicted_true += torch.sum(y_pred).item()
        self.true_true += torch.sum(y_true).item()

    def precision(self):
        return self.correct_true / self.predicted_true

    def recall(self):
        return self.correct_true / self.true_true

    def f1(self):
        precision = self.precision()
        recall = self.recall()

        return 2 * (precision * recall) / (precision + recall)


class Training:
    def get_model(
            num_layers: int,
            input_dim: int,
            hidden_dim: int,
            output_dim: int):
        return MLP(num_layers=num_layers,
                   input_dim=input_dim,
                   hidden_dim=hidden_dim,
                   output_dim=output_dim)

    def get_loss():
        # ??: what loss should be here
        return nn.BCEWithLogitsLoss(reduction="mean")

    def get_optim(model, lr):
        return optim.Adam(model.parameters(), lr=lr)

    def get_scheduler(optimizer, step=50):
        pass

    def train(
            model: nn.Module,
            training_data: DataLoader,
            criterion,
            device,
            optimizer,
            **kwargs) -> float:

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

        return average_loss

    def evaluate(
            model: nn.Module,
            test_data: DataLoader,
            criterion,
            device,
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

            loss = criterion(output, y)
            accum_loss.append(loss.detach().cpu().numpy())

            output = torch.sigmoid(output)
            _, y_pred = output.max(dim=1)

            metric(y, y_pred)

        average_loss = np.mean(accum_loss)

        return average_loss, metric.precision(), metric.recall()
