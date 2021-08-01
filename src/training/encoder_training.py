import logging
from typing import List, Literal, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pytorch_metric_learning import losses
from torch.utils.data import DataLoader

from src.models import MLP

from . import Trainer

logger = logging.getLogger(__name__)
logger_metrics = logging.getLogger("metrics")


class EncoderTrainer(Trainer):
    loss: nn.Module
    model: MLP
    optim: torch.optim.Optimizer
    train_loader: DataLoader
    test_loader: DataLoader

    available_metrics = [
        "train_loss",
        "test_loss",
    ]

    def __init__(
        self,
        seed: int = None,
        logging_variables: Union[Literal["all"], List[str]] = "all",
    ):
        super().__init__(seed=seed, logging_variables=logging_variables)
        logger_metrics.info(",".join(self.metric_logger.keys()))

    def init_model(
        self,
        *,
        num_layers: int,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        use_batch_norm: bool = True,
        hidden_layers: List[int] = None,
        **kwargs,
    ):
        self.model = MLP(
            num_layers=num_layers,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            use_batch_norm=use_batch_norm,
            hidden_layers=hidden_layers,
            **kwargs,
        )

        return self.model

    def init_trainer(self, **optim_params):
        self.init_loss()
        self.model = self.model.to(self.device)
        self.init_optim(**optim_params)

    def move_model_to_device(self):
        self.model = self.model.to(self.device)
        return self.model

    def init_loss(self):
        self.loss = losses.ContrastiveLoss()
        return self.loss

    def init_optim(self, lr):
        self.optim = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)
        return self.optim

    def init_dataloader(
        self, data, mode: Union[Literal["train"], Literal["test"]], **kwargs
    ):

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

            if x.size(0) == 1:
                continue

            embedding = self.model(x)
            loss = self.loss(embedding, y)

            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            accum_loss.append(loss.detach().cpu().numpy())

        average_loss = np.mean(accum_loss)

        self.metric_logger.update(train_loss=average_loss)

        return average_loss

    def evaluate(self, use_train_data, **kwargs):

        #!########
        self.model.eval()
        #!########

        if use_train_data:
            return

        loader = self.test_loader

        accum_loss = []

        for x, y in loader:
            x = x.to(self.device)
            y = y.to(self.device)

            with torch.no_grad():
                embedding = self.model(x)

            loss = self.loss(embedding, y)
            accum_loss.append(loss.detach().cpu().numpy())

        average_loss = np.mean(accum_loss)

        metrics = {"test_loss": average_loss}
        self.metric_logger.update(**metrics)

        return metrics

    def get_embedding_with_label(self, data, **kwargs):
        loader = DataLoader(data, **kwargs)

        _embeddings = []
        _labels = []

        for x, y in loader:
            x = x.to(self.device)

            if x.size(0) == 1:
                continue

            with torch.no_grad():
                embedding = self.model(x)

            _embeddings.append(embedding)
            _labels.append(y)

        embeddings = torch.cat(_embeddings).detach().cpu().numpy()
        labels = torch.cat(_labels).numpy()

        return embeddings, labels

    def log(self):
        return self.metric_logger.log()

    def get_models(self):
        return [self.model]
