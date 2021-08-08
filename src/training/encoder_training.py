import logging
from typing import List, Literal, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pytorch_metric_learning import losses, samplers
from sklearn.neighbors import KNeighborsClassifier
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
        "train_acc",
        "test_loss",
        "test_acc",
    ]

    def __init__(
        self,
        evaluate_with_train: bool,
        seed: int = None,
        logging_variables: Union[Literal["all"], List[str]] = "all",
        loss_name: Literal[
            "contrastive",
            "triplet",
            "lifted_structure",
            "angular",
        ] = "contrastive",
        use_m_per_class_sampler: bool = True,
    ):
        super().__init__(seed=seed, logging_variables=logging_variables)
        logger_metrics.info(",".join(self.metric_logger.keys()))

        self.loss_name = loss_name
        self.use_sampler = use_m_per_class_sampler
        self.evaluate_with_train = evaluate_with_train

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

    @staticmethod
    def __select_loss(loss_name: str):
        available_losses = {
            "contrastive": losses.ContrastiveLoss,
            "triplet": losses.TripletMarginLoss,
            "lifted_structure": losses.LiftedStructureLoss,
            "angular": losses.AngularLoss,
        }
        try:
            return available_losses[loss_name]
        except KeyError as e:
            raise ValueError(f"Passed {loss_name} is not a supported loss") from e

    def init_loss(self):
        self.loss = self.__select_loss(self.loss_name)()
        return self.loss

    def init_optim(self, lr):
        self.optim = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)
        return self.optim

    def init_dataloader(self, data, mode: Literal["train", "test"], **kwargs):
        if mode not in ["train", "test"]:
            raise ValueError("Supported modes are only `train` and `test`")

        if mode == "train":
            if self.evaluate_with_train:
                self.train_eval_dataloader = DataLoader(data, **kwargs)

            if self.use_sampler:
                sampler_enforced_kwargs = {
                    **kwargs,
                    "shuffle": False,
                    "drop_last": True,
                }
                sampler_enforced_kwargs["sampler"] = samplers.MPerClassSampler(
                    labels=[sample[-1] for sample in data],
                    m=2,
                    batch_size=kwargs["batch_size"],
                )
                kwargs = sampler_enforced_kwargs
            self.train_loader = DataLoader(data, **kwargs)
            loader = self.train_loader
        elif mode == "test":
            self.test_loader = DataLoader(data, **kwargs)
            loader = self.test_loader

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

    def _get_embeddings_with_loss_and_labels(
        self, dataloader
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        #!########
        self.model.eval()
        #!########

        accum_loss = []
        _embeddings = []
        _labels = []

        for x, y in dataloader:
            x = x.to(self.device)
            y = y.to(self.device)

            if x.size(0) == 1:
                continue

            with torch.no_grad():
                embedding = self.model(x)
                loss = self.loss(embedding, y)
            accum_loss.append(loss.detach().cpu().numpy())

            _embeddings.append(embedding.detach().cpu())
            _labels.append(y.detach().cpu())

        embeddings = torch.cat(_embeddings).detach().cpu().numpy()
        labels = torch.cat(_labels).numpy()
        average_loss = np.mean(accum_loss)

        return embeddings, labels, average_loss

    def evaluate(self, use_train_data, **kwargs):

        #!########
        self.model.eval()
        #!########

        if use_train_data:
            self.metric_logger.update(train_acc=0)
            return

        (
            test_embeddings,
            test_labels,
            test_loss,
        ) = self._get_embeddings_with_loss_and_labels(dataloader=self.test_loader)

        if self.evaluate_with_train:
            assert hasattr(self, "train_eval_dataloader")
            (
                train_embeddings,
                train_labels,
                _,
            ) = self._get_embeddings_with_loss_and_labels(
                dataloader=self.train_eval_dataloader
            )
        else:
            train_embeddings = test_embeddings
            train_labels = test_labels

        evaluator = KNeighborsClassifier(n_neighbors=15, n_jobs=4)
        evaluator.fit(train_embeddings, train_labels)
        accuracy = evaluator.score(test_embeddings, test_labels)

        metrics = {"test_loss": test_loss, "test_acc": accuracy}

        self.metric_logger.update(**metrics)

        return metrics

    def get_embedding_with_label(self, data, **kwargs):
        #!########
        self.model.eval()
        #!########

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
