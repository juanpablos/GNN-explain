import logging
from typing import List, Literal, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pytorch_metric_learning import distances, losses, miners, reducers, samplers
from sklearn.metrics._classification import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from torch.utils.data import DataLoader

from src.models import MLP

from . import Trainer

logger = logging.getLogger(__name__)
logger_metrics = logging.getLogger("metrics")


class NullMiner(miners.BaseMiner):
    def forward(self, embeddings, labels, ref_emb=None, ref_labels=None):
        return None


class EncoderTrainer(Trainer):
    loss: nn.Module
    test_loss: nn.Module
    miner: nn.Module
    model: MLP
    optim: torch.optim.Optimizer
    train_loader: DataLoader
    test_loader: DataLoader

    output_size: int

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
            "triplet_cosine",
            "lifted_structure",
            "angular",
            "ntxent",
        ] = "contrastive",
        miner_name: Literal["none", "similarity", "triplet", "triplet_cosine"] = "none",
        miner_pairs: Literal["semihard", "all", None] = "all",
        use_cross_batch: bool = True,
        use_m_per_class_sampler: bool = True,
    ):
        super().__init__(seed=seed, logging_variables=logging_variables)
        logger_metrics.info(",".join(self.metric_logger.keys()))

        self.loss_name = loss_name
        self.miner_name = miner_name
        self.miner_pairs = miner_pairs
        self.use_cross_batch = use_cross_batch
        self.use_sampler = use_m_per_class_sampler
        self.evaluate_with_train = evaluate_with_train

        logger.info(f"Will evaluate with train data: {self.evaluate_with_train}")

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

        self.output_size = output_dim

        return self.model

    def init_trainer(self, **optim_params):
        self.init_miner()
        self.init_loss()
        self.model = self.model.to(self.device)
        self.init_optim(**optim_params)

    def move_model_to_device(self):
        self.model = self.model.to(self.device)
        return self.model

    @staticmethod
    def __select_miner(miner_name: str, miner_pairs: Optional[str]) -> miners.BaseMiner:
        available_miners = {
            "triplet": miners.TripletMarginMiner(
                margin=0.2, type_of_triplets=miner_pairs
            ),
            "triplet_cosine": miners.TripletMarginMiner(
                margin=0.2,
                type_of_triplets=miner_pairs,
                distance=distances.CosineSimilarity(),
            ),
            "similarity": miners.MultiSimilarityMiner(epsilon=0.1),
            "none": NullMiner(),
        }
        try:
            return available_miners[miner_name]
        except KeyError as e:
            raise ValueError(f"Passed {miner_name} is not a supported miner") from e

    @staticmethod
    def __select_loss(
        loss_name: str, output_size: int, use_cross_batch: bool
    ) -> Tuple[nn.Module, nn.Module]:
        available_losses = {
            "contrastive": losses.ContrastiveLoss(),
            "triplet": losses.TripletMarginLoss(margin=0.2),
            "triplet_cosine": losses.TripletMarginLoss(
                margin=0.2,
                distance=distances.CosineSimilarity(),
                reducer=reducers.ThresholdReducer(low=0),
            ),
            "lifted_structure": losses.LiftedStructureLoss(),
            "angular": losses.AngularLoss(),
            "ntxent": losses.NTXentLoss(),
        }
        try:
            loss = available_losses[loss_name]
            return (
                losses.CrossBatchMemory(
                    loss,
                    embedding_size=output_size,
                    memory_size=1024,
                )
                if use_cross_batch
                else loss,
                loss,
            )
        except KeyError as e:
            raise ValueError(f"Passed {loss_name} is not a supported loss") from e

    def init_loss(self):
        self.loss, self.test_loss = self.__select_loss(
            self.loss_name,
            output_size=self.output_size,
            use_cross_batch=self.use_cross_batch,
        )
        return self.loss

    def init_miner(self):
        self.miner = self.__select_miner(self.miner_name, self.miner_pairs)
        return self.miner

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

            mined_indices = self.miner(embedding, y)
            loss = self.loss(embedding, y, mined_indices)

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
                loss = self.test_loss(embedding, y)
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
        predictions = evaluator.predict(test_embeddings)

        unique_labels, label_counts = np.unique(predictions, return_counts=True)
        count_distributions = label_counts / np.sum(label_counts)
        distribution = dict(zip(unique_labels, count_distributions))

        accuracy = accuracy_score(y_true=test_labels, y_pred=predictions)

        metrics = {"test_loss": test_loss, "test_acc": accuracy}

        self.metric_logger.update(**metrics)
        self.metric_logger.update_extra_data(prediction_distribution=distribution)

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
