from typing import List, Literal, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import DataLoader
from torch_scatter import scatter_mean

from src.gnn import ACGNN

from . import Trainer


def _loss_aux(output, loss, data, binary):
    if binary:
        # REV: F.one_hot(output, 2).float().to(device) should do the same
        labels = torch.zeros_like(output).scatter_(dim=1,
                                                   index=data.y.unsqueeze(1),
                                                   value=1.)
    else:
        # TODO: missing option when not just predicting 0/1
        raise NotImplementedError(
            "GNN trainig only supports binary classifiction")

    return loss(output, labels)


def _accuracy_aux(node_labels, predicted_labels, batch, device):
    results = torch.eq(
        predicted_labels,
        node_labels).float().to(device)

    micro = torch.sum(results)
    macro = torch.sum(scatter_mean(results, batch))

    return micro, macro


class GNNTrainer(Trainer):
    available_metrics = [
        "all",
        "train_loss",
        "test_loss",
        "train_macro",
        "train_micro",
        "test_macro",
        "test_micro"
    ]

    def __init__(self,
                 logging_variables: Union[Literal["all"],
                                          List[str]] = "all"):
        super().__init__(logging_variables=logging_variables)

    def init_model(self,
                   *,
                   name: str,
                   input_dim: int,
                   hidden_dim: int,
                   output_dim: int,
                   aggregate_type: str,
                   combine_type: str,
                   num_layers: int,
                   combine_layers: int,
                   mlp_layers: int,
                   task: str,
                   use_batch_norm: bool = True,
                   truncated_fn: Tuple[int, int] = None,
                   **kwargs):

        if name == "acgnn":
            self.model = ACGNN(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                output_dim=output_dim,
                aggregate_type=aggregate_type,
                combine_type=combine_type,
                num_layers=num_layers,
                combine_layers=combine_layers,
                mlp_layers=mlp_layers,
                task=task,
                use_batch_norm=use_batch_norm,
                truncated_fn=truncated_fn,
                **kwargs
            )
        else:
            raise NotImplementedError("Only acgnn supported")

        # just in case
        self.model = self.model.to(self.device)
        return self.model

    def init_loss(self):
        self.loss = nn.BCEWithLogitsLoss(reduction="mean")
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

    def train(self, *, binary_prediction: bool, **kwargs):

        #!########
        self.model.train()
        #!########

        accum_loss = []

        for data in self.train_loader:
            data = data.to(self.device)

            output = self.model(x=data.x,
                                edge_index=data.edge_index,
                                batch=data.batch)

            loss = _loss_aux(
                output=output,
                loss=self.loss,
                data=data,
                binary=binary_prediction
            )

            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            accum_loss.append(loss.detach().cpu().numpy())

        average_loss = np.mean(accum_loss)

        self.metric_logger.update(train_loss=average_loss)

        return average_loss

    def evaluate(self,
                 use_train_data: bool,
                 *,
                 binary_prediction: bool,
                 **kwargs):

        #!########
        self.model.eval()
        #!########

        micro_avg = 0.
        macro_avg = 0.
        n_nodes = 0
        n_graphs = 0
        accum_loss = []

        loader = self.train_loader if use_train_data else self.test_loader

        for data in loader:
            data = data.to(self.device)

            with torch.no_grad():
                output = self.model(
                    x=data.x,
                    edge_index=data.edge_index,
                    batch=data.batch
                )

            loss = _loss_aux(
                output=output,
                loss=self.loss,
                data=data,
                binary=binary_prediction
            )

            accum_loss.append(loss.detach().cpu().numpy())

            output = torch.sigmoid(output)
            _, predicted_labels = output.max(dim=1)

            micro, macro = _accuracy_aux(
                node_labels=data.y,
                predicted_labels=predicted_labels,
                batch=data.batch,
                device=self.device
            )

            micro_avg += micro.cpu().numpy()
            macro_avg += macro.cpu().numpy()

            n_nodes += data.num_nodes
            n_graphs += data.num_graphs

        average_loss: float = np.mean(accum_loss)
        micro_avg = micro_avg / n_nodes
        macro_avg = macro_avg / n_graphs

        if use_train_data:
            self.metric_logger.update(
                train_micro=micro_avg,
                train_macro=macro_avg)
        else:
            self.metric_logger.update(
                test_loss=average_loss,
                test_micro=micro_avg,
                test_macro=macro_avg)

        return average_loss, micro_avg, macro_avg

    def log(self):
        return self.metric_logger.log()
