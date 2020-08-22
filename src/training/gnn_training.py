from typing import List, Literal, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch_scatter import scatter_mean

from src.gnn import ACGNN
from src.training.utils import MetricLogger
from src.typing import Trainer


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


class Training(Trainer):
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
        if logging_variables != "all" and not all(
                var in self.available_metrics for var in logging_variables):
            raise ValueError(
                "Encountered not supported metric. "
                f"Supported are: {self.available_metrics}")
        self.metric_logger = MetricLogger(logging_variables)

    def get_metric_logger(self):
        return self.metric_logger

    def get_model(self,
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
            return ACGNN(
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

    def get_loss(self):
        return nn.BCEWithLogitsLoss(reduction="mean")

    def get_optim(self, model, lr):
        return optim.Adam(model.parameters(), lr=lr)

    def get_scheduler(self, optimizer, step=50):
        # return optim.lr_scheduler.StepLR(optimizer, step_size=step,
        # gamma=0.5)
        pass

    def train(self,
              model,
              training_data,
              criterion,
              device,
              optimizer,
              *,
              binary_prediction: bool,
              **kwargs):

        #!########
        model.train()
        #!########

        accum_loss = []

        for data in training_data:
            data = data.to(device)

            output = model(x=data.x,
                           edge_index=data.edge_index,
                           batch=data.batch)

            loss = _loss_aux(
                output=output,
                loss=criterion,
                data=data,
                binary=binary_prediction
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # scheduler.step()

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
                 *,
                 binary_prediction: bool,
                 **kwargs):

        #!########
        model.eval()
        #!########

        micro_avg = 0.
        macro_avg = 0.
        n_nodes = 0
        n_graphs = 0
        accum_loss = []

        for data in test_data:
            data = data.to(device)

            with torch.no_grad():
                output = model(
                    x=data.x,
                    edge_index=data.edge_index,
                    batch=data.batch
                )

            loss = _loss_aux(
                output=output,
                loss=criterion,
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
                device=device
            )

            micro_avg += micro.cpu().numpy()
            macro_avg += macro.cpu().numpy()

            n_nodes += data.num_nodes
            n_graphs += data.num_graphs

        average_loss: float = np.mean(accum_loss)
        micro_avg = micro_avg / n_nodes
        macro_avg = macro_avg / n_graphs

        if using_train_data:
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
