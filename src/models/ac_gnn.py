from typing import Tuple

import torch
import torch.nn as nn
import torch_geometric.nn as geom_nn

from src.models.layers import ACConv, NetworkConv
from src.models.utils import reset


class ACGNN(torch.nn.Module):

    def __init__(
            self,
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
            **kwargs
    ):
        super(ACGNN, self).__init__()

        self.num_layers = num_layers

        if task != "node":
            raise NotImplementedError(
                "No support for task other than `node` yet")
        self.task = task

        self.bigger_input = input_dim > hidden_dim
        self.weighted_combine = combine_type != "identity"

        if not self.bigger_input:
            self.padding = nn.ConstantPad1d(
                (0, hidden_dim - input_dim), value=0)

        if truncated_fn is not None:
            self.activation = nn.Hardtanh(
                min_val=truncated_fn[0],
                max_val=truncated_fn[1])
        else:
            self.activation = nn.ReLU()

        # add the convolutions
        self.convs = torch.nn.ModuleList()
        for layer in range(self.num_layers):
            if layer == 0 and self.bigger_input:
                self.convs.append(ACConv(input_dim=input_dim,
                                         output_dim=hidden_dim,
                                         aggregate_type=aggregate_type,
                                         combine_type=combine_type,
                                         combine_layers=combine_layers,
                                         mlp_layers=mlp_layers))
            else:
                self.convs.append(ACConv(input_dim=hidden_dim,
                                         output_dim=hidden_dim,
                                         aggregate_type=aggregate_type,
                                         combine_type=combine_type,
                                         combine_layers=combine_layers,
                                         mlp_layers=mlp_layers))

        self.batch_norms = torch.nn.ModuleList()
        # placeholder
        identity = nn.Identity()
        for _ in range(self.num_layers):
            if use_batch_norm:
                # add the batchnorms if selected
                self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
            else:
                self.batch_norms.append(identity)

        self.linear_prediction = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, edge_index, batch):

        h = x
        if not self.bigger_input:
            h = self.padding(h)

        for conv, norm in zip(self.convs, self.batch_norms):
            h = conv(h=h, edge_index=edge_index, batch=batch)
            # ?? we only apply the activation function if no combine function is selected (eg. identity, that is a noop)
            if not self.weighted_combine:
                h = self.activation(h)
            h = norm(h)

        if self.task == "node":
            return self.linear_prediction(h)
        else:
            # TODO: do a global readout here to summarize the whole hidden
            # state
            raise NotImplementedError()

    def reset_parameters(self):
        reset(self.convs)
        reset(self.batch_norms)
        reset(self.linear_prediction)


class NetworkACGNN(torch.nn.Module):

    def __init__(
            self,
            hidden_dim: int,
            output_dim: int,
            mlp_layers: int,
            **kwargs
    ):
        super(NetworkACGNN, self).__init__()

        self.num_layers = 8

        self.padding = nn.ConstantPad1d((0, hidden_dim - 1), value=0)

        self.activation = nn.ReLU()

        # add the convolutions
        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        for _ in range(self.num_layers):
            self.convs.append(NetworkConv(input_dim=hidden_dim,
                                          output_dim=hidden_dim,
                                          mlp_layers=mlp_layers))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        # self.readout = geom_nn.global_mean_pool
        _gate = nn.Linear(hidden_dim, 1)
        _nn = nn.Linear(hidden_dim, hidden_dim)
        self.readout = geom_nn.GlobalAttention(_gate, _nn)

        self.linear_prediction = nn.Sequential(
            nn.Linear(hidden_dim * self.num_layers, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim)
        )
        # self.linear_prediction = nn.Linear(hidden_dim * self.num_layers, output_dim)

    def forward(self, x, edge_index, edge_weight, batch):

        # (N, H)
        h = self.padding(x.view(-1, 1))
        weight = edge_weight.view(-1, 1)

        layers = []

        for conv, norm in zip(self.convs, self.batch_norms):
            h = conv(
                h=h,
                edge_index=edge_index,
                edge_weight=weight)
            h = self.activation(h)
            h = norm(h)

            # readout for each layer output
            layers.append(self.readout(h, batch=batch))
            # layers.append(h)

        # concat residuals
        cat_h = torch.cat(layers, dim=1)
        return self.linear_prediction(cat_h)
        # return self.readout(cat_h, batch=batch)

    def reset_parameters(self):
        reset(self.convs)
        reset(self.batch_norms)
