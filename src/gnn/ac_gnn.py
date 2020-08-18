from typing import Tuple

import torch
import torch.nn as nn

from .layers import ACConv
from .utils import reset


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
