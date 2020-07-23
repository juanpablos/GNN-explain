from typing import List, Optional
import torch
import torch.nn as nn

from .utils import reset


class MLP(nn.Module):
    def __init__(
            self,
            num_layers: int,
            input_dim: int,
            hidden_dim: int,
            output_dim: int,
            hidden_layers: Optional[List[int]] = None):

        # !! TODO: implement hidden_layers
        super(MLP, self).__init__()

        self.layers = num_layers
        self.is_linear = True

        # the reason this is defined here is so that the weight parsing is
        # easier
        self.linears = nn.ModuleList()

        if num_layers < 1:
            # do nothing
            self.linears.append(nn.Identity())
        elif num_layers == 1:
            # linear model
            self.linears.append(nn.Linear(input_dim, output_dim))
        else:
            # multi layer model
            self.is_linear = False
            self.batch_norms = nn.ModuleList()

            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for _ in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))

            for _ in range(num_layers - 1):
                self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

    def forward(self, x):
        if self.is_linear:
            return self.linears[0](x)
        else:
            h = x
            for layer in range(self.layers - 1):
                h = torch.relu(self.batch_norms[layer](self.linears[layer](h)))
            return self.linears[self.layers - 1](h)

    def reset_parameters(self):
        reset(self.linears)
        if not self.is_linear:
            reset(self.batch_norms)
