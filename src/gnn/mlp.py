import torch
import torch.nn as nn
from utils import reset


class MLP(nn.Module):
    def __init__(
            self,
            num_layers: int,
            input_dim: int,
            hidden_dim: int,
            output_dim: int):

        super(MLP, self).__init__()

        self.layers = num_layers
        self.is_linear = True

        if num_layers < 1:
            # do nothing
            self.linear = nn.Identity()
        elif num_layers == 1:
            # linear model
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            # multi layer model
            self.is_linear = False
            self.linears = nn.ModuleList()
            self.batch_norms = nn.ModuleList()

            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for _ in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))

            for _ in range(num_layers - 1):
                self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

    def forward(self, x):
        if self.is_linear:
            return self.linear(x)
        else:
            h = x
            for layer in range(self.layers - 1):
                h = torch.relu(self.batch_norms[layer](self.linears[layer](h)))
            return self.linears[self.layers - 1](h)

    def reset_parameters(self):
        if self.is_linear:
            reset(self.linear)
        else:
            reset(self.linears)
            reset(self.batch_norms)
