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
        """A simple MLP class with batchnorm.

        Args:
            num_layers (int): the number of layers. If `hidden_layers` is set, then this argument is ignored. When the value is negative the Identity function is returned.
            input_dim (int): the input dimension for the first layer.
            hidden_dim (int): the dimension of all the hidden layers. If `hidden_layers` is set, then this argument is ignored.
            output_dim (int): the output dimension of the last layer.
            hidden_layers (List[int], optional): A list indicating the different dimensions of the hidden layers. Each element of the list is the dimension of a layer. Defaults to None.
        """

        super(MLP, self).__init__()

        if hidden_layers is not None:
            # plus 1 because these are the hidden dims, the intermediate values
            # (1, 2) - (2, 3) - (3, 4), here we have hidden=[2, 3]
            # with input 1 and output 4
            # if hidden_layers is empty, then a Linear[in, out] is returned
            self.num_layers = len(hidden_layers) + 1
            layers = hidden_layers
        else:
            self.num_layers = num_layers
            # num_layers - 1 = number of hidden layers
            layers = [hidden_dim] * (num_layers - 1)

        self.is_linear = True

        # this allows us the parse the weights easier
        self.linears = nn.ModuleList()

        if self.num_layers < 1:
            # no-op
            self.linears.append(nn.Identity())
        elif self.num_layers == 1:
            # linear model
            self.linears.append(nn.Linear(input_dim, output_dim))
        else:
            # multi layer model
            self.is_linear = False
            self.batch_norms = nn.ModuleList()

            last_dim = input_dim
            for hidden in layers:
                self.linears.append(nn.Linear(last_dim, hidden))
                self.batch_norms.append(nn.BatchNorm1d(hidden))

                last_dim = hidden

            self.linears.append(nn.Linear(last_dim, output_dim))

    def forward(self, x):
        if self.is_linear:
            return self.linears[0](x)
        else:
            h = x
            for layer in range(self.num_layers - 1):
                h = torch.relu(self.batch_norms[layer](self.linears[layer](h)))
            return self.linears[self.num_layers - 1](h)

    def reset_parameters(self):
        reset(self.linears)
        if not self.is_linear:
            reset(self.batch_norms)
