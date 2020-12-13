import torch
from torch_geometric.nn.conv import MessagePassing

from src.models.mlp import MLP


class ACConv(MessagePassing):
    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            aggregate_type: str,
            combine_type: str,
            combine_layers: int,
            mlp_layers: int,
            use_batch_norm: bool = True,
            **kwargs):

        if aggregate_type not in ["add", "mean", "max"]:
            raise ValueError(
                "`aggregate_type` must be one of: add, mean or max")
        if combine_type not in ["identity", "linear", "mlp"]:
            raise ValueError(
                "`combine_type` must be one of: identity, linear or mlp")

        super(ACConv, self).__init__(aggr=aggregate_type, **kwargs)

        _args = {
            "num_layers": mlp_layers,
            "input_dim": input_dim,
            "hidden_dim": output_dim,
            "output_dim": output_dim,
            "use_batch_norm": use_batch_norm
        }

        if combine_type == "identity":
            # to have an Identity layer
            self.combine = MLP(
                num_layers=0,
                input_dim=output_dim,
                hidden_dim=output_dim,
                output_dim=output_dim,
                use_batch_norm=use_batch_norm
            )
        elif combine_type == "linear":
            self.combine = MLP(
                num_layers=1,
                input_dim=output_dim,
                hidden_dim=output_dim,
                output_dim=output_dim,
                use_batch_norm=use_batch_norm
            )
        else:
            self.combine = MLP(
                num_layers=combine_layers,
                input_dim=output_dim,
                hidden_dim=output_dim,
                output_dim=output_dim,
                use_batch_norm=use_batch_norm
            )

        self.V = MLP(**_args)
        self.A = MLP(**_args)

    def forward(self, h, edge_index, batch):
        return self.propagate(
            edge_index=edge_index,
            h=h
        )

    def message(self, h_j):
        return h_j

    def update(self, aggr, h):
        updated = self.V(h) + self.A(aggr)
        return self.combine(updated)

    def reset_parameters(self):
        self.V.reset_parameters()
        self.A.reset_parameters()
        self.combine.reset_parameters()


class NetworkConv(MessagePassing):
    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            mlp_layers: int,
            **kwargs):

        super(NetworkConv, self).__init__(aggr='add', **kwargs)

        _args = {
            "num_layers": mlp_layers,
            "input_dim": input_dim,
            "hidden_dim": output_dim,
            "output_dim": output_dim,
            "use_batch_norm": True
        }

        self.V = MLP(**_args)
        self.A = MLP(**_args)

        self.C = MLP(num_layers=1,
                     input_dim=output_dim,
                     hidden_dim=output_dim,
                     output_dim=output_dim,
                     use_batch_norm=True)

    def forward(self, h, edge_index, edge_weight):
        # assumes the input is 2d
        # h: (N, H)
        # edge_index: (E, 2)
        # edge_weight: (E, 1)

        return self.propagate(
            edge_index=edge_index,
            h=h,
            edge_weight=edge_weight
        )

    def message(self, h_j, edge_weight):
        return h_j * edge_weight

    def update(self, aggr, h):
        return self.C(torch.relu(self.V(h) + self.A(aggr)))

    def reset_parameters(self):
        self.V.reset_parameters()
        self.A.reset_parameters()
