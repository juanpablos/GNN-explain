from torch_geometric.nn.conv import MessagePassing
from .mlp import MLP


class ACConv(MessagePassing):
    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            aggregate_type: str = "add",
            combine_type: str = None,
            combine_layers: int = 1,
            mlp_layers: int = 1,
            **kwargs):

        assert aggregate_type in ["add", "mean", "max"]
        assert combine_type in [None, "mlp"]

        super(ACConv, self).__init__(aggr=aggregate_type, **kwargs)

        _args = {
            "num_layers": combine_layers,
            "input_dim": input_dim,
            "hidden_dim": output_dim,
            "output_dim": output_dim
        }

        if combine_type is not None:
            self.combine = MLP(
                num_layers=mlp_layers,
                input_dim=output_dim,
                hidden_dim=output_dim,
                output_dim=output_dim
            )
        else:
            # to have an Identity layer
            self.combine = MLP(
                num_layers=-mlp_layers,
                input_dim=output_dim,
                hidden_dim=output_dim,
                output_dim=output_dim
            )

        self.V = MLP(**_args)
        self.A = MLP(**_args)

    def forward(self, h, edge_index):
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


# if __name__ == "__main__":
#     t = ACConv(5, 1, "add", None, 2, 2)

#     # print(t.state_dict())
#     for i, m in enumerate(t.modules()):
#         print(type(m))
#         for p in m.parameters(recurse=False):
#             print(i, p)
#     # print(list(t.parameters()))
