from typing import Literal

import torch
import torch.nn as nn

from src.models.mlp import MLP


class GNNLayerSingleEncoder(nn.Module):
    def __init__(self, input_dim: int, num_layers: int, hidden_dim: int):
        super(GNNLayerSingleEncoder, self).__init__()

        # (Batch, GNN layers, parameters) -> (Batch, GNN layers, H)
        self.encoder = MLP(
            num_layers=num_layers,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            # ?? True?
            use_batch_norm=False
        )

    def forward(self, x):
        # assume only 1 MLP layer
        # x: (Batch, GNN layers, 1, parameters)

        # (Batch, GNN layers, parameters)
        x = x.squeeze(2)
        # (Batch, GNN layers, H)
        return self.encoder(x)


class GNNEncoder(nn.Module):
    def __init__(
        self,
        layer_input_dim: int,
        output_input_dim: int,
        encoder_num_layers: int,
        encoder_hidden_dim: int,
        layer_embedding_dim: int,
        merge_strategy: Literal['cat', 'sum', 'prod'],
        output_dim: int
    ):
        super(GNNEncoder, self).__init__()

        # (Batch, GNN layers, *, parameters1) -> (Batch, GNN layers, H1)
        self.A_consumer = GNNLayerSingleEncoder(
            input_dim=layer_input_dim,
            num_layers=encoder_num_layers,
            hidden_dim=encoder_hidden_dim)
        self.V_consumer = GNNLayerSingleEncoder(
            input_dim=layer_input_dim,
            num_layers=encoder_num_layers,
            hidden_dim=encoder_hidden_dim)
        # (Batch, parameters2) -> (Batch, H2)
        self.output_consumer = MLP(
            input_dim=output_input_dim,
            num_layers=encoder_num_layers,
            hidden_dim=encoder_hidden_dim,
            output_dim=encoder_hidden_dim,
        )

        # (Batch, H1) x (Batch, H1) -> (Batch, H3)
        self.merge, merge_dim = self._get_merge(
            merge_strategy,
            encoder_hidden_dim
        )

        self.rnn_embedding_dim = layer_embedding_dim

        # (Batch, H3) x (Batch, H4) -> (Batch, H4)
        self.layer_consumer = nn.RNNCell(
            input_size=merge_dim,
            hidden_size=layer_embedding_dim
        )

        # (Batch, H2 + H4) -> (Batch, embedding)
        self.output_layer = nn.Linear(
            encoder_hidden_dim + layer_embedding_dim,
            output_dim
        )

    @staticmethod
    def __cat(a, b):
        return torch.cat([a, b], dim=1)

    @staticmethod
    def __sum(a, b):
        return a + b

    @staticmethod
    def __prod(a, b):
        return a * b

    def _get_merge(self,
                   strategy: Literal['cat', 'sum', 'prod'],
                   hidden_dim: int):
        if strategy == 'cat':
            return self.__cat, 2 * hidden_dim
        elif strategy == 'sum':
            return self.__sum, hidden_dim
        elif strategy == 'prod':
            return self.__prod, hidden_dim
        else:
            raise ValueError('Merge strategy not supported')

    def init_hidden_state(self, tensor):
        return tensor.new_zeros(tensor.size(0), self.rnn_embedding_dim)

    def forward(self, A, V, output):
        # A: (Batch, GNN layers, MLP layers, parameters1)
        # V: (Batch, GNN layers, MLP layers, parameters1)
        # output: (Batch, parameters2)

        # (Batch, GNN layers, H1)
        A_emb = self.A_consumer(A)
        V_emb = self.V_consumer(V)
        # (Batch, H2)
        output_emb = self.output_consumer(output)

        # (Batch, H4)
        state = self.init_hidden_state(A_emb)

        for l in range(A.size(1)):
            A_emb_l = A_emb[:, l, :]
            V_emb_l = V_emb[:, l, :]

            # (Batch, H3)
            layer_emb = self.merge(A_emb_l, V_emb_l)

            # (Batch, H4)
            state = self.layer_consumer(layer_emb, state)

        # (Batch, H2 + H4)
        final = torch.cat([output_emb, state], dim=1)
        return self.output_layer(final)
