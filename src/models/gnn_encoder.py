from typing import Literal, Tuple

import torch
import torch.nn as nn

from src.models.mlp import MLP


class GNNLayerSimpleSingleEncoder(nn.Module):
    def __init__(self, input_dim: int, num_layers: int, hidden_dim: int):
        super(GNNLayerSimpleSingleEncoder, self).__init__()

        # (Batch, GNN layers, parameters) -> (Batch, GNN layers, H)
        self.encoder = MLP(
            num_layers=num_layers,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            # ?? True?
            use_batch_norm=False,
        )

    def forward(self, x):
        # assume only 1 MLP layer
        # x: (Batch, GNN layers, 1, parameters)

        # (Batch, GNN layers, parameters)
        x = x.squeeze(2)
        # (Batch, GNN layers, H)
        return self.encoder(x)


class GNNEmbVariableEncoder(nn.Module):
    def __init__(
        self,
        temp_dim: int,
        hidden_dim: int,
    ):
        # (B, P) -> (B, P, H)
        self.weight_transformer = nn.Linear(1, temp_dim * temp_dim)
        self.bias_transformer = nn.Linear(1, temp_dim)

        # (B, H * (H + 1)) -> (B, H_out)
        self.encoder = nn.Linear(temp_dim * (temp_dim + 1), hidden_dim)

    @staticmethod
    def single_forward(data, layer_parameters, transformer):
        # group P
        x_p_grouped = torch.nn.utils.rnn.pack_padded_sequence(
            data, lengths=layer_parameters, batch_first=True, enforce_sorted=False
        )

        # (B * P, 1)
        p_grouped = x_p_grouped.data.unsqueeze(1)
        # (B * P, H)
        expanded_p = transformer(p_grouped)

        expanded_p_grouped = torch.nn.utils.rnn.PackedSequence(
            expanded_p,
            batch_sizes=x_p_grouped.batch_sizes,
            sorted_indices=x_p_grouped.sorted_indices,
            unsorted_indices=x_p_grouped.unsorted_indices,
        )

        # (B, P, H)
        expanded_p_padded, _ = torch.nn.utils.rnn.pad_packed_sequence(
            expanded_p_grouped, batch_first=True
        )

        # (B, H), remove P dimension
        aggregated_p = expanded_p_padded.sum(dim=1)

        # ?? apply activation?
        return torch.sigmoid(aggregated_p)

    def forward(self, x):
        weights = x["weight"]
        bias = x["bias"]
        weight_parameters = x["weight_parameters"]
        bias_parameters = x["bias_parameters"]

        # (B, H * H), embedding for layer
        weight_embedding = self.single_forward(
            weights, weight_parameters, self.weight_transformer
        )
        # (B, H), embedding for layer
        bias_embedding = self.single_forward(
            bias, bias_parameters, self.bias_transformer
        )

        # (B, H * (H + 1))
        embedding = torch.cat([weight_embedding, bias_embedding], dim=1)

        # (B, H_out), embedding for layer
        encoded_layer = self.encoder(embedding)

        # output is a Tensor of size (B, H_out)
        # ?? apply activation?
        return torch.relu(encoded_layer)


class GNNLayerVariableEncoder(nn.Module):
    def __init__(
        self,
        temp_dim: int,
        hidden_dim: int,
    ):

        # (B, G, P) -> (B, G, P, H)
        self.weight_transformer = nn.Linear(1, temp_dim * temp_dim)
        self.bias_transformer = nn.Linear(1, temp_dim)

        # (B, G, H * (H + 1)) -> (B, G, H_out)
        self.encoder = nn.Linear(temp_dim * (temp_dim + 1), hidden_dim)

    @staticmethod
    def single_forward(data, gnn_layers, layer_parameters, transformer):
        # data: (B, G, P)
        # G is padded
        # P is padded

        # group G
        x_g_grouped = torch.nn.utils.rnn.pack_padded_sequence(
            data, lengths=gnn_layers, batch_first=True, enforce_sorted=False
        )

        # (B * G, P), P is padded
        g_grouped = x_g_grouped.data
        # (B * G * P), represents the lengths of original P
        param_lens = torch.repeat_interleave(layer_parameters, gnn_layers, dim=0)

        # group G and P
        x_gp_grouped = torch.nn.utils.rnn.pack_padded_sequence(
            g_grouped, lengths=param_lens, batch_first=True, enforce_sorted=False
        )

        # (B * G * P, 1)
        gp_grouped = x_gp_grouped.data.unsqueeze(1)
        # (B * G * P, H)
        expanded_gp = transformer(gp_grouped)

        expanded_gp_grouped = torch.nn.utils.rnn.PackedSequence(
            expanded_gp,
            batch_sizes=x_gp_grouped.batch_sizes,
            sorted_indices=x_gp_grouped.sorted_indices,
            unsorted_indices=x_gp_grouped.unsorted_indices,
        )

        # (B * G, P, H), P is padded
        expanded_gp_padded, _ = torch.nn.utils.rnn.pad_packed_sequence(
            expanded_gp_grouped, batch_first=True
        )

        # (B * G, H), remove P dimension
        aggregated_gp = expanded_gp_padded.sum(dim=1)

        # ?? apply activation?
        return (
            torch.sigmoid(aggregated_gp),
            x_g_grouped.batch_sizes,
            x_g_grouped.sorted_indices,
            x_g_grouped.unsorted_indices,
        )

    def forward(self, x):
        weights = x["weight"]
        bias = x["bias"]
        weight_parameters = x["weight_parameters"]
        bias_parameters = x["bias_parameters"]
        gnn_layers = x["gnn_layers"]

        # (B * G, H * H), embedding for layer
        (
            weight_embedding,
            batch_sizes,
            sorted_indices,
            unsorted_indices,
        ) = self.single_forward(
            weights, gnn_layers, weight_parameters, self.weight_transformer
        )
        # (B * G, H), embedding for layer
        bias_embedding, *_ = self.single_forward(
            bias, gnn_layers, bias_parameters, self.bias_transformer
        )

        # (B * G, H * (H + 1))
        embedding = torch.cat([weight_embedding, bias_embedding], dim=1)

        # (B * G, H_out), embedding for layer
        encoded_layer = torch.relu(self.encoder(embedding))

        encoded_layer_back = torch.nn.utils.rnn.PackedSequence(
            encoded_layer,
            batch_sizes=batch_sizes,
            sorted_indices=sorted_indices,
            unsorted_indices=unsorted_indices,
        )

        # output is a PackedSequence for batch and graph layers with H_out
        return encoded_layer_back


class GNNEncoderVariableLayers(nn.Module):
    def __init__(
        self,
        layer_input_dim: int,
        output_input_dim: int,
        encoder_num_layers: int,
        encoder_hidden_dim: int,
        layer_embedding_dim: int,
        merge_strategy: Literal["cat", "sum", "prod"],
        output_dim: int,
    ):
        super(GNNEncoderVariableLayers, self).__init__()

        # (Batch, GNN layers, *, parameters1) -> (Batch, GNN layers, H1)
        self.A_consumer = GNNLayerSimpleSingleEncoder(
            input_dim=layer_input_dim,
            num_layers=encoder_num_layers,
            hidden_dim=encoder_hidden_dim,
        )
        self.V_consumer = GNNLayerSimpleSingleEncoder(
            input_dim=layer_input_dim,
            num_layers=encoder_num_layers,
            hidden_dim=encoder_hidden_dim,
        )
        # (Batch, parameters2) -> (Batch, H2)
        self.output_consumer = MLP(
            input_dim=output_input_dim,
            num_layers=encoder_num_layers,
            hidden_dim=encoder_hidden_dim,
            output_dim=encoder_hidden_dim,
        )

        # (Batch, H1) x (Batch, H1) -> (Batch, H3)
        self.merge, merge_dim = self._get_merge(merge_strategy, encoder_hidden_dim)

        self.rnn_embedding_dim = layer_embedding_dim

        # (Batch, H3) x (Batch, H4) -> (Batch, H4)
        self.layer_consumer = nn.RNNCell(
            input_size=merge_dim, hidden_size=layer_embedding_dim
        )

        # (Batch, H2 + H4) -> (Batch, embedding)
        self.output_layer = nn.Linear(
            encoder_hidden_dim + layer_embedding_dim, output_dim
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

    def _get_merge(self, strategy: Literal["cat", "sum", "prod"], hidden_dim: int):
        if strategy == "cat":
            return self.__cat, 2 * hidden_dim
        elif strategy == "sum":
            return self.__sum, hidden_dim
        elif strategy == "prod":
            return self.__prod, hidden_dim
        else:
            raise ValueError("Merge strategy not supported")

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


class GNNEncoderFullVariable(nn.Module):
    def __init__(  # TODO: missing variables to input
        self,
        input_embedding: int,
        input_embedding_temp: int,
        output_embedding: int,
        _embedding: int,
        gnn_output_layer_dim: Tuple[int, int],
        gnn_layer_dim: Tuple[int, int],
        layer_encoder_num_layers: int,
        layer_encoder_hidden_dim: int,
        recurrent_encoder_embedding_dim: int,
        encoder_output_dim: int,
        merge_strategy: Literal["cat", "sum", "prod"],
        input_layer_present: bool = True,
    ):
        super(GNNEncoderFullVariable, self).__init__()

        self.input_layer = input_layer_present

        if input_layer_present:
            # (Batch, parameters2) -> (Batch, H)
            self.input_consumer = GNNEmbVariableEncoder(
                temp_dim=...,
                hidden_dim=...,
            )

        # (Batch, GNN layers, parameters1) -> (Batch * GNN layers, H)
        self.A_consumer = GNNLayerVariableEncoder(
            temp_dim=...,
            hidden_dim=...,
        )
        self.V_consumer = GNNLayerVariableEncoder(
            temp_dim=...,
            hidden_dim=...,
        )
        # (Batch, parameters3) -> (Batch, H)
        self.output_consumer = GNNEmbVariableEncoder(
            temp_dim=...,
            hidden_dim=...,
        )

        # (Batch * GNN layers, H) x (Batch * GNN layers, H)
        #   -> (Batch * GNN layers, H2)
        self.merge, merge_dim = self._get_merge(
            merge_strategy, layer_encoder_hidden_dim
        )

        self.rnn_embedding_dim = recurrent_encoder_embedding_dim

        # (Batch * GNN layers, H2) x (Batch, H3) -> (Batch, H3)
        self.layer_consumer = nn.RNN(input_size=..., hidden_size=..., batch_first=True)

        if input_layer_present:
            # (Batch, H) -> (Batch, H3)
            self.state_init = nn.Linear(..., ...)

        # (Batch, H3 + H) -> (Batch, embedding)
        self.output_layer = nn.Linear(... + ..., encoder_output_dim)

    @staticmethod
    def __cat(a, b):
        return torch.cat([a, b], dim=2)

    @staticmethod
    def __sum(a, b):
        return a + b

    @staticmethod
    def __prod(a, b):
        return a * b

    def _get_merge(self, strategy: Literal["cat", "sum", "prod"], hidden_dim: int):
        if strategy == "cat":
            return self.__cat, 2 * hidden_dim
        elif strategy == "sum":
            return self.__sum, hidden_dim
        elif strategy == "prod":
            return self.__prod, hidden_dim
        else:
            raise ValueError("Merge strategy not supported")

    def init_hidden_state(self, tensor):
        if self.input_layer:
            return self.state_init(tensor)
        else:
            return tensor.new_zeros(tensor.size(0), self.rnn_embedding_dim)

    def forward(self, A, V, output_params, input_params=None):
        # A: (Batch, GNN layers, parameters) dict with 'weight' and 'bias'
        # V: (Batch, GNN layers, parameters) dict with 'weight' and 'bias'
        # input_params: (Batch, parameters2) dict with 'weight' and 'bias'
        # output_params: (Batch, parameters3) dict with 'weight' and 'bias'

        # (Batch, H)
        output_emb = self.output_consumer(output_params)

        # PackedSequence(Batch * GNN layers, H)
        A_emb = self.A_consumer(A)
        V_emb = self.V_consumer(V)

        # (Batch * GNN layers, H2)
        layer_merge_emb = torch.nn.utils.rnn.PackedSequence(
            self.merge(A_emb.data, V_emb.data),
            batch_sizes=A_emb.batch_sizes,
            sorted_indices=A_emb.sorted_indices,
            unsorted_indices=A_emb.unsorted_indices,
        )

        if input_params is not None:
            # (Batch, H)
            # TODO: add to layer_merge_emb?
            input_emb = self.input_consumer(input_params)
        else:
            input_emb = output_emb

        # (Batch, H3)
        state = self.init_hidden_state(input_emb)

        _, layers_embedding = self.layer_consumer(
            layer_merge_emb, state  # type: ignore
        )

        # (Batch, H3 + H)
        final = torch.cat([layers_embedding, output_emb], dim=1)
        return self.output_layer(final)
