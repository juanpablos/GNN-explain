import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from src.models.utils import Waiter


class LSTMDecoder(nn.Module):
    # * maybe faster, but less flexible
    def __init__(
            self,
            encoder_dim: int,
            embedding_dim: int,
            hidden_dim: int,
            vocab_size: int,
            init_state_context: bool,
            concat_encoder_input: bool,
            dropout_prob: float = 0.0,
            pad_token_id: int = 0):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.vocab_dim = vocab_size

        self.use_input_init = init_state_context

        self.encoder_as_input = concat_encoder_input

        self.pad_token_id = pad_token_id

        # * targets are padded sequences
        self.embed = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim)

        if init_state_context:
            self.init_h = nn.Linear(encoder_dim, hidden_dim)
            self.init_c = nn.Linear(encoder_dim, hidden_dim)

        if self.encoder_as_input:
            lstm_input_size = encoder_dim + embedding_dim
        else:
            lstm_input_size = embedding_dim

        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=hidden_dim,
            batch_first=True)

        self.dropout = nn.Dropout(p=dropout_prob)

        self.linear = nn.Linear(hidden_dim, vocab_size)

    def init_hidden_state(self, encoder_out):
        if self.use_input_init:
            h = self.init_h(encoder_out)
            c = self.init_c(encoder_out)
        else:
            # init with zeros
            h = encoder_out.new_zeros(encoder_out.size(0), self.hidden_dim)
            c = encoder_out.new_zeros(encoder_out.size(0), self.hidden_dim)

        # (1, batch, lstm_hidden)
        return h.unsqueeze(0), c.unsqueeze(0)

    def forward(
            self,
            encoder_out,
            padded_target,
            target_lengths):
        """
        input:
            encoder_out (batch, encoder)
            padded_targets (batch, L)
            target_lengths (batch,)
        return:
            predictions (batch * L, vocab_dim)
            targets (batch * L)
        """

        # (batch, L, embedding)
        emb_targets = self.embed(padded_target)

        if self.encoder_as_input:
            # * we have to repeat each `encoder_out` `seq` times so we can concat them
            # we only have 1 `encoder_out` per batch, but we need `seq` of them
            # (batch, L, encoder)
            _expanded = encoder_out.unsqueeze(
                1).expand(-1, emb_targets.size(1), -1)
            # * teacher-forcing
            # (batch, L, encoder+embedding)
            lstm_input = torch.cat([_expanded, emb_targets], dim=2)
        else:
            lstm_input = emb_targets

        packed_input = pack_padded_sequence(
            lstm_input,
            # * we decode from <start> to <eos>, but don't decode <eos>
            # so if len=3 [<start>, hey, <eos>] we just input
            # [<start>, hey]
            lengths=target_lengths - 1,
            batch_first=True,
            enforce_sorted=False)

        # (1, batch, lstm_hidden)
        state = self.init_hidden_state(encoder_out)
        packed_predicted_sequence, _ = self.lstm(
            packed_input, state)  # type: ignore

        # (batch, L, lstm_hidden)
        padded_predicted_sequence, _ = pad_packed_sequence(
            packed_predicted_sequence, batch_first=True)  # type: ignore

        # (*, lstm_hidden)
        flattened_seq = padded_predicted_sequence.view(-1, self.hidden_dim)
        flattened_seq = self.dropout(flattened_seq)

        # (*, vocab_dim)
        prediction = self.linear(flattened_seq)

        # skip <start>
        target_predictions = padded_target[:, 1:]

        # prediction includes pads
        # (*, vocab_dim), (*)
        return prediction, target_predictions.flatten()

    def single_step(
            self,
            encoder_out,
            sequence_step,
            step_state,
            **kwargs):
        """
        input:
            encoder_out (batch, encoder)
            sequence_step (batch,)
            step_state tuple[(1, batch, lstm_hidden)]
        return:
            prediction (batch, vocab_dim)
            tuple[ h, c ] tuple[(batch, lstm_hidden)]
        """

        # (batch, embedding)
        step_emb = self.embed(sequence_step)

        if self.encoder_as_input:
            # (batch, encoder+embedding)
            lstm_input = torch.cat([encoder_out, step_emb], dim=1)
        else:
            # (batch, embedding)
            lstm_input = step_emb

        # (batch, 1, embedding+encoder)
        concat_input = lstm_input.unsqueeze(1)

        # h: (1, batch, lstm_hidden)
        # c: (1, batch, lstm_hidden)
        _, (h, c) = self.lstm(concat_input, step_state)

        # h = self.dropout(h)

        # (batch, lstm_hidden)
        output = h.squeeze(0)
        # (batch, vocab_dim)
        prediction = self.linear(output)

        # prediction: (batch, vocab_dim)
        # h: (1, batch, lstm_hidden)
        # c: (1, batch, lstm_hidden)
        return prediction, (h, c)


class LSTMCellDecoder(nn.Module):
    def __init__(
            self,
            encoder_dim: int,
            embedding_dim: int,
            hidden_dim: int,
            vocab_size: int,
            init_state_context: bool,
            compose_encoder_state: bool,
            concat_encoder_input: bool,
            compose_dim: int = 0,
            dropout_prob: float = 0.0,
            pad_token_id: int = 0):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.vocab_dim = vocab_size

        self.init_state_encoder = init_state_context
        self.use_compose_encoder = compose_encoder_state

        self.encoder_as_input = concat_encoder_input

        self.pad_token_id = pad_token_id

        # * targets are padded sequences
        self.embed = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim)

        if self.init_state_encoder:
            self.encoder2hidden = nn.Linear(encoder_dim, hidden_dim)
            self.encoder2cell = nn.Linear(encoder_dim, hidden_dim)

        if self.use_compose_encoder:
            self.encoder2hiddentemp = nn.Linear(encoder_dim, compose_dim)
            self.encoder2celltemp = nn.Linear(encoder_dim, compose_dim)
            self.hidden2temp = nn.Linear(hidden_dim, compose_dim)
            self.cell2temp = nn.Linear(hidden_dim, compose_dim)

            self.temp2hidden = nn.Linear(compose_dim, hidden_dim)
            self.temp2cell = nn.Linear(compose_dim, hidden_dim)

        if self.encoder_as_input:
            lstm_input_size = encoder_dim + embedding_dim
        else:
            lstm_input_size = embedding_dim

        self.lstm = nn.LSTMCell(
            input_size=lstm_input_size,
            hidden_size=hidden_dim)

        self.dropout = nn.Dropout(p=dropout_prob)

        self.linear = nn.Linear(hidden_dim, vocab_size)

        self.default_wait = 1 if self.init_state_encoder else 0

    def init_hidden_state(self, encoder_out):
        if self.init_state_encoder:
            # ?? should we use this same matrix for the composer?
            h = self.encoder2hidden(encoder_out)
            c = self.encoder2cell(encoder_out)
        else:
            # init with zeros
            h = encoder_out.new_zeros(encoder_out.size(0), self.hidden_dim)
            c = encoder_out.new_zeros(encoder_out.size(0), self.hidden_dim)

        # (batch, lstm_hidden)
        return h, c

    @staticmethod
    def compose(tensor1, tensor2, A, B, C):
        # tensor1: (batch, D1)
        # tensor2: (batch, D2)
        # -> (batch, D3)

        # A: (D1, DX)
        # B: (D2, DX)
        # C: (DX, D3)

        # Formula:
        # C(A(T1) * B(T2)), * is hadamard product

        dx1 = A(tensor1)
        dx2 = B(tensor2)

        intermediate = dx1 + dx2

        return C(intermediate)

    def compose_encoder(self, encoder_out, hidden_state, cell_state):
        # encoder_out: (batch, encoder)
        # other: (batch, lstm_hidden)
        # -> (batch, lstm_hidden)

        composed_hidden = self.compose(
            encoder_out,
            hidden_state,
            self.encoder2hiddentemp,
            self.hidden2temp,
            self.temp2hidden)
        composed_cell = self.compose(
            encoder_out,
            cell_state,
            self.encoder2celltemp,
            self.cell2temp,
            self.temp2cell)

        return composed_hidden, composed_cell

    def forward(self,
                encoder_out,
                padded_target,
                target_lengths):
        """
        input:
            encoder_out (batch, encoder)
            padded_targets (batch, L)
            target_lengths (batch,)
        return:
            predictions (batch * L, vocab_dim)
            targets (batch * L)
        """

        _, sorted_indices = target_lengths.sort(
            descending=True)

        encoder_out = encoder_out[sorted_indices]
        padded_target = padded_target[sorted_indices]

        # (batch, L, embedding)
        emb_targets = self.embed(padded_target)

        # * do this independently of if the composer is used, as we need to init the states with something.
        # both: (batch, lstm_hidden)
        h, c = self.init_hidden_state(encoder_out)

        waiter = Waiter(self.default_wait)

        # * we decode from <start> to <eos>, but don't decode <eos>
        # so if len=3 [<start>, hey, <eos>] we just input [<start>, hey]
        fixed_target_lengths = target_lengths - 1

        max_len = fixed_target_lengths.max()

        # (batch, L, vocab_dim)
        prediction = encoder_out.new_full(
            (encoder_out.size(0), max_len, self.vocab_dim),
            fill_value=self.pad_token_id)

        for t in torch.arange(max_len):
            # T
            step_batch = (fixed_target_lengths > t).sum()
            # (T, encoder)
            time_x = encoder_out[:step_batch]
            # (T, embedding)
            time_t_target = emb_targets[:step_batch, t, :]

            # both: (T, lstm_hidden)
            h = h[:step_batch]
            c = c[:step_batch]

            # * do not use the composer in the first step if context init'ed
            if self.use_compose_encoder and waiter.ok():
                # both: (T, lstm_hidden)
                h, c = self.compose_encoder(time_x, h, c)

            if self.encoder_as_input:
                # (T, encoder+embedding)
                lstm_input = torch.cat([time_x, time_t_target], dim=1)
            else:
                # (T, embedding)
                lstm_input = time_t_target

            # both: (T, lstm_hidden)
            h, c = self.lstm(lstm_input, (h, c))

            # (T, vocab_size)
            t_pred = self.linear(self.dropout(h))

            # fill predictions
            prediction[:step_batch, t] = t_pred

        # flatten
        output_predictions = prediction.view(-1, self.vocab_dim)
        # skip <start>
        # padded_target is batch sorted the same as output_predictions
        target_predictions = padded_target[:, 1:]

        # prediction includes pads
        # (*, vocab_dim), (*)
        return output_predictions, target_predictions.flatten()

    def single_step(self,
                    encoder_out,
                    sequence_step,
                    step_state,
                    step: int = 0):
        """
        input:
            encoder_out (batch, encoder)
            sequence_step (batch,)
            step_state tuple[(batch, lstm_hidden)]
            step: int - to check if compose or not
        return:
            prediction (batch, vocab_dim)
            tuple[ h, c ] tuple[(batch, lstm_hidden)]
        """

        # (batch, embedding)
        step_emb = self.embed(sequence_step)

        if self.encoder_as_input:
            # (batch, encoder+embedding)
            lstm_input = torch.cat([encoder_out, step_emb], dim=1)
        else:
            # (batch, embedding)
            lstm_input = step_emb

        h, c = step_state
        if self.use_compose_encoder and step != 0:
            # if h,c == 0, then no matter the composition it will also be 0 so
            # no need to check if init with context
            h, c = self.compose_encoder(encoder_out, h, c)

        # h: (batch, lstm_hidden)
        # c: (batch, lstm_hidden)
        h, c = self.lstm(lstm_input, step_state)

        # h = self.dropout(h)

        # (batch, vocab_dim)
        prediction = self.linear(h)

        # prediction: (batch, vocab_dim)
        # h: (batch, lstm_hidden)
        # c: (batch, lstm_hidden)
        return prediction, (h, c)
