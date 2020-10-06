import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class LSTMDecoder(nn.Module):
    # * maybe faster, but less flexible
    def __init__(
            self,
            encoder_dim: int,
            embedding_dim: int,
            hidden_dim: int,
            vocab_size: int,
            context_hidden_init: bool,
            dropout_prob: float = 0.0,
            pad_token_id: int = 0):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.vocab_dim = vocab_size
        self.use_input_init = context_hidden_init
        self.pad_token_id = pad_token_id

        # * targets are padded sequences
        self.embed = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim)

        if context_hidden_init:
            self.init_h = nn.Linear(encoder_dim, hidden_dim)
            self.init_c = nn.Linear(encoder_dim, hidden_dim)

        self.lstm = nn.LSTM(
            input_size=encoder_dim + embedding_dim,
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

        # (batch, lstm_hidden)
        return h, c

    def forward(
            self,
            encoder_out,
            padded_target,
            target_lengths):
        """
        encoder_out: (batch, encoder)
        padded_target: (batch, L)
        target_lengths: (batch,)

        input:
            encoder_out (batch, encoder)
            padded_targets (batch, L)
            target_lengths (batch)
        return:
            predictions (batch * L, vocab_dim)
            targets (batch * L)
        """

        # (batch, L, embedding)
        emb_targets = self.embed(padded_target)

        # * we have to repeat each `encoder_out` `seq` times so we can concat them
        # we only have 1 `encoder_out` per batch, but we need `seq` of them
        # (batch, L, encoder)
        _expanded = encoder_out.unsqueeze(
            1).expand(-1, emb_targets.size(1), -1)
        # * teacher-forcing
        # (batch, L, encoder+embedding)
        concat_input = torch.cat([_expanded, emb_targets], dim=2)

        packed_input = pack_padded_sequence(
            concat_input,
            # * we decode from <start> to <eos>, but don't decode <eos>
            # so if len=3 [<start>, hey, <eos>] we just input
            # [<start>, hey]
            lengths=target_lengths - 1,
            batch_first=True,
            enforce_sorted=False)

        state = self.init_hidden_state(encoder_out)
        packed_predicted_sequence, _ = self.lstm(
            packed_input, state)  # type: ignore

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
            step_state):
        """
        encoder_out: (batch, encoder)
        sequence_step: (batch,)
        step_state: tuple[(batch, lstm_hidden)]

        input:
            encoder_out (batch, encoder)
            sequence_step (batch)
            step_state tuple[(batch, lstm_hidden)]
        return:
            prediction (batch, vocab_dim)
            tuple[ h, c ] tuple[(batch, lstm_hidden)]
        """

        # (batch, embedding)
        step_emb = self.embed(sequence_step)

        # (batch, encoder+embedding)
        concat_input = torch.cat([encoder_out, step_emb], dim=1)

        # (batch, 1, embedding+encoder)
        concat_input = concat_input.unsqueeze(1)

        # h: (1, batch, lstm_hidden)
        # c: (1, batch, lstm_hidden)
        _, (h, c) = self.lstm(
            concat_input, step_state)

        # (batch, lstm_hidden)
        c = c.squeeze(0)
        h = h.squeeze(0)
        # h = self.dropout(h)

        # (batch, vocab_dim)
        prediction = self.linear(h)

        # prediction: (batch, vocab_dim)
        # h: (batch, lstm_hidden)
        # c: (batch, lstm_hidden)
        return prediction, (h, c)


class LSTMCellDecoder(nn.Module):
    def __init__(
            self,
            encoder_dim: int,
            embedding_dim: int,
            hidden_dim: int,
            vocab_size: int,
            context_hidden_init: bool,
            dropout_prob: float = 0.0,
            pad_token_id: int = 0):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.vocab_dim = vocab_size
        self.use_input_init = context_hidden_init
        self.pad_token_id = pad_token_id

        # * targets are padded sequences
        self.embed = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim)

        if context_hidden_init:
            self.init_h = nn.Linear(encoder_dim, hidden_dim)
            self.init_c = nn.Linear(encoder_dim, hidden_dim)

        self.lstm = nn.LSTMCell(
            input_size=encoder_dim + embedding_dim,
            hidden_size=hidden_dim)

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

        # (batch, lstm_hidden)
        return h, c

    def forward(self,
                encoder_out,
                padded_target,
                target_lengths):
        """
        encoder_out: (batch, encoder)
        padded_target: (batch, L)
        target_lengths: (batch,)

        input:
            encoder_out (batch, encoder)
            padded_targets (batch, L)
            target_lengths (batch)
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

        # both: (batch, lstm_hidden)
        h, c = self.init_hidden_state(encoder_out)

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
            h, c = self.lstm(
                torch.cat([time_x, time_t_target], dim=1),
                (h[:step_batch], c[:step_batch])
            )

            # (T, vocab_size)
            t_pred = self.linear(self.dropout(h))

            # fill predictions
            prediction[:step_batch, t, :] = t_pred

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
                    step_state):
        """
        encoder_out: (batch, encoder)
        sequence_step: (batch,)
        step_state: tuple[(batch, lstm_hidden)]

        input:
            encoder_out (batch, encoder)
            sequence_step (batch)
            step_state tuple[(batch, lstm_hidden)]
        return:
            prediction (batch, vocab_dim)
            tuple[ h, c ] tuple[(batch, lstm_hidden)]
        """

        # (batch, embedding)
        step_emb = self.embed(sequence_step)

        # (batch, encoder+embedding)
        concat_input = torch.cat([encoder_out, step_emb], dim=1)

        # h: (batch, lstm_hidden)
        # c: (batch, lstm_hidden)
        h, c = self.lstm(concat_input, step_state)

        # h = self.dropout(h)

        # (batch, vocab_dim)
        prediction = self.linear(h)

        # prediction: (batch, vocab_dim)
        # h: (batch, lstm_hidden)
        # c: (batch, lstm_hidden)
        return prediction, (h, c)
