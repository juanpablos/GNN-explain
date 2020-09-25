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
            use_embedding: bool,
            context_hidden_init: bool,
            dropout_prob: float = 0.0):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.vocab_dim = vocab_size
        self.use_input_init = context_hidden_init

        if use_embedding:
            # * targets are padded sequences
            self.embed = nn.Embedding(
                num_embeddings=vocab_size,
                embedding_dim=embedding_dim)
        else:
            # TODO: 1hot for targets
            raise NotImplementedError("1hot for targets: to be tested")
            # * targets are padded 1-hot vectors
            self.embed = nn.Identity()

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
            h = torch.zeros(
                encoder_out.size(0),
                self.hidden_dim,
                dtype=encoder_out.dtype,
                device=encoder_out.device)
            c = torch.zeros(
                encoder_out.size(0),
                self.hidden_dim,
                dtype=encoder_out.dtype,
                device=encoder_out.device)
        return h, c

    def forward(self, x, padded_target, target_lengths):
        # x: (batch, encoder)
        # padded_target:
        #   (batch, L)          if use_embedding=True
        #   (batch, L, 1hot)    if use_embedding=False
        # target_lengths: (batch,)

        # (batch, L, embedding)
        emb_targets = self.embed(padded_target)

        # * we have to repeat each `x` `seq` times so we can concat them
        # we only have 1 `x` per batch, but we need `seq` of them
        # (batch, L, encoder)
        _expanded = x.unsqueeze(1).expand(-1, emb_targets.size(1), -1)
        # !! teacher-forcing
        # (batch, L, embedding+encoder)
        concat_input = torch.cat([_expanded, emb_targets], dim=2)

        packed_input = pack_padded_sequence(
            concat_input,
            # * we decode from <start> to <eos>, but don't decode <eos>
            # so if len=3 [<start>, hey, <eos>] we just input [<start>, hey]
            lengths=target_lengths - 1,
            batch_first=True,
            enforce_sorted=False)

        state = self.init_hidden_state(x)
        packed_predicted_sequence, _ = self.lstm(
            packed_input, state)  # type: ignore

        padded_predicted_sequence, _ = pad_packed_sequence(
            packed_predicted_sequence, batch_first=True)  # type: ignore

        # (*, lstm_hidden)
        flattened_seq = padded_predicted_sequence.view(-1, self.hidden_dim)
        flattened_seq = self.dropout(flattened_seq)

        # (*, vocab_dim)
        prediction = self.linear(flattened_seq)

        # return (batch, L, vocab_dim)
        return prediction.view(x.size(0), -1, self.vocab_dim)


class LSTMCellDecoder(nn.Module):
    def __init__(
            self,
            encoder_dim: int,
            embedding_dim: int,
            hidden_dim: int,
            vocab_size: int,
            use_embedding: bool,
            context_hidden_init: bool,
            dropout_prob: float = 0.0):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.vocab_dim = vocab_size
        self.use_input_init = context_hidden_init

        if use_embedding:
            # * targets are padded sequences
            self.embed = nn.Embedding(
                num_embeddings=vocab_size,
                embedding_dim=embedding_dim)
        else:
            # TODO: 1hot for targets
            raise NotImplementedError("1hot for targets: to be tested")
            # * targets are padded 1-hot vectors
            self.embed = nn.Identity()

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
            h = torch.zeros(
                encoder_out.size(0),
                self.hidden_dim,
                dtype=encoder_out.dtype,
                device=encoder_out.device)
            c = torch.zeros(
                encoder_out.size(0),
                self.hidden_dim,
                dtype=encoder_out.dtype,
                device=encoder_out.device)
        return h, c

    def forward(self, x, padded_target, target_lengths):
        # x: (batch, encoder)
        # padded_target:
        #   (batch, L)          if use_embedding=True
        #   (batch, L, 1hot)    if use_embedding=False
        # target_lengths: (batch,)

        _, sorted_indices = target_lengths.sort(
            descending=True)

        x = x[sorted_indices]
        padded_target = padded_target[sorted_indices]

        # (batch, L, embedding)
        emb_targets = self.embed(padded_target)

        # both: (batch, lstm_hidden)
        h, c = self.init_hidden_state(x)

        # * we decode from <start> to <eos>, but don't decode <eos>
        # so if len=3 [<start>, hey, <eos>] we just input [<start>, hey]
        target_lengths = (target_lengths - 1).tolist()

        # (batch, L, vocab_dim)
        prediction = torch.zeros(
            x.size(0),
            max(target_lengths),
            self.vocab_dim,
            dtype=x.dtype,
            device=x.device)

        for t in range(max(target_lengths)):
            # T
            step_batch = sum(l > t for l in target_lengths)
            # (T, encoder)
            time_x = x[:step_batch]
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

        return prediction, sorted_indices
