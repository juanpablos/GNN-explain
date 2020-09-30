import logging
from itertools import chain
from typing import Dict, List, Literal, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from nltk.translate.bleu_score import corpus_bleu
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from src.models import MLP, LSTMCellDecoder, LSTMDecoder

from . import Trainer

logger = logging.getLogger(__name__)


class Collator:
    def __init__(self, pad_token: int):
        self.pad_token = pad_token

    def __call__(self, batch):
        # both: tuple of tensors
        x, y = zip(*batch)
        x = torch.stack(x)

        y_lens = torch.tensor([target.size(0) for target in y])
        y_pad = pad_sequence(y, batch_first=True, padding_value=self.pad_token)

        return x, y_pad, y_lens


class Metric:
    def accuracy(self, scores, targets, k, lengths):
        # scores: logits (batch, L, vocab) with padding
        # targets: indices (batch, L) with padding
        # k: int
        # lengths: (batch,)

        # (batch, L, k)
        _, indices = scores.topk(k, dim=2, largest=True, sorted=True)
        # expand the targets to check if they occur in one of the topk
        _expanded = targets.unsqueeze(dim=2).expand_as(indices)

        # check if the correct index is in one of the topk
        # matches: (batch, L)
        matches = torch.any(indices.eq(_expanded), dim=2)

        # all sequence indices must occur somewhere in the topk
        # filter by length
        # correct: int
        correct = torch.tensor(0.0)
        for i, l in enumerate(lengths):
            correct += torch.all(matches[i, :l]).float()

        # average over predictions that have the correct index in the topk
        return (correct / targets.size(0)).item()

    def blue_score(self, predictions, targets, lengths):
        # use indices instead of string tokens
        # predictions: indices (batch, L) with padding
        # targets: indices (batch, L) with padding

        references = []
        hypothesis = []
        for i, l in enumerate(lengths):
            references.append([targets[i, :l].tolist()])
            hypothesis.append(predictions[i, :l].tolist())
        return corpus_bleu(references, hypothesis)


class RecurrentTrainer(Trainer):
    loss: nn.Module
    encoder: MLP
    decoder: Union[LSTMDecoder, LSTMCellDecoder]
    optim: torch.optim.Optimizer
    train_loader: DataLoader
    test_loader: DataLoader

    available_metrics = [
        "train_acc1",
        "train_acc5",
        "train_blue4",
        "test_acc1",
        "test_acc5",
        "test_blue4"
    ]

    def __init__(self,
                 vocabulary: Dict[str, int],
                 logging_variables: Union[Literal["all"], List[str]] = "all"):

        super().__init__(logging_variables=logging_variables)
        self.pad_token = vocabulary["<pad>"]
        self.start_token = vocabulary["<start>"]
        self.vocabulary = vocabulary
        self.metrics = Metric()

    def activation(self, output, dim=1):
        return torch.log_softmax(output, dim=dim)

    def inference(self, output, dim=1):
        _, y_pred = output.max(dim=dim)
        return y_pred

    def init_encoder(self,
                     *,
                     num_layers: int,
                     input_dim: int,
                     hidden_dim: int,
                     output_dim: int,
                     use_batch_norm: bool = True,
                     hidden_layers: List[int] = None,
                     **kwargs):
        self.encoder = MLP(num_layers=num_layers,
                           input_dim=input_dim,
                           hidden_dim=hidden_dim,
                           output_dim=output_dim,
                           use_batch_norm=use_batch_norm,
                           hidden_layers=hidden_layers,
                           **kwargs)

        return self.encoder

    def init_decoder(self,
                     *,
                     name: str,
                     encoder_dim: int,
                     embedding_dim: int,
                     hidden_dim: int,
                     vocab_size: int,
                     context_hidden_init: bool,
                     dropout_prob: float = 0.0,
                     **kwargs):

        if name == "lstm":
            self.decoder = LSTMDecoder(
                encoder_dim=encoder_dim,
                embedding_dim=embedding_dim,
                hidden_dim=hidden_dim,
                vocab_size=vocab_size,
                context_hidden_init=context_hidden_init,
                dropout_prob=dropout_prob)
        elif name == "lstmcell":
            self.decoder = LSTMCellDecoder(
                encoder_dim=encoder_dim,
                embedding_dim=embedding_dim,
                hidden_dim=hidden_dim,
                vocab_size=vocab_size,
                context_hidden_init=context_hidden_init,
                dropout_prob=dropout_prob)
        else:
            raise ValueError("Only values `lstm` and `lstmcell` are supported")

        return self.decoder

    def init_trainer(self, **optim_params):
        self.init_loss()
        self.encoder = self.encoder.to(self.device)
        self.decoder = self.decoder.to(self.device)
        self.init_optim(**optim_params)

    def init_loss(self):
        self.loss = nn.CrossEntropyLoss(
            reduction="mean", ignore_index=self.pad_token)
        return self.loss

    def init_optim(self, lr):
        encoder_parameters = self.encoder.parameters()
        decoder_parameters = self.decoder.parameters()
        self.optim = optim.Adam(
            chain(encoder_parameters, decoder_parameters),
            lr=lr)
        return self.optim

    def init_dataloader(self,
                        data,
                        mode: Union[Literal["train"], Literal["test"]],
                        **kwargs):

        if mode not in ["train", "test"]:
            raise ValueError("Supported modes are only `train` and `test`")

        loader = DataLoader(
            data,
            collate_fn=Collator(self.pad_token),
            **kwargs)
        if mode == "train":
            self.train_loader = loader
        elif mode == "test":
            self.test_loader = loader

        return loader

    def train(self, **kwargs):

        #!########
        self.encoder.train()
        self.decoder.train()
        #!########

        accum_loss = []

        for x, y, y_lens in self.train_loader:
            x = x.to(self.device)
            y = y.to(self.device)
            y_lens = y_lens.to(self.device)

            # (batch, encoder_dim)
            encoder_out = self.encoder(x)
            # output: (batch * L, vocab_dim), L max seq
            # targets: (batch * L,)
            output, targets = self.decoder(
                encoder_out=encoder_out,
                padded_target=y,
                target_lengths=y_lens,
                train=True)

            # the loss ignores padding
            loss = self.loss(output, targets)

            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            accum_loss.append(loss.detach().cpu().numpy())

        average_loss = np.mean(accum_loss)

        self.metric_logger.update(train_loss=average_loss)

        return average_loss

    def evaluate(self,
                 use_train_data,
                 **kwargs):

        #!########
        self.encoder.eval()
        self.decoder.eval()
        #!########

        loader = self.train_loader if use_train_data else self.test_loader

        accum_loss = []

        predictions = []
        scores = []
        targets = []
        lengths = []

        with torch.no_grad():
            for x, y, y_lens in loader:
                x = x.to(self.device)
                y = y.to(self.device)
                y_lens = y_lens.to(self.device)

                # remove the <start> token
                y = y[:, 1:]
                y_lens = y_lens - 1

                # (batch, encoder_dim)
                encoder_out = self.encoder(x)

                # (batch,)
                input_tokens = y.new_full(
                    (x.size(0),),
                    fill_value=self.start_token).to(self.device)

                # (batch, L), L is max seq
                batch_predictions = y.new_full(
                    (x.size(0), y.size(1)),
                    fill_value=self.pad_token).to(self.device)

                # (batch, L, vocab_dim), L is max seq
                batch_scores = torch.full(
                    (x.size(0), y.size(1), self.decoder.vocab_dim),
                    fill_value=self.pad_token,
                    dtype=torch.float).to(self.device)

                # tuple (batch, lstm_hidden)
                states = self.decoder.init_hidden_state(
                    encoder_out=encoder_out)

                for t in range(y.size(1)):
                    # batch_pred: (batch, vocab_dim)
                    # states: tuple (batch, lstm_hidden)
                    batch_pred, states = self.encoder(
                        encoder_out=encoder_out,
                        sequence_step=input_tokens,
                        step_state=states,
                        train=False
                    )

                    batch_scores[:, t, :] = batch_pred

                    # (batch, vocab_dim)
                    output = self.activation(batch_pred)
                    # (batch,)
                    output = self.inference(output)
                    # copy predicted tokens to batch_predictions
                    batch_predictions[:, t] = output

                    input_tokens = output

                scores.append(batch_scores.detach())
                predictions.append(batch_predictions.detach())
                targets.append(y.detach())
                lengths.append(y_lens.detach())

                # flatten the batch scores to (*, vocab_dim) and the targets to
                # a vector
                # the loss will ignore the padding tokens
                loss = self.loss(batch_scores.view(-1, self.decoder.vocab_dim),
                                 y.view(-1))
                accum_loss.append(loss.detach().cpu().numpy())

        average_loss = np.mean(accum_loss)

        epoch_scores = torch.cat(scores, dim=0).cpu()
        epoch_predictions = torch.cat(predictions, dim=0).cpu()
        epoch_targets = torch.cat(targets, dim=0).cpu()
        epoch_lengths = torch.cat(lengths, dim=0).cpu()

        metrics = {
            "acc1": self.metrics.accuracy(
                scores=epoch_scores,
                targets=epoch_targets,
                k=1,
                lengths=epoch_lengths),
            "acc5": self.metrics.accuracy(
                scores=epoch_scores,
                targets=epoch_targets,
                k=5,
                lengths=epoch_lengths),
            "blue4": self.metrics.blue_score(
                predictions=epoch_predictions,
                targets=epoch_targets,
                lengths=epoch_lengths)
        }

        if use_train_data:
            metrics = {
                f"train_{name}": value for name,
                value in metrics.items()}

            self.metric_logger.update(**metrics)
        else:
            metrics = {
                f"test_{name}": value for name,
                value in metrics.items()}

            self.metric_logger.update(test_loss=average_loss, **metrics)

        return average_loss

    def log(self):
        return self.metric_logger.log()
