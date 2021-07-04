import logging
import os
from itertools import chain
from typing import Dict, List, Literal, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from src.data.auxiliary import FormulaAppliedDatasetWrapper
from src.data.vocabulary import Vocabulary
from src.models import MLP, LSTMCellDecoder, LSTMDecoder
from src.training.metrics import SequenceMetrics

from . import Trainer

logger = logging.getLogger(__name__)
logger_metrics = logging.getLogger("metrics")


class Collator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, batch):
        # x,y: tuple of tensors
        # inds: tuple of numbers
        x, y, inds = zip(*batch)
        x = torch.stack(x)

        y_lens = torch.tensor([target.size(0) for target in y])
        y_pad = pad_sequence(y, batch_first=True, padding_value=self.pad_token_id)

        return x, y_pad, y_lens, inds


class RecurrentTrainer(Trainer):
    loss: nn.Module
    encoder: MLP
    decoder: Union[LSTMDecoder, LSTMCellDecoder]
    optim: torch.optim.Optimizer
    train_loader: DataLoader
    test_loader: DataLoader

    available_metrics = [
        "train_loss",
        "train_token_acc1",
        "train_token_acc3",
        "train_sent_acc",
        "train_bleu4",
        "train_valid",
        "train_semvalPRE",
        "train_semvalREC",
        "train_semvalACC",
        "test_loss",
        "test_token_acc1",
        "test_token_acc3",
        "test_sent_acc",
        "test_bleu4",
        "test_valid",
        "test_semvalPRE",
        "test_semvalREC",
        "test_semvalACC",
    ]

    def __init__(
        self,
        vocabulary: Vocabulary,
        target_apply_mapping: FormulaAppliedDatasetWrapper = None,
        seed: int = None,
        subset_size: float = 0.2,
        logging_variables: Union[Literal["all"], List[str]] = "all",
        write_checkpoints: bool = False,
        checkpoints_path: str = "results",
        checkpoints_name: str = "model",
    ):

        super().__init__(seed=seed, logging_variables=logging_variables)
        self.vocabulary = vocabulary
        self.metrics = SequenceMetrics(
            seed=seed,
            subset_size=subset_size,
            vocabulary=vocabulary,
            result_mapping=target_apply_mapping,
        )
        self.write_checkpoints = write_checkpoints
        self.checkpoints_path = checkpoints_path
        self.checkpoints_name = checkpoints_name

        self._internal_epoch = 0

        logger_metrics.info(",".join(self.metric_logger.keys()))

    def activation(self, output, dim=1):
        return torch.log_softmax(output, dim=dim)

    def inference_func(self, output, dim=1):
        _, y_pred = output.max(dim=dim)
        return y_pred

    def init_encoder(
        self,
        *,
        num_layers: int,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        use_batch_norm: bool = True,
        hidden_layers: List[int] = None,
        model_weights: Dict[str, torch.Tensor] = None,
        **kwargs,
    ):
        self.encoder = MLP(
            num_layers=num_layers,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            use_batch_norm=use_batch_norm,
            hidden_layers=hidden_layers,
            **kwargs,
        )

        if model_weights is not None:
            self.encoder.load_state_dict(model_weights)

        return self.encoder

    def init_decoder(
        self,
        *,
        name: str,
        encoder_dim: int,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        init_state_context: bool,
        concat_encoder_input: bool,
        dropout_prob: float = 0.0,
        model_weights: Dict[str, torch.Tensor] = None,
        **kwargs,
    ):

        pad_token_id = self.vocabulary.pad_token_id

        if name == "lstm":
            self.decoder = LSTMDecoder(
                encoder_dim=encoder_dim,
                embedding_dim=embedding_dim,
                hidden_dim=hidden_dim,
                vocab_size=vocab_size,
                init_state_context=init_state_context,
                concat_encoder_input=concat_encoder_input,
                dropout_prob=dropout_prob,
                pad_token_id=pad_token_id,
            )
        elif name == "lstmcell":
            self.decoder = LSTMCellDecoder(
                encoder_dim=encoder_dim,
                embedding_dim=embedding_dim,
                hidden_dim=hidden_dim,
                vocab_size=vocab_size,
                init_state_context=init_state_context,
                concat_encoder_input=concat_encoder_input,
                dropout_prob=dropout_prob,
                pad_token_id=pad_token_id,
                **kwargs,
            )
        else:
            raise ValueError("Only values `lstm` and `lstmcell` are supported")

        if model_weights is not None:
            self.decoder.load_state_dict(model_weights)

        return self.decoder

    def init_trainer(self, inference: bool = False, **optim_params):
        self.encoder = self.encoder.to(self.device)
        self.decoder = self.decoder.to(self.device)
        if not inference:
            self.init_loss()
            self.init_optim(**optim_params)

    def init_loss(self):
        self.loss = nn.CrossEntropyLoss(
            reduction="mean", ignore_index=self.vocabulary.pad_token_id
        )
        return self.loss

    def init_optim(self, lr: float = 0.01):
        encoder_parameters = self.encoder.parameters()
        decoder_parameters = self.decoder.parameters()
        self.optim = optim.Adam(chain(encoder_parameters, decoder_parameters), lr=lr)
        return self.optim

    def init_dataloader(
        self, data, mode: Union[Literal["train", "test"], None], **kwargs
    ):
        loader = DataLoader(
            data, collate_fn=Collator(self.vocabulary.pad_token_id), **kwargs
        )
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

        for x, y, y_lens, _ in self.train_loader:
            x = x.to(self.device)
            y = y.to(self.device)
            y_lens = y_lens.to(self.device)

            # (batch, encoder_dim)
            encoder_out = self.encoder(x)
            # output: (batch * L, vocab_dim), L max seq
            # targets: (batch * L,)
            # output order is not guaranteed
            # when model is LSTMCell they are sorted by real length
            # when model is LSTM they are sorted by input order
            output, targets = self.decoder(
                encoder_out=encoder_out, padded_target=y, target_lengths=y_lens
            )

            # the loss ignores padding
            loss = self.loss(output, targets)

            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            accum_loss.append(loss.detach().cpu().numpy())

        average_loss = np.mean(accum_loss)

        self.metric_logger.update(train_loss=average_loss)
        self._internal_epoch += 1

        return average_loss

    def evaluate(self, use_train_data, **kwargs):

        #!########
        self.encoder.eval()
        self.decoder.eval()
        #!########

        loader = self.train_loader if use_train_data else self.test_loader

        (
            epoch_scores,
            epoch_predictions,
            epoch_targets,
            epoch_lengths,
            epoch_indices,
            average_loss,
        ) = self.run_pass(loader, keep_device=True)

        metrics = {
            "token_acc1": self.metrics.token_accuracy(
                scores=epoch_scores, targets=epoch_targets, k=1, lengths=epoch_lengths
            ),
            "token_acc3": self.metrics.token_accuracy(
                scores=epoch_scores, targets=epoch_targets, k=3, lengths=epoch_lengths
            ),
            "sent_acc": self.metrics.sentence_accuracy(
                predictions=epoch_predictions,
                targets=epoch_targets,
                lengths=epoch_lengths,
            ),
            "bleu4": self.metrics.bleu_score(
                predictions=epoch_predictions,
                targets=epoch_targets,
                lengths=epoch_lengths,
            ),
            "valid": self.metrics.syntax_check(predictions=epoch_predictions),
        }

        semval = self.metrics.semantic_validation(
            predictions=epoch_predictions, indices=epoch_indices
        )
        for metric_name, value in semval.items():
            metrics[f"semval{metric_name}"] = value

        return_metrics = {"loss": average_loss, **metrics}

        if use_train_data:
            metrics = {f"train_{name}": value for name, value in metrics.items()}

            self.metric_logger.update(**metrics)
        else:
            metrics = {f"test_{name}": value for name, value in metrics.items()}

            self.metric_logger.update(test_loss=average_loss, **metrics)

        return return_metrics

    def run_pass(self, dataloader, keep_device: bool = True):

        #!########
        self.encoder.eval()
        self.decoder.eval()
        #!########

        accum_loss = []

        predictions = []
        scores = []
        targets = []
        lengths = []

        indices = []

        total_batch = 0
        max_sequence = 0

        with torch.no_grad():
            for x, y, y_lens, inds in dataloader:
                x: torch.Tensor = x.to(self.device)
                y = y.to(self.device)
                y_lens = y_lens.to(self.device)

                # remove the <start> token
                y = y[:, 1:]
                y_lens = y_lens - 1

                # (batch, encoder_dim)
                encoder_out = self.encoder(x)

                # (batch,)
                input_tokens = y.new_full(
                    (x.size(0),), fill_value=self.vocabulary.start_token_id
                )

                # (batch, L), L is max seq
                batch_predictions = y.new_full(
                    (x.size(0), y.size(1)), fill_value=self.vocabulary.pad_token_id
                )

                # (batch, L, vocab_dim), L is max seq
                batch_scores = x.new_full(
                    (x.size(0), y.size(1), self.decoder.vocab_dim),
                    fill_value=self.vocabulary.pad_token_id,
                )

                total_batch += batch_scores.size(0)
                max_sequence = max(max_sequence, batch_scores.size(1))

                # tuple
                # lstm (1, batch, lstm_hidden)
                # lstmcell (batch, lstm_hidden)
                states = self.decoder.init_hidden_state(encoder_out=encoder_out)

                for t in range(y.size(1)):
                    # batch_pred: (batch, vocab_dim)
                    # states
                    # lstm tuple (1, batch, lstm_hidden)
                    # lstmcell (batch, lstm_hidden)
                    batch_pred, states = self.decoder.single_step(
                        encoder_out=encoder_out,
                        sequence_step=input_tokens,
                        step_state=states,
                        step=t,
                    )

                    batch_scores[:, t, :] = batch_pred

                    # (batch, vocab_dim)
                    # output = self.activation(batch_pred)
                    # (batch,)
                    output = self.inference_func(batch_pred, dim=1)
                    # copy predicted tokens to batch_predictions
                    batch_predictions[:, t] = output

                    input_tokens = output

                scores.append(batch_scores.detach())
                predictions.append(batch_predictions.detach())
                targets.append(y.detach())
                lengths.append(y_lens.detach())

                indices.extend(inds)

                # flatten the batch scores to (*, vocab_dim) and the targets to
                # a vector
                # the loss will ignore the padding tokens
                loss = self.loss(
                    batch_scores.view(-1, self.decoder.vocab_dim), y.flatten()
                )
                accum_loss.append(loss.detach().cpu().numpy())

        average_loss = np.mean(accum_loss)

        epoch_scores = self.concat0_tensors(
            scores, batch_total=total_batch, max_variable=max_sequence
        )
        epoch_predictions = self.concat0_tensors(
            predictions, batch_total=total_batch, max_variable=max_sequence
        )
        epoch_targets = self.concat0_tensors(
            targets, batch_total=total_batch, max_variable=max_sequence
        )
        # no need to pad this one, it is a 1D tensor
        epoch_lengths = torch.cat(lengths, dim=0).cpu()
        epoch_indices = torch.tensor(indices)

        if not keep_device:
            epoch_scores = epoch_scores.cpu()
            epoch_predictions = epoch_predictions.cpu()
            epoch_targets = epoch_targets.cpu()

        return (
            epoch_scores,
            epoch_predictions,
            epoch_targets,
            epoch_lengths,
            epoch_indices,
            average_loss,
        )

    def concat0_tensors(
        self,
        tensor_list: List[torch.Tensor],
        batch_dim: int = 0,
        pad_dim: int = 1,
        batch_total: int = None,
        max_variable: int = None,
    ):
        if batch_total is None:
            batch_total = sum(t.size(batch_dim) for t in tensor_list)
        if max_variable is None:
            max_variable = max(t.size(pad_dim) for t in tensor_list)

        size = list(tensor_list[0].size())
        size[batch_dim] = batch_total
        size[pad_dim] = max_variable

        collected_tensor = tensor_list[0].new_full(
            size, fill_value=self.vocabulary.pad_token_id
        )

        current0 = 0
        for tensor in tensor_list:
            # if batch_dim=0 and pad_dim=1 then it is the same as
            # collected[current:current+tensor.size(0),:tensor.size(1)] = tensor
            collected_tensor.narrow(batch_dim, current0, tensor.size(batch_dim)).narrow(
                pad_dim, 0, tensor.size(pad_dim)
            )[:] = tensor
            current0 += tensor.size(0)

        return collected_tensor

    def log(self):
        return self.metric_logger.log()

    def get_models(self):
        return [self.encoder, self.decoder]

    def write_checkpoint(self):
        if self.write_checkpoints:
            logger.debug("Saving model checkpoint")
            data = {
                "encoder": self.encoder.state_dict(),
                "decoder": self.decoder.state_dict(),
                "vocabulary": self.vocabulary.token2id,
            }
            name = f"{self.checkpoints_name}_{self._internal_epoch}.pt"
            torch.save(data, os.path.join(self.checkpoints_path, name))

    def inference(self, data):

        #!########
        self.encoder.eval()
        self.decoder.eval()
        #!########

        max_seq_length = 100

        predictions = []
        scores = []

        with torch.no_grad():
            # for x in data:
            x: torch.Tensor = data.to(self.device)

            # (batch, encoder_dim), float
            encoder_out = self.encoder(x)

            # (batch,), int/long
            input_tokens = x.new_full(
                (x.size(0),), fill_value=self.vocabulary.start_token_id
            ).long()

            # (batch, 100), 100 is max seq length, int/long
            batch_predictions = x.new_full(
                (x.size(0), max_seq_length), fill_value=self.vocabulary.pad_token_id
            ).long()

            # (batch, 100, vocab_dim), 100 is max seq, float
            batch_scores = x.new_full(
                (x.size(0), max_seq_length, self.decoder.vocab_dim),
                fill_value=self.vocabulary.pad_token_id,
            )

            # tuple
            # lstm (1, batch, lstm_hidden)
            # lstmcell (batch, lstm_hidden)
            states = self.decoder.init_hidden_state(encoder_out=encoder_out)

            for t in range(max_seq_length):
                # batch_pred: (batch, vocab_dim)
                # states
                # lstm tuple (1, batch, lstm_hidden)
                # lstmcell (batch, lstm_hidden)
                batch_pred, states = self.decoder.single_step(
                    encoder_out=encoder_out,
                    sequence_step=input_tokens,
                    step_state=states,
                    step=t,
                )

                batch_scores[:, t, :] = batch_pred

                # (batch, vocab_dim)
                # output = self.activation(batch_pred)
                # (batch,)
                output = self.inference_func(batch_pred, dim=1)
                # copy predicted tokens to batch_predictions
                batch_predictions[:, t] = output

                input_tokens = output

            scores.append(batch_scores.detach())
            predictions.append(batch_predictions.detach())

        inference_predictions = torch.cat(predictions, dim=0)
        inference_scores = torch.cat(scores, dim=0)

        return inference_predictions, inference_scores
