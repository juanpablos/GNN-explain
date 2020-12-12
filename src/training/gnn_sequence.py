import logging
from itertools import chain
from multiprocessing import Pool
from typing import List, Literal, Optional, Tuple, Union, overload

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from nltk.translate.bleu_score import corpus_bleu
from torch.nn.utils.rnn import pad_sequence
from torch_geometric.data.dataloader import Collater

from src.data.auxiliary import FormulaAppliedDatasetWrapper
from src.data.vocabulary import Vocabulary
from src.graphs.foc import FOC
from src.models import LSTMCellDecoder, LSTMDecoder
from src.models.ac_gnn import NetworkACGNN
from src.training.check_formulas import FormulaReconstruction

from . import Trainer

logger = logging.getLogger(__name__)
logger_metrics = logging.getLogger('metrics')

# TODO: duplicated code with sequence_training
# FIX: implement Mixin or utils to add metrics and other functionalities


class Collator:
    def __init__(self, pad_token_id: int, follow_batch):
        self.pad_token_id = pad_token_id
        self.geometric_collator = Collater(follow_batch)

    def __call__(self, batch):
        # x: list of Data
        # y: Tensor
        # inds: tuple of numbers
        x, y, inds = zip(*batch)

        x = self.geometric_collator(x)

        y_lens = torch.tensor([target.size(0) for target in y])
        y_pad = pad_sequence(
            y,
            batch_first=True,
            padding_value=self.pad_token_id)

        return x, y_pad, y_lens, inds


class DataLoader(torch.utils.data.DataLoader):
    def __init__(self,
                 dataset,
                 pad_token_id,
                 batch_size=1,
                 shuffle=False,
                 follow_batch=[],
                 **kwargs):
        super(DataLoader, self).__init__(
            dataset,
            batch_size,
            shuffle,
            collate_fn=Collator(
                pad_token_id=pad_token_id,
                follow_batch=follow_batch),
            **kwargs)


class Metric:
    def __init__(
            self,
            vocabulary: Vocabulary,
            result_mapping: FormulaAppliedDatasetWrapper,
            seed: int = None,
            subset_size: float = 0.2):

        assert 0 < subset_size <= 1, 'subset_size must be between 0 and 1'

        self.vocabulary = vocabulary
        self.formula_reconstruction = FormulaReconstruction(vocabulary)
        self.formula_mapping = result_mapping

        self.seed = seed
        self.subset_size = subset_size

        self.cached_formulas = None
        self.cached_indices = None

    def token_accuracy(self, scores, targets, k, lengths):
        # scores: logits (batch, L, vocab) with padding
        # targets: indices (batch, L) with padding
        # k: int
        # lengths: (batch,)

        # (batch, L, k)
        _, indices = scores.topk(k, dim=2, largest=True, sorted=False)
        # expand the targets to check if they occur in one of the topk
        _expanded = targets.unsqueeze(dim=2).expand_as(indices)

        # check if the correct index is in one of the topk
        # matches: (batch, L)
        matches = torch.any(indices.eq(_expanded), dim=2)

        # flatten, but ignore the padding
        clean_flatten = torch.nn.utils.rnn.pack_padded_sequence(
            matches,
            lengths,
            batch_first=True,
            enforce_sorted=False)
        # sum all correct
        correct = clean_flatten.data.sum().float().item()

        # average over predictions that have the correct index in the topk
        return correct / lengths.sum().float().item()

    def sentence_accuracy(self, predictions: torch.Tensor, targets, lengths):
        # predictions: indices (batch, L) with padding
        # targets: indices (batch, L) with padding
        # lengths: (batch,)

        # this deletes the extra predictions and replaces them with padding
        cleaned = torch.nn.utils.rnn.pack_padded_sequence(
            predictions,
            lengths,
            batch_first=True,
            enforce_sorted=False)

        # predictions have the same size with targets, but when removing the
        # extra values of prediction and padding again the paddings are of
        # length batch_length, that is not necessarily the max_len of the data
        # so we have to extend with the extra bit that was removed with
        # total_length.
        padded, _ = torch.nn.utils.rnn.pad_packed_sequence(
            cleaned, batch_first=True, total_length=predictions.size(1))

        # option 2
        # correct_padded = torch.full_like(
        #     predictions, fill_value=self.pad_token_id)
        # correct_padded[:, :padded.size(1)] = padded

        # option 3
        # padded = torch.nn.functional.pad(
        #     padded,
        #     [0, targets.size(1) - padded.size(1)],
        #     mode="constant",
        #     value=self.pad_token_id)

        return padded.eq(targets).all(dim=1).float().mean().item()

    def bleu_score(self, predictions, targets, lengths):
        # use indices instead of string tokens
        # predictions: indices (batch, L) with padding
        # targets: indices (batch, L) with padding

        # converting everything into a list first is faster than indexing the
        # tensors
        predictions = predictions.tolist()
        targets = targets.tolist()
        lengths = lengths.tolist()

        references = []
        hypothesis = []
        for i, l in enumerate(lengths):
            references.append([targets[i][:l]])
            hypothesis.append(predictions[i][:l])

        return corpus_bleu(references, hypothesis)

    def get_random_indices(self, data_size):
        local_rand = np.random.default_rng(self.seed)
        size = int(data_size * self.subset_size)
        subset_indices = local_rand.choice(
            data_size, size, replace=False)
        return subset_indices

    @overload
    def syntax_check(
        self,
        predictions,
        run_all: bool = ...,
        return_formulas: Literal[False] = ...) -> float: ...

    @overload
    def syntax_check(
        self,
        predictions,
        run_all: bool = ...,
        return_formulas: Literal[True] = ...
    ) -> Tuple[float, List[Optional[FOC]]]: ...

    def syntax_check(
            self,
            predictions,
            run_all: bool = False,
            return_formulas: bool = False):
        # predictions: indices (batch, L) with padding

        logger.debug("Running syntax check")
        # ! run_all does not save in cache

        subset_indices = self.get_random_indices(predictions.size(0))

        if run_all:
            predictions = predictions.tolist()
        else:
            predictions = predictions[subset_indices].tolist()

        # compile the formula into a FOC object
        formulas, correct = self.formula_reconstruction.batch2expression(
            predictions)

        if not run_all:
            self.cached_formulas = formulas
            self.cached_indices = subset_indices

        valid_syntax = float(correct) / len(predictions)

        if return_formulas:
            return valid_syntax, formulas
        return valid_syntax

    def _single_validation(self, index, formula):
        correct = self.formula_mapping[index]

        tp: float = 0.
        tn: float = 0.
        fp: float = 0.
        fn: float = 0.

        if formula is not None:
            # run the formula over the predefined set of graphs
            # ! this takes ~0.01 sec per formula
            pred = self.formula_mapping.run_formula(formula)

            # get the matching indices
            matching = correct == pred
            matching_select = correct[matching]

            # positive matching
            tp_sum = (matching_select == 1).sum()

            # positives
            true_sum = (correct == 1).sum()
            # predicted positives
            pred_sum = (pred == 1).sum()

            fp = pred_sum - tp_sum
            fn = true_sum - tp_sum
            tp = tp_sum
            tn = correct.shape[0] - tp - fp - fn

        # metric for valid formulas. Invalid formulas are not considered
        return tp, tn, fp, fn

    @staticmethod
    def _div(a: float, b: float):
        try:
            return a / b
        except ZeroDivisionError:
            return 0.

    def semantic_validation(self,
                            predictions,
                            indices,
                            formulas: List[Optional[FOC]] = None):
        # predictions: indices (batch, L) with padding

        logger.debug("Running syntax validation")

        if formulas is None:
            if self.cached_formulas is None:
                subset_indices = self.get_random_indices(predictions.size(0))
                predictions = predictions[subset_indices].tolist()

                formulas, _ = self.formula_reconstruction.batch2expression(
                    predictions)
                indices = indices[subset_indices]
            else:
                formulas = self.cached_formulas
                indices = indices[self.cached_indices]

        indices = indices.tolist()

        assert len(formulas) == len(
            indices), 'formulas and indices dont have the same length'

        with Pool(4) as p:
            indicators = p.starmap(self._single_validation, zip(
                indices, formulas), chunksize=len(formulas) // 4)

        tp: float = 0.
        tn: float = 0.
        fp: float = 0.
        fn: float = 0.
        for _tp, _tn, _fp, _fn in indicators:
            tp += _tp
            tn += _tn
            fp += _fp
            fn += _fn

        precision = self._div(tp, tp + fp)
        recall = self._div(tp, tp + fn)
        acc = self._div(tp + tn, tp + tn + fp + fn)

        return {
            "PRE": precision,
            "REC": recall,
            "ACC": acc
        }


class GraphSequenceTrainer(Trainer):
    loss: nn.Module
    encoder: NetworkACGNN
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

    def __init__(self,
                 vocabulary: Vocabulary,
                 target_apply_mapping: FormulaAppliedDatasetWrapper,
                 seed: int = None,
                 subset_size: float = 0.2,
                 logging_variables: Union[Literal["all"], List[str]] = "all"):

        super().__init__(seed=seed, logging_variables=logging_variables)
        self.vocabulary = vocabulary
        self.metrics = Metric(
            seed=seed,
            subset_size=subset_size,
            vocabulary=vocabulary,
            result_mapping=target_apply_mapping)

        logger_metrics.info(",".join(self.metric_logger.keys()))

    def activation(self, output, dim=1):
        return torch.log_softmax(output, dim=dim)

    def inference(self, output, dim=1):
        _, y_pred = output.max(dim=dim)
        return y_pred

    def init_encoder(self,
                     *,
                     hidden_dim: int,
                     output_dim: int,
                     num_layers: int,
                     **kwargs):
        self.encoder = NetworkACGNN(hidden_dim=hidden_dim,
                                    output_dim=output_dim,
                                    mlp_layers=num_layers,
                                    **kwargs)

        return self.encoder

    def init_decoder(self,
                     *,
                     name: str,
                     encoder_dim: int,
                     embedding_dim: int,
                     hidden_dim: int,
                     vocab_size: int,
                     init_state_context: bool,
                     concat_encoder_input: bool,
                     dropout_prob: float = 0.0,
                     **kwargs):

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
                pad_token_id=pad_token_id)
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
                **kwargs)
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
            reduction="mean", ignore_index=self.vocabulary.pad_token_id)
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
                        mode: Union[Literal["train"], Literal["test"], None],
                        **kwargs):
        loader = DataLoader(
            data,
            pad_token_id=self.vocabulary.pad_token_id,
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

        for data, y, y_lens, _ in self.train_loader:
            data = data.to(self.device)
            y = y.to(self.device)
            y_lens = y_lens.to(self.device)

            # (batch, encoder_dim)
            encoder_out = self.encoder(
                x=data.x,
                edge_index=data.edge_index,
                edge_weight=data.weight,
                batch=data.batch
            )
            # output: (batch * L, vocab_dim), L max seq
            # targets: (batch * L,)
            # output order is not guaranteed
            # when model is LSTMCell they are sorted by real length
            # when model is LSTM they are sorted by input order
            output, targets = self.decoder(
                encoder_out=encoder_out,
                padded_target=y,
                target_lengths=y_lens)

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

        epoch_scores, epoch_predictions, \
            epoch_targets, epoch_lengths, \
            epoch_indices, average_loss = self.run_pass(loader, keep_device=True)

        metrics = {
            "token_acc1": self.metrics.token_accuracy(
                scores=epoch_scores,
                targets=epoch_targets,
                k=1,
                lengths=epoch_lengths),
            "token_acc3": self.metrics.token_accuracy(
                scores=epoch_scores,
                targets=epoch_targets,
                k=3,
                lengths=epoch_lengths),
            "sent_acc": self.metrics.sentence_accuracy(
                predictions=epoch_predictions,
                targets=epoch_targets,
                lengths=epoch_lengths),
            "bleu4": self.metrics.bleu_score(
                predictions=epoch_predictions,
                targets=epoch_targets,
                lengths=epoch_lengths),
            "valid": self.metrics.syntax_check(predictions=epoch_predictions)
        }

        semval = self.metrics.semantic_validation(
            predictions=epoch_predictions,
            indices=epoch_indices
        )
        for metric_name, value in semval.items():
            metrics[f"semval{metric_name}"] = value

        return_metrics = {"loss": average_loss, **metrics}

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
            for data, y, y_lens, inds in dataloader:
                data = data.to(self.device)
                y = y.to(self.device)
                y_lens = y_lens.to(self.device)

                # remove the <start> token
                y = y[:, 1:]
                y_lens = y_lens - 1

                # (batch, encoder_dim)
                encoder_out = self.encoder(
                    x=data.x,
                    edge_index=data.edge_index,
                    edge_weight=data.weight,
                    batch=data.batch
                )

                # (batch,)
                input_tokens = y.new_full(
                    (y.size(0),),
                    fill_value=self.vocabulary.start_token_id)

                # (batch, L), L is max seq
                batch_predictions = y.new_full(
                    (y.size(0), y.size(1)),
                    fill_value=self.vocabulary.pad_token_id)

                # (batch, L, vocab_dim), L is max seq
                batch_scores = data.x.new_full(
                    (y.size(0), y.size(1), self.decoder.vocab_dim),
                    fill_value=self.vocabulary.pad_token_id)

                total_batch += batch_scores.size(0)
                max_sequence = max(max_sequence, batch_scores.size(1))

                # tuple
                # lstm (1, batch, lstm_hidden)
                # lstmcell (batch, lstm_hidden)
                states = self.decoder.init_hidden_state(
                    encoder_out=encoder_out)

                for t in range(y.size(1)):
                    # batch_pred: (batch, vocab_dim)
                    # states
                    # lstm tuple (1, batch, lstm_hidden)
                    # lstmcell (batch, lstm_hidden)
                    batch_pred, states = self.decoder.single_step(
                        encoder_out=encoder_out,
                        sequence_step=input_tokens,
                        step_state=states,
                        step=t
                    )

                    batch_scores[:, t, :] = batch_pred

                    # (batch, vocab_dim)
                    # output = self.activation(batch_pred)
                    # (batch,)
                    output = self.inference(batch_pred, dim=1)
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
                loss = self.loss(batch_scores.view(-1, self.decoder.vocab_dim),
                                 y.flatten())
                accum_loss.append(loss.detach().cpu().numpy())

        average_loss = np.mean(accum_loss)

        epoch_scores = self.concat0_tensors(
            scores,
            batch_total=total_batch,
            max_variable=max_sequence)
        epoch_predictions = self.concat0_tensors(
            predictions,
            batch_total=total_batch,
            max_variable=max_sequence)
        epoch_targets = self.concat0_tensors(
            targets,
            batch_total=total_batch,
            max_variable=max_sequence)
        # no need to pad this one, it is a 1D tensor
        epoch_lengths = torch.cat(lengths, dim=0)
        epoch_indices = torch.tensor(indices)

        if not keep_device:
            epoch_scores = epoch_scores.cpu()
            epoch_predictions = epoch_predictions.cpu()
            epoch_targets = epoch_targets.cpu()
            epoch_lengths = epoch_lengths.cpu()

        return epoch_scores, epoch_predictions, epoch_targets, epoch_lengths, epoch_indices, average_loss

    def concat0_tensors(self,
                        tensor_list: List[torch.Tensor],
                        batch_dim: int = 0,
                        pad_dim: int = 1,
                        batch_total: int = None,
                        max_variable: int = None):
        if batch_total is None:
            batch_total = sum(t.size(batch_dim) for t in tensor_list)
        if max_variable is None:
            max_variable = max(t.size(pad_dim) for t in tensor_list)

        size = list(tensor_list[0].size())
        size[batch_dim] = batch_total
        size[pad_dim] = max_variable

        collected_tensor = tensor_list[0].new_full(
            size,
            fill_value=self.vocabulary.pad_token_id)

        current0 = 0
        for tensor in tensor_list:
            # if batch_dim=0 and pad_dim=1 then it is the same as
            # collected[current:current+tensor.size(0),:tensor.size(1)] = tensor
            collected_tensor.narrow(
                batch_dim, current0, tensor.size(batch_dim)).narrow(
                pad_dim, 0, tensor.size(pad_dim))[:] = tensor
            current0 += tensor.size(0)

        return collected_tensor

    def log(self):
        return self.metric_logger.log()

    def get_models(self):
        return [self.encoder, self.decoder]
