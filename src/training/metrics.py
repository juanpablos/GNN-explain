import logging
from typing import List, Literal, Optional, Tuple, overload

import numpy as np
import torch
import torch.utils.data
from nltk.translate.bleu_score import corpus_bleu

from src.data.auxiliary import FormulaAppliedDatasetWrapper
from src.data.vocabulary import Vocabulary
from src.graphs.foc import FOC
from src.training.check_formulas import FormulaReconstruction

logger = logging.getLogger(__name__)


def _single_validation(index, formula, formula_mapping, cached_formula_evaluations):
    assert formula_mapping is not None, "Formula mapping cannot be None"
    correct = formula_mapping[index]

    tp: float = 0.0
    tn: float = 0.0
    fp: float = 0.0
    fn: float = 0.0

    if formula is not None:
        formula_str = repr(formula)
        if formula_str in cached_formula_evaluations:
            pred = cached_formula_evaluations[formula_str]
        else:
            # run the formula over the predefined set of graphs
            # ! this takes ~0.01 sec per formula
            pred = formula_mapping.run_formula(formula)
            cached_formula_evaluations[formula_str] = pred

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


class SequenceMetrics:
    def __init__(
        self,
        vocabulary: Vocabulary,
        result_mapping: FormulaAppliedDatasetWrapper = None,
        seed: int = None,
        subset_size: float = 0.2,
    ):

        assert 0 < subset_size <= 1, "subset_size must be between 0 and 1"

        self.vocabulary = vocabulary
        self.formula_reconstruction = FormulaReconstruction(vocabulary)
        self.formula_mapping = result_mapping

        self.seed = seed
        self.subset_size = subset_size

        self.cached_formulas = None
        self.cached_indices = None

        # self.shared_memory_manager = Manager()
        # self.cached_formula_evaluations = self.shared_memory_manager.dict()
        self.cached_formula_evaluations = {}

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
            matches, lengths, batch_first=True, enforce_sorted=False
        )
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
            predictions, lengths, batch_first=True, enforce_sorted=False
        )

        # predictions have the same size with targets, but when removing the
        # extra values of prediction and padding again the paddings are of
        # length batch_length, that is not necessarily the max_len of the data
        # so we have to extend with the extra bit that was removed with
        # total_length.
        padded, _ = torch.nn.utils.rnn.pad_packed_sequence(
            cleaned, batch_first=True, total_length=predictions.size(1)
        )

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
        subset_indices = local_rand.choice(data_size, size, replace=False)
        return subset_indices

    @overload
    def syntax_check(
        self, predictions, run_all: bool = ..., return_formulas: Literal[False] = ...
    ) -> float:
        ...

    @overload
    def syntax_check(
        self, predictions, run_all: bool = ..., return_formulas: Literal[True] = ...
    ) -> Tuple[float, List[Optional[FOC]]]:
        ...

    def syntax_check(
        self, predictions, run_all: bool = False, return_formulas: bool = False
    ):
        # predictions: indices (batch, L) with padding

        logger.debug("Running syntax check")
        # ! run_all does not save in cache

        subset_indices = self.get_random_indices(predictions.size(0))

        if run_all:
            predictions = predictions.tolist()
        else:
            predictions = predictions[subset_indices].tolist()

        # compile the formula into a FOC object
        formulas, correct = self.formula_reconstruction.batch2expression(predictions)

        if not run_all:
            self.cached_formulas = formulas
            self.cached_indices = subset_indices

        valid_syntax = float(correct) / len(predictions)

        if return_formulas:
            return valid_syntax, formulas
        return valid_syntax

    @staticmethod
    def _div(a: float, b: float):
        try:
            return a / b
        except BaseException:
            # returns 0 because it will always be
            # a / (a + x), so if a+x=b=0, then a is also 0
            # when a and x are positive numbers
            return 0.0

    def semantic_validation(
        self, predictions, indices, formulas: List[Optional[FOC]] = None
    ):
        # predictions: indices (batch, L) with padding

        logger.debug("Running syntax validation")

        if formulas is None:
            if self.cached_formulas is None:
                subset_indices = self.get_random_indices(predictions.size(0))
                predictions = predictions[subset_indices].tolist()

                formulas, _ = self.formula_reconstruction.batch2expression(predictions)
                indices = indices[subset_indices]
            else:
                formulas = self.cached_formulas
                indices = indices[self.cached_indices]

        indices = indices.tolist()

        assert len(formulas) == len(
            indices
        ), "formulas and indices dont have the same length"

        tp: float = 0.0
        tn: float = 0.0
        fp: float = 0.0
        fn: float = 0.0
        for a, b in zip(indices, formulas):
            _tp, _tn, _fp, _fn = _single_validation(
                a, b, self.formula_mapping, self.cached_formula_evaluations
            )

            tp += _tp
            tn += _tn
            fp += _fp
            fn += _fn
        # with Pool(4) as p:
        #     indicators = p.starmap(
        #         _single_validation,
        #         zip(
        #             indices,
        #             formulas,
        #             repeat(self.formula_mapping),
        #             repeat(self.cached_formula_evaluations),
        #         ),
        #         chunksize=len(formulas) // 4,
        #     )

        # tp: float = 0.0
        # tn: float = 0.0
        # fp: float = 0.0
        # fn: float = 0.0
        # for _tp, _tn, _fp, _fn in indicators:
        #     tp += _tp
        #     tn += _tn
        #     fp += _fp
        #     fn += _fn

        precision = self._div(tp, tp + fp)
        recall = self._div(tp, tp + fn)
        acc = self._div(tp + tn, tp + tn + fp + fn)

        return {"PRE": precision, "REC": recall, "ACC": acc}
