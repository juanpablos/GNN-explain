import logging
from collections import defaultdict
from typing import Any, DefaultDict, Dict, List, Union

import torch
from torch.utils.data import DataLoader

from src.data.auxiliary import NetworkDatasetCollectionWrapper
from src.data.datasets import (
    LabeledDataset,
    LabeledSubset,
    TextSequenceDataset
)
from src.graphs.foc import Element
from src.training.gnn_sequence import GraphSequenceTrainer
from src.training.mlp_training import MLPTrainer
from src.training.sequence_training import RecurrentTrainer
from src.typing import S, T

logger = logging.getLogger(__name__)


def evaluate_model(model: torch.nn.Module,
                   test_data: Union[LabeledSubset[T, S],
                                    LabeledDataset[T, S]],
                   reconstruction: NetworkDatasetCollectionWrapper,
                   trainer: MLPTrainer,
                   multilabel: bool,
                   gpu: int = 0):

    logger.debug("Evaluating model")

    # FIX: we can refactor this into MLPTrainer
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{gpu}")
    else:
        device = torch.device("cpu")

    data_loader = DataLoader(
        test_data,
        batch_size=1024,
        pin_memory=False,
        shuffle=False,
        num_workers=0)

    y_true = []
    y_pred = []
    for x, y in data_loader:
        x = x.to(device)

        with torch.no_grad():
            output = model(x)

        output = trainer.activation(output)
        _y_pred = trainer.inference(output)

        y_true.extend(y.tolist())
        y_pred.extend(_y_pred.tolist())

    if isinstance(test_data, LabeledSubset):
        test_indices = test_data.indices
    else:
        test_indices = list(range(len(test_data)))

    mistakes: DefaultDict[Element, Dict[int, int]] = \
        DefaultDict(lambda: DefaultDict(int))
    formula_count: DefaultDict[Element, int] = DefaultDict(int)

    for true, pred, index in zip(y_true, y_pred, test_indices):
        formula = reconstruction[index]

        if multilabel:
            for i, (true_i, pred_i) in enumerate(zip(true, pred)):
                # only consider labels that are possitive
                if true_i == 1 and true_i != pred_i:
                    mistakes[formula][i] += 1
        else:
            if true != pred:
                mistakes[formula][true] += 1

        formula_count[formula] += 1

    return y_true, y_pred, mistakes, formula_count


def evaluate_text_model(trainer: Union[RecurrentTrainer, GraphSequenceTrainer],
                        test_data: Union[TextSequenceDataset[T],
                                         LabeledSubset[T, torch.Tensor]],
                        reconstruction: NetworkDatasetCollectionWrapper):

    def get_metrics(
            eval_scores,
            eval_predictions,
            eval_targets,
            eval_lengths,
            eval_indices):

        _metrics = {
            "token_acc1": trainer.metrics.token_accuracy(
                scores=eval_scores,
                targets=eval_targets,
                k=1,
                lengths=eval_lengths),
            "token_acc3": trainer.metrics.token_accuracy(
                scores=eval_scores,
                targets=eval_targets,
                k=3,
                lengths=eval_lengths),
            "sent_acc": trainer.metrics.sentence_accuracy(
                predictions=eval_predictions,
                targets=eval_targets,
                lengths=eval_lengths),
            "bleu4": trainer.metrics.bleu_score(
                predictions=eval_predictions,
                targets=eval_targets,
                lengths=eval_lengths)
        }

        valid_metric, formulas = trainer.metrics.syntax_check(
            predictions=eval_predictions,
            run_all=True,
            return_formulas=True)
        _metrics["valid"] = valid_metric

        semval = trainer.metrics.semantic_validation(
            predictions=eval_predictions,
            indices=eval_indices,
            formulas=formulas
        )
        for metric_name, value in semval.items():
            _metrics[f"semval{metric_name}"] = value

        return _metrics

    data_loader = trainer.init_dataloader(
        data=test_data,
        mode=None,
        batch_size=1024,
        pin_memory=False,
        shuffle=False,
        num_workers=0
    )

    scores, predictions, targets, lengths, indices, _ = trainer.run_pass(
        data_loader, keep_device=True)

    if isinstance(test_data, LabeledSubset):
        test_indices = test_data.indices
    else:
        test_indices = list(range(len(test_data)))

    formula_samples: Dict[Element, Dict[str, List]] = defaultdict(
        lambda: {
            "predictions": [],
            "scores": [],
            "targets": [],
            "lengths": [],
            "indices": []})

    for i, test_ind in enumerate(test_indices):
        formula_col = formula_samples[reconstruction[test_ind]]

        formula_col["scores"].append(scores[i])
        formula_col["predictions"].append(predictions[i])
        formula_col["targets"].append(targets[i])
        formula_col["lengths"].append(lengths[i])
        formula_col["indices"].append(indices[i])

    # metric_name -> all|formula -> value
    formula_metrics: Dict[str, Dict[str, Any]] = {}

    whole_dataset_metrics = get_metrics(
        scores, predictions, targets, lengths, indices)
    for metric_name, metric_value in whole_dataset_metrics.items():
        formula_metrics.setdefault(metric_name, {})["all"] = metric_value

    for formula, metrics in formula_samples.items():
        _scores = torch.stack(metrics["scores"])
        _predictions = torch.stack(metrics["predictions"])
        _targets = torch.stack(metrics["targets"])
        _lengths = torch.tensor(metrics["lengths"])
        _indices = torch.tensor(metrics["indices"])

        single_metrics = get_metrics(
            _scores, _predictions, _targets, _lengths, _indices)
        for metric_name, metric_value in single_metrics.items():
            formula_metrics.setdefault(metric_name,
                                       {})[repr(formula)] = metric_value

    return formula_metrics
