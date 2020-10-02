import logging
from collections import defaultdict
from typing import Any, DefaultDict, Dict, List, Union

import torch
from torch.utils.data import DataLoader

from src.data.datasets import (
    LabeledDataset,
    LabeledSubset,
    NetworkDatasetCollectionWrapper,
    TextSequenceDataset
)
from src.graphs.foc import Element
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


def evaluate_text_model(trainer: RecurrentTrainer,
                        test_data: Union[TextSequenceDataset[T],
                                         LabeledSubset[T, torch.Tensor]],
                        reconstruction: NetworkDatasetCollectionWrapper):

    data_loader = DataLoader(
        test_data,
        batch_size=1024,
        pin_memory=False,
        shuffle=False,
        num_workers=0)

    scores, predictions, targets, lengths, _ = trainer.run_pass(data_loader)

    if isinstance(test_data, LabeledSubset):
        test_indices = test_data.indices
    else:
        test_indices = list(range(len(test_data)))

    formula_samples: Dict[Element, Dict[str, List]] = defaultdict(
        lambda: {
            "predictions": [],
            "scores": [],
            "targets": [],
            "lengths": []})

    for i, score, pred, target, length in zip(
            test_indices, scores, predictions, targets, lengths):
        formula_col = formula_samples[reconstruction[i]]

        formula_col["scores"].append(score)
        formula_col["predictions"].append(pred)
        formula_col["targets"].append(target)
        formula_col["lengths"].append(length)

    formula_metrics: Dict[Element, Dict[str, Any]] = {}

    for formula, metrics in formula_samples.items():
        _scores = torch.cat(metrics["scores"], dim=0)
        _predictions = torch.cat(metrics["predictions"], dim=0)
        _targets = torch.cat(metrics["targets"], dim=0)
        _lengths = torch.cat(metrics["lengths"], dim=0)

        formula_metrics[formula] = {
            "token_acc1": trainer.metrics.token_accuracy(
                scores=_scores,
                targets=_targets,
                k=1,
                lengths=_lengths),
            "token_acc3": trainer.metrics.token_accuracy(
                scores=_scores,
                targets=_targets,
                k=3,
                lengths=_lengths),
            "sent_acc": trainer.metrics.sentence_accuracy(
                predictions=_predictions,
                targets=_targets,
                lengths=_lengths),
            "bleu4": trainer.metrics.bleu_score(
                predictions=_predictions,
                targets=_targets,
                lengths=_lengths)
        }

    return formula_metrics
