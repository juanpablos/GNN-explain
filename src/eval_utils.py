import logging
from typing import DefaultDict, Dict, Union

import torch
from torch.utils.data import DataLoader

from src.data.datasets import (
    LabeledDataset,
    LabeledSubset,
    NetworkDatasetCollectionWrapper
)
from src.graphs.foc import Element
from src.training.mlp_training import Training
from src.typing import S, T

logger = logging.getLogger(__name__)


def evaluate_model(model: torch.nn.Module,
                   test_data: Union[LabeledSubset[T, S],
                                    LabeledDataset[T, S]],
                   reconstruction: NetworkDatasetCollectionWrapper[S],
                   trainer: Training,
                   multilabel: bool,
                   gpu: int = 0):

    logger.debug("Evaluating model")

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

    if isinstance(test_data, LabeledDataset):
        test_indices = list(range(len(test_data)))
    else:
        test_indices = test_data.indices

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
