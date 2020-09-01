from typing import DefaultDict, Union

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


def evaluate_model(model: torch.nn.Module,
                   test_data: Union[LabeledSubset[T, S],
                                    LabeledDataset[T, S]],
                   reconstruction: NetworkDatasetCollectionWrapper[S],
                   trainer: Training,
                   multi_label: bool,
                   gpu: int = 0):

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
        _, _y_pred = output.max(dim=1)

        y_true.extend(y.tolist())
        y_pred.extend(_y_pred.tolist())

    mistakes: DefaultDict[Element, int] = DefaultDict(int)
    formula_count: DefaultDict[Element, int] = DefaultDict(int)

    if not multi_label:
        if isinstance(test_data, LabeledDataset):
            test_indices = list(range(len(test_data)))
        else:
            test_indices = test_data.indices

        for true, pred, index in zip(y_true, y_pred, test_indices):
            formula = reconstruction[index]
            if true != pred:
                mistakes[formula] += 1
            formula_count[formula] += 1

    return y_true, y_pred, mistakes, formula_count
