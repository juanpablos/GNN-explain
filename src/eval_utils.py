from typing import DefaultDict

import torch
from torch.utils.data import DataLoader

from src.data.datasets import LabeledSubset, NetworkDatasetCollectionWrapper
from src.graphs.foc import Element
from src.training.mlp_training import Training
from src.typing import S_co, T_co


def evaluate_model(model: torch.nn.Module,
                   test_data: LabeledSubset[T_co, S_co],
                   reconstruction: NetworkDatasetCollectionWrapper[S_co],
                   trainer: Training,
                   additional_info: bool,
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
    if additional_info:
        test_indices = test_data.indices
        for true, pred, index in zip(y_true, y_pred, test_indices):
            formula = reconstruction[index]
            if true != pred:
                mistakes[formula] += 1
            formula_count[formula] += 1

    return y_true, y_pred, mistakes, formula_count
