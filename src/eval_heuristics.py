from abc import ABC, abstractmethod
from typing import Dict, Union

import torch
from torch.functional import Tensor

from src.graphs.foc import *
from src.graphs.foc import Element

__all__ = [
    "EvalHeuristic",
    "SingleFormulaHeuristic",
    "MaxSumFormulaHeuristic",
    "MinSumFormulaHeuristic",
    "MaxDiffSumFormulaHeuristic",
]

INDEX_TO_PROPERTY = {
    0: Property("RED"),
    1: Property("BLUE"),
    2: Property("GREEN"),
    3: Property("BLACK"),
}


def _get_embedding_weights_from_tensor(weights: Tensor) -> Tensor:
    # the first 4 x 8 weights corresponds to the embedding layer
    # 4 inputs 8 outputs
    embedding_weights = weights[: 4 * 8]
    # then reshape the weights to 8x4
    return embedding_weights.reshape(8, 4)


class EvalHeuristic(ABC):
    @abstractmethod
    def predict(self, weights: Union[Tensor, Dict[str, Tensor]]) -> FOC:
        ...

    def __str__(self):
        return self.__class__.__name__


class SingleFormulaHeuristic(EvalHeuristic):
    def __init__(self, formula: Union[Element, FOC]):
        if isinstance(formula, Element):
            formula = FOC(formula)
        self.formula = formula

    def predict(self, weights: Union[Tensor, Dict[str, Tensor]]) -> FOC:
        return self.formula

    def __str__(self):
        return f"{self.__class__.__name__}({self.formula.get_hash()})"


class WeightHeuristic(EvalHeuristic):
    def get_embedding_weights(
        self, weights: Union[Tensor, Dict[str, Tensor]]
    ) -> Tensor:
        if isinstance(weights, Tensor):
            return _get_embedding_weights_from_tensor(weights)
        return weights["input_embedding.weight"]


class MaxSumFormulaHeuristic(WeightHeuristic):
    def predict(self, weights: Union[Tensor, Dict[str, Tensor]]) -> FOC:
        embedding_weights = self.get_embedding_weights(weights)
        per_input = embedding_weights.sum(0)
        color_index: int = torch.argmax(per_input).item()

        return FOC(INDEX_TO_PROPERTY[color_index])


class MinSumFormulaHeuristic(WeightHeuristic):
    def predict(self, weights: Union[Tensor, Dict[str, Tensor]]) -> FOC:
        embedding_weights = self.get_embedding_weights(weights)
        per_input = embedding_weights.sum(0)
        color_index: int = torch.argmin(per_input).item()

        return FOC(INDEX_TO_PROPERTY[color_index])


class MaxDiffSumFormulaHeuristic(WeightHeuristic):
    def predict(self, weights: Union[Tensor, Dict[str, Tensor]]) -> FOC:
        embedding_weights = self.get_embedding_weights(weights)
        per_input = embedding_weights.sum(0)

        avg_value = torch.mean(per_input)
        diffs = torch.abs(per_input - avg_value)

        color_index: int = torch.argmax(diffs).item()

        return FOC(INDEX_TO_PROPERTY[color_index])
