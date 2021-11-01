from abc import ABC, abstractmethod
from typing import Dict, Generic, List, TypeVar, Union

import torch
from torch.functional import Tensor

from src.data.formulas.extract import ColorsExtractor
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

T_co = TypeVar("T_co", covariant=True)


def _get_embedding_weights_from_tensor(weights: Tensor) -> Tensor:
    # the first 4 x 8 weights corresponds to the embedding layer
    # 4 inputs 8 outputs
    embedding_weights = weights[: 4 * 8]
    # then reshape the weights to 8x4
    return embedding_weights.reshape(8, 4)


class EvalHeuristic(ABC, Generic[T_co]):
    @abstractmethod
    def predict(self, weights: Union[Tensor, Dict[str, Tensor]]) -> FOC:
        ...

    @abstractmethod
    def extract_valid_elements(self, formula: FOC) -> List[T_co]:
        ...

    @abstractmethod
    def match(self, candicate_value: FOC, allowed_values: List[T_co]) -> bool:
        ...

    def __str__(self):
        return self.__class__.__name__


class SingleFormulaHeuristic(EvalHeuristic[FOC]):
    def __init__(self, formula: Union[Element, FOC]):
        if isinstance(formula, Element):
            formula = FOC(formula)
        self.formula = formula

    def predict(self, **kwargs) -> FOC:
        return self.formula

    def extract_valid_elements(self, **kwargs) -> List[FOC]:
        return [self.formula]

    def match(self, candicate_value: FOC, **kwargs) -> bool:
        return self.formula == candicate_value

    def __str__(self):
        return f"{self.__class__.__name__}({self.formula.get_hash()})"


class FirstColorHeuristic(EvalHeuristic[Element]):
    def get_embedding_weights(
        self, weights: Union[Tensor, Dict[str, Tensor]]
    ) -> Tensor:
        if isinstance(weights, Tensor):
            return _get_embedding_weights_from_tensor(weights)
        return weights["input_embedding.weight"]

    def extract_valid_elements(self, formula: FOC) -> List[Element]:
        return ColorsExtractor(hop=0)(formula.expression)

    def match(
        self, candicate_value: FOC, allowed_values: List[Element], **kwargs
    ) -> bool:
        return candicate_value.expression in allowed_values


class MaxSumFormulaHeuristic(FirstColorHeuristic):
    def predict(self, weights: Union[Tensor, Dict[str, Tensor]]) -> FOC:
        embedding_weights = self.get_embedding_weights(weights)
        per_input = embedding_weights.sum(0)
        color_index: int = torch.argmax(per_input).item()

        return FOC(INDEX_TO_PROPERTY[color_index])


class MinSumFormulaHeuristic(FirstColorHeuristic):
    def predict(self, weights: Union[Tensor, Dict[str, Tensor]]) -> FOC:
        embedding_weights = self.get_embedding_weights(weights)
        per_input = embedding_weights.sum(0)
        color_index: int = torch.argmin(per_input).item()

        return FOC(INDEX_TO_PROPERTY[color_index])


class MaxDiffSumFormulaHeuristic(FirstColorHeuristic):
    def predict(self, weights: Union[Tensor, Dict[str, Tensor]]) -> FOC:
        embedding_weights = self.get_embedding_weights(weights)
        per_input = embedding_weights.sum(0)

        avg_value = torch.mean(per_input)
        diffs = torch.abs(per_input - avg_value)

        color_index: int = torch.argmax(diffs).item()

        return FOC(INDEX_TO_PROPERTY[color_index])
