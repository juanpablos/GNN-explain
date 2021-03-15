from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from src.graphs.foc import (
    AND,
    NEG,
    OR,
    Element,
    Exist,
    ForAll,
    Operator,
    Property,
    Role,
)

T_co = TypeVar("T_co", covariant=True)


class Visitor(ABC, Generic[T_co]):
    result: T_co

    def _visit_Operator(self, node: Operator):
        if isinstance(node, NEG):
            node.expression._accept(self)
        else:
            for el in node.operands:
                el._accept(self)

    def _visit_AND(self, node: AND):
        self._visit_Operator(node)

    def _visit_NEG(self, node: NEG):
        self._visit_Operator(node)

    def _visit_OR(self, node: OR):
        self._visit_Operator(node)

    def _visit_Exist(self, node: Exist):
        node.expression._accept(self)

    def _visit_ForAll(self, node: ForAll):
        node.expression._accept(self)

    def _visit_Property(self, node: Property):
        pass

    def _visit_Role(self, node: Role):
        pass

    def __call__(self, node: Element) -> T_co:
        node._accept(self)
        self.process(node)
        res = self.result
        self.reset()
        return res

    @abstractmethod
    def reset(self):
        ...

    def process(self, formula: Element):
        pass

    @abstractmethod
    def __str__(self):
        ...
