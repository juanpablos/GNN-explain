from typing import List, Optional

from zss import Node

from src.data.formulas.visitor import Visitor
from src.graphs.foc import AND, Element, NEG, OR, Exist, Operator, Property, Role


class FOC2Node(Visitor[Node]):
    def __init__(self):
        self.result = Node("_EMPTY_")

    def reset(self):
        self.result = Node("_EMPTY_")

    def _visit_Operator(self, node: Operator) -> List[Node]:
        if isinstance(node, NEG):
            node.expression._accept(self)
            return [self.result]
        else:
            operands = []
            for el in node.operands:
                el._accept(self)
                operands.append(self.result)
            return operands

    def _visit_AND(self, node: AND):
        operands = self._visit_Operator(node)
        self.result = Node("AND", children=operands)

    def _visit_OR(self, node: OR):
        operands = self._visit_Operator(node)
        self.result = Node("OR", children=operands)

    def _visit_NEG(self, node: NEG):
        super()._visit_NEG(node)
        self.result = Node("NEG", children=[self.result])

    def _visit_Exist(self, node: Exist):
        super()._visit_Exist(node)
        self.result = Node(
            "Exist",
            children=[Node(str(node.lower)), Node(str(node.upper)), self.result],
        )

    def _visit_Property(self, node: Property):
        super()._visit_Property(node)
        self.result = Node(node.name)

    def _visit_Role(self, node: Role):
        super()._visit_Role(node)
        self.result = Node(node.name)

    def __call__(self, node: Optional[Element]):
        if node is None:
            self.reset()
            return self.result
        return super().__call__(node)

    def __str__(self):
        return "FOC2Node"
