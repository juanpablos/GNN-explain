from typing import List

from src.data.formulas.visitor import Visitor
from src.graphs.foc import Element, Exist, Property


class ElementExtractor(Visitor[List[Element]]):
    def __init__(self):
        super().__init__()
        self.result = []

    def reset(self):
        self.result = []


class ColorsExtractor(ElementExtractor):
    """
    Extract colors in the selected position
    """

    def __init__(self, hop: int):
        super().__init__()
        if hop < 0:
            raise ValueError("Selected hop must be a positive number")

        self.current_hop = 0
        self.target_hop = hop

    def _visit_Exist(self, node: Exist):
        self.current_hop += 1
        super()._visit_Exist(node)
        self.current_hop -= 1

    def _visit_Property(self, node: Property):
        if self.current_hop == self.target_hop:
            self.result.append(node)

    def reset(self):
        super().reset()
        self.current_hop = 0

    def __str__(self):
        return f"ColorsExtractor"
