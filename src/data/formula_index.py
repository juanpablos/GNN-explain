import json
from typing import Dict

from src.graphs.foc import *
from src.graphs.foc import Element


class FormulaMapping:
    def __init__(self, file="formulas.json"):
        with open(file) as f:
            self.mapping: Dict[str, str] = json.load(f)

    def __getitem__(self, key: str) -> Element:
        formula: Element = eval(self.mapping[key])
        return formula.validate()

    def __len__(self):
        return len(self.mapping)

    def __iter__(self):
        for key in self.mapping.keys():
            yield key, self[key]
