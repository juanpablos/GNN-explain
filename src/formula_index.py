import json

from .graphs.foc import *


class FormulaMapping:
    def __init__(self, file="formulas.json"):
        with open(file) as f:
            self.mapping = json.load(f)

    def __getitem__(self, key: str) -> FOC:
        return FOC(eval(self.mapping[key]))
