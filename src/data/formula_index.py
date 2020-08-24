import json
from typing import Dict

from src.graphs.foc import *


class FormulaMapping:
    def __init__(self, file="formulas.json"):
        with open(file) as f:
            self.mapping: Dict[str, str] = json.load(f)

    def __getitem__(self, key: str) -> FOC:
        return FOC(eval(self.mapping[key]))
