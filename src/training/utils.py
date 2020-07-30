
from typing import Optional

from src.typing import StopFormat


class StopTraining:
    def __init__(self, conditions: Optional[StopFormat]):
        if conditions is None:
            self.operation = None
        else:
            if "operation" not in conditions:
                raise ValueError("`conditions` must have key 'operation'")

            if conditions["operation"] not in ["and", "or"]:
                raise ValueError("`operation` must be either 'and' or 'or'")

            self.operation = all if conditions["operation"] == "and" else any
            self.conditions = conditions["conditions"]
            self.check = True
            self.stop = False
            self.stay = conditions.get("stay", 1)
            self.original_stay = conditions.get("stay", 1)

    def __call__(self, **kwargs):
        if self.operation is None:
            return False

        if self.check:
            if not all(cond in kwargs for cond in self.conditions):
                _conds = [c for c in self.conditions if c not in kwargs]
                raise ValueError(
                    "Not all selected metrics are available "
                    f"from the training: {_conds}")
            self.check = False

        current_state = [
            kwargs[cond] >= value for cond,
            value in self.conditions.items()]

        if self.operation(current_state):
            self.stay -= 1
        else:
            self.stay = self.original_stay

        return self.stay < 0
