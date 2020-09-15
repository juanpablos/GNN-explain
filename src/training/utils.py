import warnings
from collections import defaultdict
from typing import Dict, Iterator, List, Literal, Optional, Union

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
                    f"from the training: {_conds}. "
                    f"Available are: {list(kwargs)}")
            self.check = False

        current_state = [
            kwargs[cond] >= value for cond,
            value in self.conditions.items()]

        if self.operation(current_state):
            self.stay -= 1
        else:
            self.stay = self.original_stay

        return self.stay < 0


class MetricLogger:
    def __init__(self,
                 variables: Union[Literal["all"],
                                  List[str],
                                  None] = "all"):
        if variables is None:
            variables = []
        self.variables: Dict[str, List[float]] = defaultdict(list)

        self.log_all = variables == "all"
        if isinstance(variables, list):
            self._selection = variables

        self.warned = False

    def __getitem__(self, key: str):
        return self.variables[key][-1]

    def items(self, select: bool = False):
        for metric in self.keys(select=select):
            yield metric, self.variables[metric]

    def keys(self, select: bool = False) -> Iterator[str]:
        if select:
            yield from self.selection
        else:
            yield from self.variables.keys()

    def update(self, **kwargs: float):
        for name, value in kwargs.items():
            self.variables[name].append(value)

    def get_history(self, key: str):
        return self.variables[key]

    def log(self):
        if self.log_all:
            msg = self.__full_logger()
        else:
            msg = self.__select_logger()

        return msg

    @property
    def selection(self):
        if self.log_all:
            return self.variables.keys()
        else:
            return self._selection

    def __full_logger(self):
        metrics: List[str] = []
        for name, values in self.variables.items():
            last_value = values[-1]
            metrics.append(f"{name} {last_value:<10.6f}")
        return "".join(metrics)

    def __select_logger(self):
        metrics: List[str] = []
        for name in self.selection:
            try:
                last_value = self.variables[name][-1]
                metrics.append(f"{name} {last_value:<10.6f}")
            except KeyError as e:
                if not self.warned:
                    key = e.args[0]
                    warnings.warn(
                        "Not all selected metrics are available. "
                        f"{key} is not present. "
                        f"Available are: {list(self.variables)}",
                        UserWarning,
                        stacklevel=2)
                    self.warned = True

        return "".join(metrics)
