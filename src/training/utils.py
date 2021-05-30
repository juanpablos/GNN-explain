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

            if conditions["operation"] not in ["and", "or", "early"]:
                raise ValueError(
                    f"{conditions['operation']} is not a supported operation"
                )

            self.conditions = conditions["conditions"]
            self.operation = self.__get_operation(conditions["operation"])
            self.check = True
            self.stop = False
            self.stay = conditions.get("stay", 1)
            self.original_stay = conditions.get("stay", 1)

            self._auxiliary_storage = {}

    def __get_operation(self, operation):
        operations = {
            "and": lambda: self._and_condition,
            "or": lambda: self._or_condition,
            "early": self._early_condition_pre_check,
        }
        return operations[operation]()

    def _and_condition(self, **kwargs):
        current_state = [
            kwargs[cond] >= value for cond, value in self.conditions.items()
        ]
        return all(current_state)

    def _or_condition(self, **kwargs):
        current_state = [
            kwargs[cond] >= value for cond, value in self.conditions.items()
        ]
        return any(current_state)

    def _early_condition(self, **kwargs):
        condition, tolerance = next(iter(self.conditions.items()))
        current_value = kwargs[condition]

        decreasing = True
        last_low = self._auxiliary_storage.get(condition)

        if last_low is None or last_low - current_value >= tolerance:
            # its not yet set, or it is decreasing faster than `tolerance`
            self._auxiliary_storage[condition] = current_value
        else:
            # its increasing or decreasing too slowly
            decreasing = False

        # returning True is 1 step closer to stopping
        return not decreasing

    def _early_condition_pre_check(self):
        assert len(self.conditions) == 1, "early stopping only accepts one variable"
        return self._early_condition

    def __call__(self, **kwargs):
        if self.operation is None:
            return False

        if self.check:
            if not all(cond in kwargs for cond in self.conditions):
                _conds = [c for c in self.conditions if c not in kwargs]
                raise ValueError(
                    "Not all selected metrics are available "
                    f"from the training: {_conds}. "
                    f"Available are: {list(kwargs)}"
                )
            self.check = False

        if self.operation(**kwargs):
            self.stay -= 1
        else:
            self.stay = self.original_stay

        return self.stay < 0


class MetricLogger:
    def __init__(self, variables: Union[Literal["any"], List[str], None] = "any"):
        if variables is None:
            variables = []
        self.variables: Dict[str, List[float]] = defaultdict(list)

        self.log_any = variables == "any"
        if isinstance(variables, list):
            for var in variables:
                self.variables[var] = []

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

    def log(self, **kwargs):
        if self.log_any:
            msg = self.__full_logger(**kwargs)
        else:
            msg = self.__select_logger(**kwargs)

        return msg

    @property
    def selection(self):
        return self.variables.keys()

    def __full_logger(self, tocsv: bool = False, **kwargs):
        metric_names: List[str] = []
        metric_values = []
        for name, values in self.variables.items():
            last_value = values[-1]

            metric_names.append(name)
            metric_values.append(last_value)

        if tocsv:
            return ",".join([f"{value:.8f}" for value in metric_values])
        else:
            msg = [
                f"{name} {value:.8f}"
                for name, value in zip(metric_names, metric_values)
            ]
            return " ".join(msg)

    def __select_logger(self, tocsv: bool = False, **kwargs):
        metric_names: List[str] = []
        metric_values = []
        for name in self.selection:
            try:
                last_value = self.variables[name][-1]

                metric_names.append(name)
                metric_values.append(last_value)
            except KeyError as e:
                if not self.warned:
                    key = e.args[0]
                    warnings.warn(
                        "Not all selected metrics are available. "
                        f"{key} is not present. "
                        f"Available are: {list(self.variables)}",
                        UserWarning,
                        stacklevel=2,
                    )
                    self.warned = True

        if tocsv:
            return ",".join([f"{value:.8f}" for value in metric_values])
        else:
            msg = [
                f"{name} {value:.8f}"
                for name, value in zip(metric_names, metric_values)
            ]
            return " ".join(msg)
