from typing import Iterable, List, Literal, Mapping, Union

from src.data.formulas.visitor import Visitor
from src.graphs.foc import Element, Exist, Property

# REV: seach for a better way to do this
__available = ["RED", "BLUE", "GREEN", "BLACK"]


def _check_atomic(atomic):
    if not all(at in __available for at in atomic):
        _not = [at for at in atomic if at not in __available]
        raise ValueError(
            f"Not all selected atomic formulas are available. {_not}")


class Filterer(Visitor[bool]):
    def __init__(self):
        self.result = False

    def reset(self):
        self.result = False


class AtomicFilter(Filterer):
    def __init__(self, atomic: Union[Literal["all"], Iterable[str]]):
        super().__init__()
        if atomic == "all":
            self.selected = __available
        else:
            _check_atomic(atomic)
            self.selected = atomic

    def _visit_Property(self, node: Property):
        if node.name in self.selected:
            self.result = True


class AtomicHopFilter(AtomicFilter):
    def __init__(self, atomic: Union[Literal["all"], Iterable[str]], hop: int):
        super().__init__(atomic)
        self.current_hop = 0
        self.target_hop = hop

    def _visit_Exist(self, node: Exist):
        self.current_hop += 1
        super()._visit_Exist(node)
        self.current_hop -= 1

    def _visit_Property(self, node: Property):
        if node.name in self.selected and self.current_hop == self.target_hop:
            self.result = True


class RestrictionFilter(Filterer):
    def __init__(self, lower: int = None, upper: int = None):
        super().__init__()
        # * None is a placeholder
        self.lower = lower
        self.upper = upper

    def _visit_Exist(self, node: Exist):
        lower = self.lower == node.lower if self.lower is not None else True
        upper = self.upper == node.upper if self.upper is not None else True
        if lower and upper:
            self.result = True

        # ?? may be stop here
        super()._visit_Exist(node)


class FilterApply:
    def __init__(self,
                 filters: List[Filterer] = None,
                 condition: Union[Literal["and"],
                                  Literal["or"]] = "and"):
        if filters is None:
            self.filters: List[Filterer] = []
        else:
            self.filters = filters

        self.condition = all if condition == "and" else any

    def add(self, filterer: Filterer):
        self.filters.append(filterer)

    def _apply(self, formula: Element):
        return self.condition(filterer(formula) for filterer in self.filters)

    def __call__(self, formulas: Mapping[str, Element]):
        if not self.filters:
            raise ValueError(
                "There must be at least 1 filter set to be applied")
        return {_hash: formula for _hash, formula
                in formulas.items() if self._apply(formula)}
