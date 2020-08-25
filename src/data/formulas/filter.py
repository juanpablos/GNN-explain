from typing import Iterable, List, Literal, Mapping, Union

from src.data.formulas.visitor import Visitor
from src.graphs.foc import Element, Exist, Property

# REV: seach for a better way to do this
_available = ["RED", "BLUE", "GREEN", "BLACK"]


def _check_atomic(atomic):
    if not all(at in _available for at in atomic):
        _not = [at for at in atomic if at not in _available]
        raise ValueError(
            f"Not all selected atomic formulas are available. {_not}")


class Filterer(Visitor[bool]):
    def __init__(self):
        self.result = False

    def reset(self):
        self.result = False


class AtomicFilter(Filterer):
    def __init__(self,
                 atomic: Union[Literal["all"],
                               Iterable[str]],
                 hop: int = None):
        super().__init__()
        if atomic == "all":
            self.selected = _available
        else:
            _check_atomic(atomic)
            self.selected = atomic

        self.current_hop = 0
        if hop is None:
            self.target_hop = -1
        else:
            self.target_hop = hop

    def _visit_Exist(self, node: Exist):
        self.current_hop += 1
        super()._visit_Exist(node)
        self.current_hop -= 1

    def _visit_Property(self, node: Property):
        if node.name in self.selected and \
                (self.target_hop == -1 or self.current_hop == self.target_hop):
            self.result = True

    def __str__(self):
        return f"AtomicFilter({self.selected},{self.target_hop})"


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

    def __str__(self):
        return f"RestrictionFilter({self.lower},{self.upper})"


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

    def __str__(self):
        if not self.filters:
            raise ValueError(
                "There must be at least 1 filter set to be applied")

        if len(self.filters) == 1:
            return f"{self.filters[0]}"
        else:
            filter_str = ",".join(str(filt) for filt in self.filters)
            return f"{self.condition}({filter_str})"


class SelectFilter:
    def __init__(self, hashes: List[str], name: str = None):
        self.hashes = hashes
        self.name = name

    def __call__(self, formulas: Mapping[str, Element]):
        if not all(_hash in formulas for _hash in self.hashes):
            _not = [_hash for _hash in self.hashes if _hash not in formulas]
            raise ValueError(f"Not all selected hashes are available: {_not}")
        return {_hash: formulas[_hash] for _hash in self.hashes}

    def __str__(self):
        if self.name is None:
            return f"ManualFilter({len(self.hashes)})"
        else:
            return self.name


class NoFilter:
    def __call__(self, formulas: Mapping[str, Element]):
        return formulas

    def __str__(self):
        return "NoFilter()"
