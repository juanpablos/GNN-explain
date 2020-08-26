from abc import ABC, abstractmethod
from functools import reduce

import networkx as nx
import numpy as np

__all__ = ["AND", "FOC", "NEG", "OR", "Exist", "ForAll", "Property", "Role"]


class Element(ABC):
    @abstractmethod
    def __call__(self, **kwargs):
        raise NotImplementedError("Element is abstract")

    @abstractmethod
    def __repr__(self):
        raise NotImplementedError("Element is abstract")

    @abstractmethod
    def __str__(self):
        raise NotImplementedError("Element is abstract")

    def _accept(self, visitor):
        getattr(visitor, f"_visit_{self.__class__.__name__}")(self)

    def __eq__(self, other):
        if self.__class__ == other.__class__:
            return repr(self) == repr(other)
        return False

    def __hash__(self):
        return hash(repr(self))


class IndependentElement(Element):
    ...


class DependentElement(Element):
    ...


class Property(IndependentElement):
    """Returns a 1d vector with the nodes that satisfy the condition"""
    # REV: seach for a better way to do this
    available = {
        "RED": 0,
        "BLUE": 1,
        "GREEN": 2,
        "BLACK": 3
    }

    def __init__(self, prop: str, *, variable: str = None):
        if prop not in self.available:
            raise ValueError("Property not available")

        self.prop = self.available[prop]
        self.name = prop
        self.variable = variable if variable is not None else "."

    def __call__(self, properties, **kwargs):
        """Returns a numpy array with a 1 for nodes that satisfy the property, and 0 to which does not"""
        return properties == self.prop

    def __repr__(self):
        return f"{self.__class__.__name__}({self.name!r})"

    def __str__(self):
        return f"{self.name}({self.variable})"


class Role(DependentElement):
    """Returns a 2d matrix with the relations between nodes that satisfy the condition"""

    def __init__(
            self,
            relation: str,
            *,
            variable1: str = None,
            variable2: str = None):
        self.name = relation
        self.variable1 = variable1 if variable1 is not None else "."
        self.variable2 = variable2 if variable2 is not None else "."

    def __call__(self, graph, adjacency, **kwargs):
        """Returns an adjacency matrix for a graph"""
        if adjacency["value"] is None:
            adjacency["value"] = nx.adjacency_matrix(graph).toarray()
        return adjacency["value"]

    def __repr__(self):
        return f"{self.__class__.__name__}({self.name!r})"

    def __str__(self):
        return f"{self.name}({self.variable1}, {self.variable2})"


class Operator(IndependentElement):
    def __init__(self, *args: Element):
        self.operands = args

    def __repr__(self):
        args = ",".join([repr(el) for el in self.operands])
        return f"{self.__class__.__name__}({args})"


class NEG(Operator):
    def __init__(self, expression: Element):
        super().__init__()
        self.expression = expression

    def __repr__(self):
        expr = self.expression
        return f"{self.__class__.__name__}({expr!r})"

    def __str__(self):
        return f"¬({self.expression})"

    def __call__(self, **kwargs):
        first = self.expression(**kwargs)
        return np.logical_not(first)  # type: ignore


class AND(Operator):
    def __init__(self, first: Element, second: Element, *args: Element):
        super().__init__(first, second, *args)

    def __str__(self):
        rep = " ∧ ".join(([str(op) for op in self.operands]))
        return f"({rep})"

    def __call__(self, **kwargs):
        intermediate = [expr(**kwargs) for expr in self.operands]
        return reduce(np.logical_and, intermediate)  # type: ignore


class OR(Operator):
    def __init__(self, first: Element, second: Element, *args: Element):
        super().__init__(first, second, *args)

    def __str__(self):
        rep = " ∨ ".join(([str(op) for op in self.operands]))
        return f"({rep})"

    def __call__(self, **kwargs):
        intermediate = [expr(**kwargs) for expr in self.operands]
        return reduce(np.logical_or, intermediate)  # type: ignore


class Exist(IndependentElement):
    def __init__(
            self,
            expression: Element,
            lower: int = None,
            upper: int = None,
            *,
            variable: str = None):
        self.variable = variable if variable is not None else "."
        self.expression = expression
        self._lower = lower
        self.upper = upper
        self.symbol = "∃"

    @property
    def lower(self):
        # ! lower is only forced 1 if upper is also None
        # ! it can be that we have Exist(None, 3), in which case
        # ! 0 elements are also valid
        if self._lower is None and self.upper is None:
            return 1
        elif self._lower is None and self.upper is not None:
            return 0
        else:
            return self._lower

    def __repr__(self):
        # * Exist(None, None) is the same as Exist(1, None) and
        # !! Exist(None, 4) should be the same as Exist(0, 4)
        lower = self._lower
        if lower is None and self.upper is None:
            lower = 1
        return (f"{self.__class__.__name__}"
                f"({self.expression!r},{lower},{self.upper})")

    def __str__(self):
        s = self.symbol

        lower = self._lower
        if lower is None and self.upper is None:
            lower = 1

        if lower is not None and self.upper is not None:
            return f"{s}({lower}<={self.variable}<={self.upper}){self.expression}"
        elif lower is not None:
            return f"{s}({lower}<={self.variable}){self.expression}"
        else:
            return f"{s}({self.variable}<={self.upper}){self.expression}"

    def __call__(self, **kwargs):
        lower = self.lower
        upper = self.upper if self.upper is not None else float("inf")

        res = self.expression(**kwargs)
        if res.ndim <= 1:
            raise ValueError(
                "Cannot have a restriction property with 1d array "
                "(there must be a relation operation)")

        per_node = np.sum(res, axis=1)
        return (per_node >= lower) & (per_node <= upper)


class ForAll(IndependentElement):
    def __init__(self, expression: Element, *, variable: str = None):
        self.variable = variable if variable is not None else "."
        self.expression = expression
        self.symbol = "∀"

    def __repr__(self):
        return f"{self.__class__.__name__}({self.expression!r})"

    def __str__(self):
        s = self.symbol
        return f"{s}({self.variable}){self.expression}"

    def __call__(self, **kwargs):
        res = self.expression(**kwargs)
        if res.ndim <= 1:
            raise ValueError(
                "Cannot have a restriction property with single values")

        # * for all will always come with a 2d array from a relation between nodes. Unless we accept reflexive relationships. EDGE is not reflexive.
        np.fill_diagonal(res, True)
        return np.all(res, axis=1)


class FOC:
    def __init__(self, expression: IndependentElement):
        self.expression = expression

    def __call__(self, graph: nx.Graph) -> np.ndarray:
        adjacency = {"value": None}
        properties = list(nx.get_node_attributes(graph, "properties").values())
        properties = np.array(properties)
        res = self.expression(
            graph=graph,
            adjacency=adjacency,
            properties=properties)

        if res.ndim > 1:
            res = np.squeeze(res)

        assert res.ndim == 1, "Labels must be one per item"
        return res.astype(int)

    def __str__(self):
        return str(self.expression)

    def __repr__(self):
        return repr(self.expression)
