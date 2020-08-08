from abc import ABC, abstractmethod
from functools import reduce

import networkx as nx
import numpy as np

__all__ = ["AND", "FOC", "NEG", "OR", "Exist", "ForAll", "Property", "Role"]


class Element(ABC):
    @abstractmethod
    def __call__(self, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def __repr__(self):
        raise NotImplementedError


class Concept(ABC):
    @abstractmethod
    def __call__(self, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def __repr__(self):
        raise NotImplementedError


class Property(Concept):
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


class Role(Concept):
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


class Operator(Element):
    def __init__(self, *args, **kwargs):
        self.operands = args
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __repr__(self):
        args = ",".join([repr(el) for el in self.operands])
        return f"{self.__class__.__name__}({args})"


class NEG(Operator):
    def __init__(self, expression):
        super().__init__(expression=expression)

    def __repr__(self):
        expr = self.expression  # type: ignore
        return f"{self.__class__.__name__}({expr!r})"

    def __str__(self):
        return f"¬({self.expression})"  # type: ignore

    def __call__(self, **kwargs):
        first = self.expression(**kwargs)  # type: ignore
        return np.logical_not(first)  # type: ignore


class AND(Operator):
    def __init__(self, first, second, *args):
        super().__init__(first, second, *args)

    def __str__(self):
        rep = " ∧ ".join(([str(op) for op in self.operands]))
        return f"({rep})"

    def __call__(self, **kwargs):
        intermediate = [expr(**kwargs) for expr in self.operands]
        return reduce(np.logical_and, intermediate)  # type: ignore


class OR(Operator):
    def __init__(self, first, second, *args):
        super().__init__(first, second, *args)

    def __str__(self):
        rep = " ∨ ".join(([str(op) for op in self.operands]))
        return f"({rep})"

    def __call__(self, **kwargs):
        intermediate = [expr(**kwargs) for expr in self.operands]
        return reduce(np.logical_or, intermediate)  # type: ignore


class Exist(Element):
    def __init__(
            self,
            expression,
            lower: int = None,
            upper: int = None,
            *,
            variable: str = None):
        self.variable = variable if variable is not None else "."
        self.expression = expression
        self.lower = lower
        self.upper = upper

    def __repr__(self):
        return (f"{self.__class__.__name__}"
                f"({self.expression!r},{self.lower},{self.upper})")

    def __str__(self):
        s = self.symbol()

        if self.lower is None and self.upper is None:
            return f"{s}({self.variable}){self.expression}"
        elif self.lower is not None and self.upper is not None:
            return f"{s}({self.lower}<={self.variable}<={self.upper}){self.expression}"
        elif self.lower is not None:
            return f"{s}({self.lower}<={self.variable}){self.expression}"
        else:
            return f"{s}({self.variable}<={self.upper}){self.expression}"

    def __call__(self, **kwargs):
        lower = self.lower if self.lower is not None else 1
        upper = self.upper if self.upper is not None else float("inf")

        res = self.expression(**kwargs)
        if res.ndim <= 1:
            raise ValueError(
                "Cannot have a restriction property with 1d array "
                "(there must be a relation operation)")

        per_node = np.sum(res, axis=1)
        return (per_node >= lower) & (per_node <= upper)

    def symbol(self):
        return "∃"


class ForAll(Element):
    def __init__(self, expression, *, variable: str = None):
        self.variable = variable if variable is not None else "."
        self.expression = expression

    def __repr__(self):
        return f"{self.__class__.__name__}({self.expression!r})"

    def __str__(self):
        s = self.symbol()
        return f"{s}({self.variable}){self.expression}"

    def __call__(self, **kwargs):
        res = self.expression(**kwargs)
        if res.ndim <= 1:
            raise ValueError(
                "Cannot have a restriction property with single values")

        # * for all will always come with a 2d array from a relation between nodes. Unless we accept reflexive relationships. EDGE is not reflexive.
        np.fill_diagonal(res, True)
        return np.all(res, axis=1)

    def symbol(self):
        return "∀"


class FOC:
    def __init__(self, expression):
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
