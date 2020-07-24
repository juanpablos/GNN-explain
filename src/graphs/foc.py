from abc import ABC

import networkx as nx
import numpy as np

# TODO: documentation


class Element(ABC):

    def __call__(self, **kwargs):
        raise NotImplementedError

    def __repr__(self):
        raise NotImplementedError


class Concept(ABC):
    def __call__(self, **kwargs):
        raise NotImplementedError

    def __repr__(self):
        raise NotImplementedError


class Property(Concept):
    # REV: seach for a better way to do this
    available = {
        "RED": 0,
        "BLUE": 1,
        "GREEN": 2,
        "BLACK": 3
    }

    def __init__(self, prop: str, variable: str):
        if prop not in self.available:
            raise Exception("Property not available")

        self.prop = self.available[prop]
        self.name = prop
        self.variable = variable

    def __call__(self, properties, **kwargs):
        """Returns a numpy array with a 1 for nodes that satisfy the property, and 0 to which does not
        """
        return properties == self.prop

    def __repr__(self):
        return f"{self.name}({self.variable})"


class Role(Concept):

    def __init__(self, relation: str, variable1: str, variable2: str):
        self.name = relation
        self.variable1 = variable1
        self.variable2 = variable2

    def __call__(self, graph, adjacency, **kwargs):
        """Returns an adjacency matrix for a graph
        """
        if adjacency["value"] is None:
            adjacency["value"] = nx.adjacency_matrix(graph).toarray()
        return adjacency["value"]

    def __repr__(self):
        return f"{self.name}({self.variable1}, {self.variable2})"


class Operator(Element):
    def __init__(self, first, second=None):
        self.first = first
        self.second = second


class NEG(Operator):
    def __init__(self, first):
        super().__init__(first)

    def __repr__(self):
        return f"¬({self.first})"

    def __call__(self, **kwargs):
        return np.logical_not(self.first(**kwargs))  # type: ignore


class AND(Operator):
    def __init__(self, first, second):
        super().__init__(first, second)

    def __repr__(self):
        return f"({self.first} ∧ {self.second})"

    def __call__(self, **kwargs):
        first = self.first(**kwargs)
        second = self.second(**kwargs)
        return np.logical_and(first, second)  # type: ignore


class OR(Operator):
    def __init__(self, first, second):
        super().__init__(first, second)

    def __repr__(self):
        return f"({self.first} ∨ {self.second})"

    def __call__(self, **kwargs):
        first = self.first(**kwargs)
        second = self.second(**kwargs)
        return np.logical_or(first, second)  # type: ignore


class Exist(Element):
    def __init__(
            self,
            variable: str,
            expression,
            lower: int = None,
            upper: int = None):
        self.variable = variable
        self.expression = expression
        self.lower = lower
        self.upper = upper

    def __repr__(self):
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
        if res.ndim == 1:
            raise Exception(
                "Cannot have a restriction property with 1d array (there must be a relation operation)")

        per_node = np.sum(res, axis=1)
        return (per_node >= lower) & (per_node <= upper)

    def symbol(self):
        return "∃"


class ForAll(Element):
    def __init__(self, variable, expression):
        self.variable = variable
        self.expression = expression

    def __repr__(self):
        s = self.symbol()
        return f"{s}({self.variable}){self.expression}"

    def __call__(self, **kwargs):
        res = self.expression(**kwargs)
        if res.ndim == 1:
            raise Exception(
                "Cannot have a restriction property with single values")

        # ??: should all include self?
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

    def __repr__(self):
        return repr(self.expression)
