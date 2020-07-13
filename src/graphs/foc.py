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
    # TODO: seach for a better way to do this
    available = {
        "RED": 0,
        "BLUE": 1,
        "GREEN": 2,
        "BLACK": 3
    }

    def __init__(self, prop, variable):
        if prop not in self.available:
            raise Exception("Property not available")

        self.prop = self.available[prop]
        self.name = prop
        self.variable = variable

    def __call__(self, properties, **kwargs):
        """Returns a 2d numpy array with 1 for nodes that satisfy the property, and 0 to which does not
        dimensions are (n_nodes, 1)
        """
        # ? here depends in the type of property, for now it is a any()
        res = [self.prop in node for node in properties]
        # return self.prop(graph.node[mapping[self.variable]]['property'])
        return np.array(res, dtype=int)

    def __repr__(self):
        return f"{self.name}({self.variable})"


class Role(Concept):

    def __init__(self, relation, variable1, variable2):
        # if relation not in self.available:
        #     raise Exception("Relation not available")

        # self.relation = self.available[relation]
        self.name = relation
        self.variable1 = variable1
        self.variable2 = variable2

    def __call__(self, graph, adjacency, **kwargs):
        """Returns an adjacency matrix for a graph
        """
        # return self.relation(node1=mapping[self.variable1],
        #                      node2=mapping[self.variable2],
        #                      graph=graph)
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
        return np.logical_not(self.first(**kwargs))


class AND(Operator):
    def __init__(self, first, second):
        super().__init__(first, second)

    def __repr__(self):
        return f"({self.first} ∧ {self.second})"

    def __call__(self, **kwargs):
        return np.logical_and(self.first(**kwargs), self.second(**kwargs))


class OR(Operator):
    def __init__(self, first, second):
        super().__init__(first, second)

    def __repr__(self):
        return f"({self.first} ∨ {self.second})"

    def __call__(self, **kwargs):
        return np.logical_or(self.first(**kwargs), self.second(**kwargs))


class Exist(Element):
    def __init__(self, variable, expression, lower=None, upper=None):
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
        # variable and self.variable must be different
        # self.variable must not have been used yet
        # assert self.variable not in mapping

        lower = self.lower if self.lower is not None else 1
        upper = self.upper if self.upper is not None else float("inf")

        res = self.expression(**kwargs)
        if res.ndim == 1:
            raise Exception(
                "Cannot have a restriction property with single values")

        per_node = np.sum(res, axis=1)

        # running_check = 0
        # for node in graph:
        #     mapping[self.variable] = node
        #     running_check += self.expression(
        #         graph=graph, mapping=mapping)

        #     if running_check > upper:
        #         break

        # mapping.pop(self.variable)
        # if lower <= running_check <= upper:
        #     return True
        # else:
        #     return False

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
        # # variable and self.variable must be different
        # # self.variable must not have been used yet
        # assert self.variable not in mapping

        # running_check = True
        # for node in graph:
        #     mapping[self.variable] = node
        #     running_check &= self.expression(
        #         graph=graph, mapping=mapping)

        #     if not running_check:
        #         break

        # mapping.pop(self.variable)
        # if running_check:
        #     return True
        # else:
        #     return False
        res = self.expression(**kwargs)
        if res.ndim == 1:
            raise Exception(
                "Cannot have a restriction property with single values")

        return np.all(res, axis=1)

    def symbol(self):
        return "∀"


class FOC:
    def __init__(self, expression):
        self.expression = expression

    def __call__(self, graph):
        adjacency = {"value": None}
        properties = list(nx.get_node_attributes(graph, "properties").values())
        properties = np.array(properties)

        # labels = []
        # mapping = {}
        # for node in graph:
        #     mapping[variable] = node
        #     if self.expression(
        #             graph=graph,
        #             mapping=mapping):
        #         labels.append(1)
        #     else:
        #         labels.append(0)
        # return labels
        res = self.expression(
            graph=graph,
            adjacency=adjacency,
            properties=properties)

        if res.ndim > 1:
            res = np.squeeze(res)

        return res.astype(int)

    def __repr__(self):
        return repr(self.expression)
