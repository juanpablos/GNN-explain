from abc import ABC, abstractmethod
from collections import defaultdict


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
        "RED": lambda x: x == 0,
        "BLUE": lambda x: x == 1,
        "GREEN": lambda x: x == 2,
        "BLACK": lambda x: x == 3
    }

    def __init__(self, prop, variable):
        if prop not in self.available:
            raise Exception("Property not available")

        self.prop = self.available[prop]
        self.name = prop
        self.variable = variable

    def __call__(self, graph, mapping, **kwargs):
        return self.prop(graph.node[mapping[self.variable]]['property'])

    def __repr__(self):
        return f"{self.name}({self.variable})"


def edge(node1, node2, graph):
    return node2 in graph.neighbors(node1)


class Role(Concept):
    # TODO: seach for a better way to do this
    available = {
        "EDGE": edge
    }

    def __init__(self, relation, variable1, variable2):
        if relation not in self.available:
            raise Exception("Relation not available")

        self.relation = self.available[relation]
        self.name = relation
        self.variable1 = variable1
        self.variable2 = variable2

    def __call__(self, graph, mapping, **kwargs):
        return self.relation(node1=mapping[self.variable1],
                             node2=mapping[self.variable2],
                             graph=graph)

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
        return not self.first(**kwargs)


class AND(Operator):
    def __init__(self, first, second):
        super().__init__(first, second)

    def __repr__(self):
        return f"({self.first} ∧ {self.second})"

    def __call__(self, **kwargs):
        return self.first(**kwargs) and self.second(**kwargs)


class OR(Operator):
    def __init__(self, first, second):
        super().__init__(first, second)

    def __repr__(self):
        return f"({self.first} ∨ {self.second})"

    def __call__(self, **kwargs):
        return self.first(**kwargs) or self.second(**kwargs)


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

    def __call__(self, graph, mapping, **kwargs):
        # variable and self.variable must be different
        # self.variable must not have been used yet
        assert self.variable not in mapping

        lower = self.lower if self.lower is not None else 1
        upper = self.upper if self.upper is not None else float("inf")

        running_check = 0
        for node in graph:
            mapping[self.variable] = node
            running_check += self.expression(
                graph=graph, mapping=mapping)

            if running_check > upper:
                break

        mapping.pop(self.variable)
        if lower <= running_check <= upper:
            return True
        else:
            return False

    def symbol(self):
        return "∃"


class ForAll(Element):
    def __init__(self, variable, expression):
        self.variable = variable
        self.expression = expression

    def __repr__(self):
        s = self.symbol()
        return f"{s}({self.variable}){self.expression}"

    def __call__(self, graph, mapping, **kwargs):
        # variable and self.variable must be different
        # self.variable must not have been used yet
        assert self.variable not in mapping

        running_check = True
        for node in graph:
            mapping[self.variable] = node
            running_check &= self.expression(
                graph=graph, mapping=mapping)

            if not running_check:
                break

        mapping.pop(self.variable)
        if running_check:
            return True
        else:
            return False

    def symbol(self):
        return "∀"


class FOC:
    def __init__(self, expression):
        self.expression = expression

    def __call__(self, graph, variable):
        labels = []
        mapping = {}
        for node in graph:
            mapping[variable] = node
            if self.expression(
                    graph=graph,
                    mapping=mapping):
                labels.append(1)
            else:
                labels.append(0)
        return labels

    def __repr__(self):
        return repr(self.expression)
