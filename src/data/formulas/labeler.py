from typing import Dict, Generic, List, Mapping, Optional, Tuple

from src.data.formulas.visitor import Visitor
from src.graphs.foc import Element, Exist, Property
from src.typing import S, S_co, T, T_co


class CategoricalLabeler(Visitor[T_co], Generic[T_co, S_co]):
    def __init__(self):
        self.classes: Dict[S_co, str] = {}

# *----- binary


class BinaryCategoricalLabeler(CategoricalLabeler[int, int]):
    def __init__(self, negate: bool = False):
        super().__init__()
        self.result = 0
        self.negate = negate
        self.classes[0] = "other"

    def reset(self):
        self.result = 0

    def process(self, formula: Element):
        if self.negate:
            # because the values are 0-1, we can do it like this
            self.result = (self.result + 1) % 2


class BinaryAtomicLabeler(BinaryCategoricalLabeler):
    def __init__(self, atomic: str, hop: int = None, negate: bool = False):
        super().__init__(negate=negate)
        self.selected = atomic
        self.current_hop = 0
        if hop is None:
            self.target_hop = -1
        else:
            self.target_hop = hop

        # FIX this is horrible, pls fix
        positive_text = "{} {} hop" if not negate else "NEG({} {} hop)"
        txt_hop = str(hop) if hop is not None else "any"
        self.classes[1] = positive_text.format(atomic, txt_hop)

    def _visit_Exist(self, node: Exist):
        self.current_hop += 1
        super()._visit_Exist(node)
        self.current_hop -= 1

    def _visit_Property(self, node: Property):
        if node.name == self.selected and \
                (self.target_hop == -1 or self.current_hop == self.target_hop):
            # if the atomic is seen, then mark as 1
            self.result = 1


class BinaryHopLabeler(BinaryCategoricalLabeler):
    def __init__(self, hop: int, negate: bool = False):
        super().__init__(negate=negate)
        if hop < 0:
            raise ValueError("Hop must be greater or equal to 0.")
        self.current_hop = 0
        self.max_hop = self.current_hop
        self.target_hop = hop

        self.classes[1] = f"is {hop} hop"

    def _visit_Exist(self, node: Exist):
        self.current_hop += 1
        self.max_hop = max(self.max_hop, self.current_hop)
        super()._visit_Exist(node)
        self.current_hop -= 1

    def process(self, formula: Element):
        if self.max_hop == self.target_hop:
            self.result = 1


class BinaryRestrictionLabeler(BinaryCategoricalLabeler):
    def __init__(self, lower: Optional[int], upper: Optional[int]):
        super().__init__()
        self.lower = lower
        self.upper = upper

        self.classes[1] = f"restriction({lower},{upper})"

    def _visit_Exist(self, node: Exist):
        if self.lower == node.lower and self.upper == node.upper:
            self.result = 1
        super()._visit_Exist(node)


# *----- multiclass


class SequentialCategoricalLabeler(CategoricalLabeler[int, int]):
    def __init__(self):
        super().__init__()
        self.current_counter = 0

    def reset(self):
        pass

    def __call__(self, node: Element):
        self.result = self.current_counter
        self.classes[self.result] = str(node)
        self.current_counter += 1
        return self.result


class MultiLabelCategoricalLabeler(CategoricalLabeler[List[int], int]):
    def __init__(self):
        super().__init__()
        self.current_counter = 0
        self.result: List[int] = []

    def reset(self):
        self.result = []

    def process(self, formula: Element):
        if not self.result:
            raise ValueError(
                f"Current formula don't have any label: {formula}")
        # ** we do not care about the order of the output
        self.result = list(set(self.result))


class MultiLabelAtomicLabeler(MultiLabelCategoricalLabeler):
    def __init__(self):
        super().__init__()
        # TODO: unify inverse logic between multilabel labelers
        self.inverse_classes: Dict[str, int] = {}

    def _visit_Property(self, node: Property):
        if node.name not in self.classes:
            self.classes[self.current_counter] = node.name
            self.inverse_classes[node.name] = self.current_counter
            self.current_counter += 1

        self.result.append(self.inverse_classes[node.name])


class MultilabelRestrictionLabeler(MultiLabelCategoricalLabeler):
    def __init__(self):
        super().__init__()
        self.current_hop = 0
        self.pairs: Dict[Tuple[Optional[int], Optional[int]], int] = {}
        # ?? should we support/assign label to atomic formulas?
        # something like Restriction(None) or something

    def _visit_Exist(self, node: Exist):
        self.current_hop += 1

        if (node.lower, node.upper) not in self.pairs:
            self.pairs[(node.lower, node.upper)] = self.current_counter
            _cls = (f"Restriction({node.lower},{node.upper},"
                    f"hop={self.current_hop})")
            self.classes[self.current_counter] = _cls
            self.current_counter += 1

        self.result.append(self.pairs[(node.lower, node.upper)])

        super()._visit_Exist(node)

        self.current_hop -= 1

# *------- apply --------


class LabelerApply(Generic[T, S]):
    def __init__(self, labeler: CategoricalLabeler[T, S]):
        self.labeler = labeler

    def __call__(self, formulas: Mapping[str, Element]):
        labels = {_hash: self.labeler(formula)
                  for _hash, formula in formulas.items()}
        return labels, self.labeler.classes
