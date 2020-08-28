from typing import (
    Dict,
    Generic,
    List,
    Mapping,
    Optional,
    OrderedDict,
    Tuple,
    TypeVar
)

from src.data.formulas.visitor import Visitor
from src.graphs.foc import Element, Exist, Property

T = TypeVar("T")
S = TypeVar("S")
T_co = TypeVar("T_co", covariant=True)
S_co = TypeVar("S_co", covariant=True)


class CategoricalLabeler(Visitor[T_co], Generic[T_co, S_co]):
    def __init__(self):
        self.classes: OrderedDict[S_co, str] = OrderedDict()

# *----- binary


class BinaryCategoricalLabeler(CategoricalLabeler[int, int]):
    def __init__(self, negate: bool = False):
        super().__init__()
        self.result = 0
        self.negate = negate
        # default class
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

        txt_hop = str(hop) if hop is not None else "any"
        txt = f"{atomic} {txt_hop} hop"
        if negate:
            txt = f"NEG({txt})"
        # possitive class
        self.classes[1] = txt

    def _visit_Exist(self, node: Exist):
        self.current_hop += 1
        super()._visit_Exist(node)
        self.current_hop -= 1

    def _visit_Property(self, node: Property):
        if node.name == self.selected and \
                (self.target_hop == -1 or self.current_hop == self.target_hop):
            # if the atomic is seen, then mark as 1
            self.result = 1

    def reset(self):
        super().reset()
        self.current_hop = 0

    def __str__(self):
        return f"BinaryAtomic({self.selected},{self.target_hop},{self.negate})"


class BinaryHopLabeler(BinaryCategoricalLabeler):
    def __init__(self, hop: int, negate: bool = False):
        if hop < 0:
            raise ValueError("Hop must be greater or equal to 0.")
        super().__init__(negate=negate)
        self.current_hop = 0
        self.max_hop = 0
        self.target_hop = hop

        txt = f"is {hop} hop"
        if negate:
            txt = f"NEG({txt})"
        # possitive class
        self.classes[1] = txt

    def _visit_Exist(self, node: Exist):
        self.current_hop += 1
        self.max_hop = max(self.max_hop, self.current_hop)
        super()._visit_Exist(node)
        self.current_hop -= 1

    def process(self, formula: Element):
        if self.max_hop == self.target_hop:
            self.result = 1

    def reset(self):
        super().reset()
        self.current_hop = 0
        self.max_hop = 0

    def __str__(self):
        return f"BinaryHop({self.target_hop},{self.negate})"


class BinaryRestrictionLabeler(BinaryCategoricalLabeler):
    def __init__(
            self,
            lower: Optional[int],
            upper: Optional[int],
            negate: bool = False):
        if lower is None and upper is None:
            raise ValueError("Can't have both open intervals.")
        super().__init__(negate=negate)
        self.lower = lower if lower is not None else 0
        self.upper = upper

        txt = f"restriction({lower},{upper})"
        if negate:
            txt = f"NEG({txt})"
        # possitive class
        self.classes[1] = txt

    def _visit_Exist(self, node: Exist):
        if self.lower == node.lower and self.upper == node.upper:
            self.result = 1
        super()._visit_Exist(node)

    def __str__(self):
        return f"BinaryRestriction({self.lower},{self.upper},{self.negate})"


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

    def __str__(self):
        return "Sequential()"


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
        # no need for this to be ordered
        self.inverse_classes: Dict[str, int] = {}

    def _visit_Property(self, node: Property):
        if node.name not in self.classes:
            self.classes[self.current_counter] = node.name
            self.inverse_classes[node.name] = self.current_counter
            self.current_counter += 1

        self.result.append(self.inverse_classes[node.name])

    def __str__(self):
        return "MultiLabelAtomic()"

# !! implement labeler that labels the number of hops in a formula


class MultilabelRestrictionLabeler(MultiLabelCategoricalLabeler):
    def __init__(self):
        super().__init__()
        self.current_hop = 0
        # no need for this to be ordered
        self.pairs: Dict[Tuple[Optional[int], Optional[int]], int] = {}
        # ?? should we support/assign label to atomic formulas?
        # something like Restriction(None) or something
        # !! this should actually be for an specific hop, not any

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

    def reset(self):
        super().reset()
        self.current_hop = 0

    def __str__(self):
        return "MultiLabelRestriction()"

# *------- apply --------


class LabelerApply(Generic[T, S]):
    def __init__(self, labeler: CategoricalLabeler[T, S]):
        self.labeler = labeler

    def __call__(self, formulas: Mapping[str, Element]):
        labels = {_hash: self.labeler(formula)
                  for _hash, formula in formulas.items()}
        return labels, self.labeler.classes

    def __str__(self):
        return str(self.labeler)
