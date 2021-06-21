from typing import (
    Dict,
    Generic,
    List,
    Literal,
    Mapping,
    Optional,
    OrderedDict,
    Tuple,
    TypeVar,
)

from src.data.formulas.visitor import Visitor
from src.data.vocabulary import Vocabulary
from src.graphs.foc import AND, NEG, OR, Element, Exist, Property, Role

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
        if node.name == self.selected and (
            self.target_hop == -1 or self.current_hop == self.target_hop
        ):
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
        self, lower: Optional[int], upper: Optional[int], negate: bool = False
    ):
        if lower is None and upper is None:
            raise ValueError("Can't have both open intervals.")
        if lower is not None and upper is not None and lower < 0 and upper < 0:
            raise ValueError("Can't have both intervals as ANY")
        super().__init__(negate=negate)
        self.lower = lower
        self.upper = upper

        if self.lower == -1:
            self.lower_checker = lambda _: True
            self.lower_str = "ANY"
        else:
            self.lower_checker = lambda lower: lower == self.lower
            self.lower_str = str(self.lower)

        if self.upper == -1:
            self.upper_checker = lambda _: True
            self.upper_str = "ANY"
        else:
            self.upper_checker = lambda upper: upper == self.upper
            self.upper_str = str(self.upper)

        txt = f"restriction({self.lower_str},{self.upper_str})"
        if negate:
            txt = f"NEG({txt})"
        # possitive class
        self.classes[1] = txt

    def _visit_Exist(self, node: Exist):
        if self.lower_checker(node.lower) and self.upper_checker(node.upper):
            self.result = 1
        super()._visit_Exist(node)

    def __str__(self):
        return f"BinaryRestriction({self.lower_str},{self.upper_str},{self.negate})"


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


# *----- multilabel


class MultiLabelCategoricalLabeler(CategoricalLabeler[Tuple[int, ...], int]):
    def __init__(self, class_for_no_label: bool = False):
        super().__init__()
        self.current_counter = 0
        self.result: Tuple[int, ...] = ()
        self.current_result: List[int] = []

        self.class_for_no_label = class_for_no_label
        if class_for_no_label:
            self.classes[self.current_counter] = "N/C"
            self.current_counter += 1

    def reset(self):
        self.current_result = []
        self.result = ()

    def process(self, formula: Element):
        if not self.current_result:
            if self.class_for_no_label:
                self.current_result.append(0)
            else:
                raise ValueError(f"Current formula don't have any label: {formula}")
        # ** we do not care about the order of the output
        self.result = tuple(set(self.current_result))


class MultiLabelAtomicLabeler(MultiLabelCategoricalLabeler):
    def __init__(self, class_for_no_label: bool = False):
        super().__init__(class_for_no_label=class_for_no_label)
        # TODO: unify inverse logic between multilabel labelers
        # no need for this to be ordered
        self.inverse_classes: Dict[str, int] = {}

    def _visit_Property(self, node: Property):
        if node.name not in self.inverse_classes:
            self.classes[self.current_counter] = node.name
            self.inverse_classes[node.name] = self.current_counter
            self.current_counter += 1

        self.current_result.append(self.inverse_classes[node.name])

    def __str__(self):
        return "MultiLabelAtomic()"


class MultilabelRestrictionLabeler(MultiLabelCategoricalLabeler):
    """
    Labels are for compound Exist operations:
    Exist(..., 2, 3) -> Exist(..., 2, None) + Exist(..., None, 3)
    Exist(..., 2, None) -> Exist(..., 2, None)

    RED -> Exist(None)
    """

    def __init__(
        self,
        mode: Literal["lower", "upper", "both"] = "both",
        class_for_no_label: bool = False,
    ):
        super().__init__(class_for_no_label=class_for_no_label)
        self.current_hop = 0
        self.max_hop = 0
        # no need for this to be ordered
        self.pairs: Dict[Tuple[Optional[int], Optional[int]], int] = {}

        if mode == "both":
            self.mode = ["lower", "upper"]
        else:
            self.mode = [mode]

    def _visit_Exist(self, node: Exist):
        self.current_hop += 1
        self.max_hop = max(self.max_hop, self.current_hop)

        if self.current_hop > 1:
            # TODO: multilabel hop restriction labele
            raise NotImplementedError(
                "Missing implementation for multihop restriction labeler"
            )

        if "lower" in self.mode and node.lower is not None:
            if (node.lower, None) not in self.pairs:
                self.pairs[(node.lower, None)] = self.current_counter
                self.classes[self.current_counter] = f"Exist({node.lower},None)"
                self.current_counter += 1
            self.current_result.append(self.pairs[(node.lower, None)])

        if "upper" in self.mode and node.upper is not None:
            if (None, node.upper) not in self.pairs:
                self.pairs[(None, node.upper)] = self.current_counter
                self.classes[self.current_counter] = f"Exist(None,{node.upper})"
                self.current_counter += 1
            self.current_result.append(self.pairs[(None, node.upper)])

        super()._visit_Exist(node)

        self.current_hop -= 1

    def process(self, formula: Element):
        if self.max_hop == 0:
            if (None, None) not in self.pairs:
                self.pairs[(None, None)] = self.current_counter
                self.classes[self.current_counter] = "Exist(None)"
                self.current_counter += 1

            self.current_result.append(self.pairs[(None, None)])
        super().process(formula)

    def reset(self):
        super().reset()
        self.current_hop = 0
        self.max_hop = 0

    def __str__(self):
        modes = ",".join(self.mode)
        return f"MultiLabelRestriction({modes})"


# *----- text sequential


class TextSequenceLabeler(Visitor[List[int]]):
    def __init__(self):
        self.vocabulary = Vocabulary()
        self.result: List[int] = []

    def reset(self):
        self.result = []

    def preload_vocabulary(self, vocabulary: Dict[str, int]):
        self.vocabulary.load_vocab(vocabulary)

    def _visit_AND(self, node: AND):
        token_id = self.vocabulary.add_or_get("AND")
        self.result.extend([token_id] * (len(node.operands) - 1))
        super()._visit_AND(node)

    def _visit_OR(self, node: OR):
        token_id = self.vocabulary.add_or_get("OR")
        self.result.extend([token_id] * (len(node.operands) - 1))
        super()._visit_OR(node)

    def _visit_NEG(self, node: NEG):
        token_id = self.vocabulary.add_or_get("NEG")
        self.result.append(token_id)
        super()._visit_NEG(node)

    def _visit_Property(self, node: Property):
        prop = node.name
        token_id = self.vocabulary.add_or_get(prop)
        self.result.append(token_id)
        super()._visit_Property(node)

    def _visit_Role(self, node: Role):
        role = node.name
        token_id = self.vocabulary.add_or_get(role)
        self.result.append(token_id)
        super()._visit_Role(node)

    def _visit_Exist(self, node: Exist):

        current_result = self.result
        self.result = []
        super()._visit_Exist(node)

        exist_result = self.result
        self.result = current_result

        lower = []
        if node.lower is not None:
            lower_id = self.vocabulary.add_or_get(f"Exist({node.lower}, None)")
            lower.append(lower_id)
            lower.extend(exist_result)

        upper = []
        if node.upper is not None:
            upper_id = self.vocabulary.add_or_get(f"Exist(None, {node.upper})")
            upper.append(upper_id)
            upper.extend(exist_result)

        if lower and upper:
            and_id = self.vocabulary.add_or_get("AND")
            self.result.append(and_id)

        self.result.extend(lower)
        self.result.extend(upper)

    def process(self, formula: Element):
        if not self.result:
            raise ValueError(f"Current formula don't have any result: {formula}")

        self.result = (
            [self.vocabulary.start_token_id]
            + self.result
            + [self.vocabulary.end_token_id]
        )

    def __str__(self):
        return "TextSequenceAtomic()"


# *------- apply --------


class LabelerApply(Generic[T, S]):
    def __init__(self, labeler: CategoricalLabeler[T, S]):
        self.labeler = labeler

    def __call__(self, formulas: Mapping[str, Element]):
        labels = {_hash: self.labeler(formula) for _hash, formula in formulas.items()}
        return labels, self.labeler.classes

    def __str__(self):
        return str(self.labeler)


class SequenceLabelerApply:
    def __init__(self, labeler: TextSequenceLabeler):
        self.labeler = labeler

    def __call__(self, formulas: Mapping[str, Element]):
        labels = {_hash: self.labeler(formula) for _hash, formula in formulas.items()}
        return labels, self.labeler.vocabulary

    def __str__(self):
        return str(self.labeler)
