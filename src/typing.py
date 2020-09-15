from typing import (
    Dict,
    Iterator,
    List,
    Literal,
    Optional,
    Protocol,
    Tuple,
    TypedDict,
    TypeVar,
    Union,
    runtime_checkable
)


from src.data.formula_index import FormulaMapping
from src.data.formulas.filter import Filter
from src.data.formulas.labeler import LabelerApply

T = TypeVar("T")
S = TypeVar("S")
T_co = TypeVar("T_co", covariant=True)
S_co = TypeVar("S_co", covariant=True)
TNum = TypeVar("TNum", int, float)


class NetworkDataConfig(TypedDict):
    root: str
    model_hash: str
    selector: Filter
    labeler: LabelerApply
    formula_mapping: FormulaMapping
    test_selector: Filter


class StopFormat(TypedDict):
    operation: str
    stay: int
    conditions: Dict[str, Union[int, float]]


class MinModelConfig(TypedDict):
    num_layers: int
    input_dim: Optional[int]
    hidden_dim: int
    output_dim: Optional[int]
    hidden_layers: Optional[List[int]]
    use_batch_norm: bool


class GNNModelConfig(MinModelConfig):
    name: str
    aggregate_type: Literal["add", "mean", "max"]
    combine_type: Literal["identity", "linear", "mlp"]
    mlp_layers: int
    combine_layers: int
    task: Literal["node", "graph"]


class MetricHistory(Protocol):
    def __getitem__(self, key: str) -> TNum: ...
    def keys(self, select: bool = ...) -> Iterator[str]: ...

    def items(self,
              select: bool = ...) -> Iterator[Tuple[str,
                                                    List[TNum]]]: ...

    def log(self) -> str: ...
    def update(self, **kwargs: TNum) -> None: ...
    def get_history(self, key: str) -> List[TNum]: ...


@runtime_checkable
class Indexable(Protocol[T_co]):
    def __getitem__(self, index: int) -> T_co: ...
    def __len__(self) -> int: ...


class IndexableIterable(Indexable[T_co], Protocol[T_co]):
    def __iter__(self) -> Iterator[T_co]: ...
