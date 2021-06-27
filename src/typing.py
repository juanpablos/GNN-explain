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
    runtime_checkable,
)


from src.data.formula_index import FormulaMapping
from src.data.formulas.filter import Filter
from src.data.formulas.labeler import LabelerApply, SequenceLabelerApply

T = TypeVar("T")
S = TypeVar("S")
T_co = TypeVar("T_co", covariant=True)
S_co = TypeVar("S_co", covariant=True)


class NetworkDataConfig(TypedDict):
    root: str
    model_hash: str
    selector: Filter
    labeler: Union[LabelerApply, SequenceLabelerApply]
    formula_mapping: FormulaMapping
    test_selector: Filter
    load_aggregated: Optional[str]
    force_preaggregated: bool


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


class GNNEncoderModelConfig(TypedDict):
    layer_input_dim: Optional[int]
    output_input_dim: Optional[int]
    encoder_num_layers: int
    encoder_hidden_dim: int
    layer_embedding_dim: int
    merge_strategy: Literal["cat", "sum", "prod"]
    output_dim: int


class GNNModelConfig(MinModelConfig):
    name: str
    aggregate_type: Literal["add", "mean", "max"]
    combine_type: Literal["identity", "linear", "mlp"]
    mlp_layers: int
    combine_layers: int
    task: Literal["node", "graph"]


class LSTMConfigBase(TypedDict):
    name: str
    encoder_dim: int
    embedding_dim: int
    hidden_dim: int
    vocab_size: Optional[int]
    init_state_context: bool
    dropout_prob: float
    concat_encoder_input: bool


class LSTMConfig(LSTMConfigBase, total=False):
    compose_encoder_state: bool
    compose_dim: int


class CrossFoldConfiguration(TypedDict):
    n_splits: int
    shuffle: bool
    random_state: int
    defer_loading: bool


class MetricHistory(Protocol):
    def __getitem__(self, key: str) -> float:
        ...

    def keys(self, select: bool = ...) -> Iterator[str]:
        ...

    def items(self, select: bool = ...) -> Iterator[Tuple[str, List[float]]]:
        ...

    def log(self) -> str:
        ...

    def update(self, **kwargs: float) -> None:
        ...

    def get_history(self, key: str) -> List[float]:
        ...


@runtime_checkable
class Indexable(Protocol[T_co]):
    def __getitem__(self, index: int) -> T_co:
        ...

    def __len__(self) -> int:
        ...


class IndexableIterable(Indexable[T_co], Protocol[T_co]):
    def __iter__(self) -> Iterator[T_co]:
        ...


@runtime_checkable
class DatasetLike(Protocol):
    @property
    def dataset(self):
        ...

    @property
    def labels(self):
        ...
