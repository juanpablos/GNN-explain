from src.models import MLP
from src.typing import MinModelConfig


def init_MLP_model(configs: MinModelConfig):
    assert configs["input_dim"] is not None
    return MLP(**configs)
