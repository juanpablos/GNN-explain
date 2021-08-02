from src.typing import MinModelConfig

from src.models import MLP


def init_MLP_model(configs: MinModelConfig):
    assert configs["input_dim"] is not None
    return MLP(**configs)
