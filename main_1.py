import hashlib
import json
from typing import Any, Dict

import torch

from src.generate_graphs import graph_stream
from src.graphs import *
from src.run_logic import run, seed_everything
from src.utils import LimitedStreamDataset
from src.utils.gnn_data import clean_state


def get_formula():
    # a0 = Property("RED", "x")

    # a1 = Property("BLUE", "y")
    # a2 = NEG(Role(relation="EDGE", variable1="x", variable2="y"))
    # a3 = AND(a1, a2)
    # a4 = Exist(variable="y", expression=a3, lower=2, upper=6)
    # a5 = AND(a0, a4)
    # _formula = FOC(a5)

    f = FOC(Property("RED", "x"))
    return f


def main(
        n_models: int,
        save_path: str,
        model_config: Dict[str, Any],
        data_config: Dict[str, Any],
        train_length: int,
        test_length: int,
        iterations: int = 100,
        gpu_num: int = 0,
        data_workers: int = 2,
        batch_size: int = 64,
        test_batch_size: int = 512,
        lr: float = 0.01):

    stream = graph_stream(**data_config)
    models = []

    for m in range(n_models):

        # TODO: remove
        print("Training model", m + 1)

        train_data = LimitedStreamDataset(stream, train_length)
        test_data = LimitedStreamDataset(stream, test_length)

        model = run(model_config=model_config,
                    train_graphs=train_data,
                    test_graphs=test_data,
                    iterations=iterations,
                    gpu_num=gpu_num,
                    data_workers=data_workers,
                    batch_size=batch_size,
                    test_batch_size=test_batch_size,
                    lr=lr)

        model.cpu()
        weights = clean_state(model.state_dict())
        models.append(weights)

    torch.save(models, save_path)


if __name__ == "__main__":
    _seed = 10
    seed_everything(_seed)

    _n_models = 1000
    _model_name = "acgnn"

    _input_dim = 2

    # TODO: write the model config to a file
    _model_config = {
        "name": _model_name,
        "input_dim": _input_dim,
        "hidden_dim": 16,
        "output_dim": 2,
        "aggregate_type": "max",
        "combine_type": None,
        "num_layers": 2,
        "combine_layers": 1,
        "num_mlp_layers": 2,
        "task": "node",
        "truncated_fn": None
    }
    _model_config_hash = hashlib.md5(
        json.dumps(
            _model_config,
            sort_keys=True).encode()).hexdigest()[
        :10]

    _formula = get_formula()
    _formula_hash = hashlib.md5(repr(_formula).encode()).hexdigest()[:10]
    # TODO: write the formula in a file, the sme file with the model config
    _data_config = {
        "formula": _formula,
        "generator_fn": "random",
        "min_nodes": 10,
        "max_nodes": 100,
        "seed": _seed,
        "n_properties": _input_dim,
        "n_property_types": 1,
        "property_distribution": "uniform",
        "distribution": None,
        "verbose": 0,
        # --- generator config
        "name": "erdos",
        "m": 3,
        "p": None
    }

    # * model_name - number of models - model hash - formula hash
    _file_name = f"{_model_name}-n{_n_models}-{_model_config_hash}-{_formula_hash}"
    # TODO: check if file already exists
    _save_path = f"data/gnns/{_file_name}.pt"

    _iterations = 10

    _train_batch = 64
    _test_batch = 512

    """number of graphs in the train dataset
    total number will be _train_length * batch_size"""
    # 100 * 64 = 6.400
    _train_length = 100
    """number of graphs in the test dataset
    total number will be _test_length * batch_size"""
    # test will be 10 times smaller than train
    # X * 512 -> train * train_batch // 10 // 512
    _test_length = _train_length * _train_batch // _test_batch // 10

    main(
        n_models=_n_models,
        save_path=_save_path,
        model_config=_model_config,
        data_config=_data_config,
        train_length=_train_length,
        test_length=_test_length,
        iterations=_iterations,
        gpu_num=0,
        data_workers=2,
        batch_size=_train_batch,
        test_batch_size=_test_batch,
        lr=0.01
    )
