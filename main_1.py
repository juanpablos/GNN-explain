import csv
import hashlib
import json
from inspect import getsource
from typing import Any, Dict

import torch

from src.generate_graphs import graph_stream
from src.graphs import *
from src.run_logic import run, seed_everything
from src.utils import LimitedStreamDataset
from src.utils.gnn_data import clean_state


"""
a0 = Property("RED", "x")
a1 = Property("BLUE", "y")
a2 = NEG(Role(relation="EDGE", variable1="x", variable2="y"))
a3 = AND(a1, a2)
a4 = Exist(variable="y", expression=a3, lower=2, upper=6)
a5 = AND(a0, a4)
f = FOC(a5)
"""

"""
"RED": 0,
"BLUE": 1,
"GREEN": 2,
"BLACK": 3
"""


def get_formula():
    f = FOC(Property("RED", "x"))
    return f


def run_experiment(
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

    m = 0
    try:
        for m in range(1, n_models + 1):

            # TODO: remove
            print("Training model", m)

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
    except KeyboardInterrupt:
        with open(f"{save_path}.error", "w") as o:
            o.write(f"Interrupted work in file {save_path}\n")
            o.write(f"Only {m} models were written\n")
    except Exception as e:
        with open(f"{save_path}.error", "w") as o:
            o.write(f"Problem in file {save_path}\n")
            o.write(f"Exception encountered: {e}\n")
            o.write(f"Only {m} models were written\n")
    finally:
        torch.save(models, save_path)


def _write_metadata(
        destination: str,
        model_config: Dict,
        model_config_hash: str,
        formula: FOC,
        formula_hash: str,
        file_name: str):
    formula_source = getsource(get_formula)

    """
    format is:
    file name, model hash, model, formula hash, formula string, formula source
    """
    with open(destination, "a", newline='') as f:
        writer = csv.writer(f)
        writer.writerow([file_name, model_config_hash, json.dumps(
            model_config), formula_hash, repr(formula), formula_source])


def main():
    seed = 10
    seed_everything(seed)

    n_models = 2000
    model_name = "acgnn"

    input_dim = 2

    model_config = {
        "name": model_name,
        "input_dim": input_dim,
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
    model_config_hash = hashlib.md5(
        json.dumps(
            model_config,
            sort_keys=True).encode()).hexdigest()[
        :10]

    formula = get_formula()
    formula_hash = hashlib.md5(repr(formula).encode()).hexdigest()[:10]

    data_config = {
        "formula": formula,
        "generator_fn": "random",
        "min_nodes": 10,
        "max_nodes": 100,
        "seed": seed,
        "n_properties": input_dim,
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
    file_name = f"{model_name}-n{n_models}-{model_config_hash}-{formula_hash}"
    # TODO: check if file already exists
    save_path = f"data/gnns/{file_name}.pt"

    iterations = 20

    train_batch = 64
    test_batch = 512

    """number of graphs in the train dataset
    total number will be _train_length * batch_size"""
    # 100 * 64 = 6.400
    train_length = 40
    """number of graphs in the test dataset
    total number will be _test_length * batch_size"""
    # test will be 10 times smaller than train
    # X * 512 -> train * train_batch // 10 // 512
    test_length = train_length * train_batch // test_batch // 10 or 1

    _write_metadata(
        destination="data/gnns/.meta.csv",
        model_config=model_config,
        model_config_hash=model_config_hash,
        formula=formula,
        formula_hash=formula_hash,
        file_name=file_name
    )

    from timeit import default_timer as timer
    start = timer()
    run_experiment(
        n_models=n_models,
        save_path=save_path,
        model_config=model_config,
        data_config=data_config,
        train_length=train_length,
        test_length=test_length,
        iterations=iterations,
        gpu_num=0,
        data_workers=2,
        batch_size=train_batch,
        test_batch_size=test_batch,
        lr=0.01
    )
    end = timer()
    print(f"Took {end-start} seconds")


if __name__ == "__main__":
    main()
