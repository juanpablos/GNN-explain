import csv
import hashlib
import json
import random
from inspect import getsource
from typing import Any, Dict

import torch

from src.data.datasets import RandomGraphDataset
from src.data.gnn_data import clean_state
from src.generate_graphs import graph_stream
from src.graphs import *
from src.run_logic import run, seed_everything
from src.training.gnn_training import Training


"""
"RED": 0,
"BLUE": 1,
"GREEN": 2,
"BLACK": 3
"""
"""
1) FOC(Property("RED", "x")) -> 25%
2) FOC(Property("BLUE", "x")) -> 25%
3) FOC(Property("GREEN", "x")) -> 25%
4) FOC(Property("BLACK", "x")) -> 25%
5) FOC(OR(Property("BLUE", "x"), Property("GREEN", "x"))) -> 50%
6) FOC(NEG(Property("BLUE", "x"))) -> 75%
7) FOC(OR(Property("RED", "x"), Property("GREEN", "x"))) -> 50%
"""


def get_formula():
    f = FOC(Property("BLACK", "x"))
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

            train_data = RandomGraphDataset(stream, train_length)
            test_data = RandomGraphDataset(stream, test_length)

            model = run(
                run_config=Training,
                model_config=model_config,
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
        pass


def _write_metadata(
        destination: str,
        model_config: Dict,
        model_config_hash: str,
        formula: FOC,
        formula_hash: str,
        seed: int,
        file_name: str):
    formula_source = getsource(get_formula)

    """
    format is:
    file name, seed, model hash, model, formula hash, formula string, formula source
    """
    with open(destination, "a", newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([file_name, seed, model_config_hash, json.dumps(
            model_config), formula_hash, repr(formula), formula_source])


def main():
    seed = random.randint(1, 1 << 30)
    # seed = 10
    seed_everything(seed)

    n_models = 10000
    model_name = "acgnn"

    input_dim = 4

    model_config = {
        "name": model_name,
        "input_dim": input_dim,
        "hidden_dim": 8,
        "output_dim": 2,
        "aggregate_type": "max",
        "combine_type": "identity",
        "num_layers": 1,
        "mlp_layers": 1,  # the number of layers in A and V
        "combine_layers": 2,  # layers in the combine MLP if combine_type=mlp
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
        "max_nodes": 60,
        "seed": seed,
        "n_properties": input_dim,
        "n_property_types": 1,
        "property_distribution": "uniform",
        "distribution": None,
        "verbose": 0,
        # --- generator config
        "name": "erdos",
        "m": 2,
        "p": None
    }

    # * model_name - number of models - model hash - formula hash
    file_name = f"{model_name}-n{n_models}-{model_config_hash}-{formula_hash}"
    # TODO: check if file already exists
    save_path = f"data/gnns/{file_name}.pt"

    iterations = 5

    train_batch = 64
    test_batch = 100

    # I want to be able to retrieve 64 graphs 20 times
    train_length = 10 * train_batch
    # I want to be able to retrieve 100 graphs 1 time
    test_length = 1 * test_batch

    _write_metadata(
        destination="data/gnns/.meta.csv",
        model_config=model_config,
        model_config_hash=model_config_hash,
        formula=formula,
        formula_hash=formula_hash,
        seed=seed,
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
