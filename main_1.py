import csv
import hashlib
import json
import logging
import os
import random
from collections import defaultdict
from inspect import getsource
from timeit import default_timer as timer
from typing import Any, Dict

import torch

from src.data.datasets import RandomGraphDataset
from src.data.utils import clean_state
from src.generate_graphs import graph_stream
from src.graphs import *
from src.run_logic import run, seed_everything
from src.training.gnn_training import Training
from src.typing import GNNModelConfig, StopFormat
from src.utils import cleanup, merge_update, save_file_exists


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
6) blue and exist green neighbor (22%)
FOC(
    AND(
        Property("BLUE", "x"),
        Exist(
            "y",
            AND(
                Role("EDGE", "x", "y"),
                Property("GREEN", "y")
            )
        )
    )
)
7) blue and exist either a red or green neighbor (25%)
FOC(
    AND(
        Property("BLUE", "x"),
        Exist(
            "y",
            AND(
                Role("EDGE", "x", "y"),
                OR(
                    Property("RED", "y"),
                    Property("GREEN", "y")
                )
            )
        )
    )
)
8) red and at least 2 blue neighbors (15%)
FOC(
    AND(
        Property("RED", "x"),
        Exist(
            "y",
            AND(
                Role("EDGE", "x", "y"),
                Property("BLUE", "y")
            ),
            2
        )
    )
)
9) ????? either a node with between 2 to 4 blue neighbors, or a node with between 4 to 6 red neighbors
FOC(
    OR(
        Exist(
            "y",
            AND(
                Role("EDGE", "x", "y"),
                Property("BLUE", "y")
            ),
            2,
            4
        ),
        Exist(
            "y",
            AND(
                Role("EDGE", "x", "y"),
                Property("RED", "y")
            ),
            4,
            6
        )
    )
)
"""


def get_formula():
    f = FOC(
        OR(
            Exist(
                "y",
                AND(
                    Role("EDGE", "x", "y"),
                    Property("BLUE", "y")
                ),
                2,
                4
            ),
            Exist(
                "y",
                AND(
                    Role("EDGE", "x", "y"),
                    Property("RED", "y")
                ),
                4,
                6
            )
        )
    )
    return f


def run_experiment(
        n_models: int,
        save_path: str,
        file_name: str,
        model_config: GNNModelConfig,
        data_config: Dict[str, Any],
        train_length: int,
        test_length: int,
        iterations: int = 100,
        gpu_num: int = 0,
        data_workers: int = 2,
        batch_size: int = 64,
        test_batch_size: int = 512,
        lr: float = 0.01,
        stop_when: StopFormat = None):

    stream = graph_stream(**data_config)
    models = []

    stats = {
        "macro": defaultdict(int),
        "micro": defaultdict(int)
    }

    try:
        for m in range(1, n_models + 1):

            logging.info(f"Training model {m}/{n_models}")

            train_data = RandomGraphDataset(stream, train_length)
            test_data = RandomGraphDataset(stream, test_length)

            model, metrics = run(
                run_config=Training(),
                model_config=model_config,
                train_data=train_data,
                test_data=test_data,
                iterations=iterations,
                gpu_num=gpu_num,
                data_workers=data_workers,
                batch_size=batch_size,
                test_batch_size=test_batch_size,
                lr=lr,
                stop_when=stop_when)

            model.cpu()
            weights = clean_state(model.state_dict())
            models.append(weights)

            stats["macro"][round(metrics["macro"], 3)] += 1
            stats["micro"][round(metrics["micro"], 3)] += 1

    except KeyboardInterrupt:
        logging.info("Manually Interrumpted")
        _error_file = f"{save_path}/{file_name.format(len(models))}.error"
        with open(_error_file, "w") as o:
            o.write(f"Interrupted work in file {save_path}\n")
            o.write(f"Only {len(models)} models were written\n")
    except Exception as e:
        logging.error(f"Exception encountered: {type(e).__name__}")
        logging.error(f"Message: {e}")
        _error_file = f"{save_path}/{file_name.format(len(models))}.error"
        with open(_error_file, "w") as o:
            o.write(f"Problem in file {save_path}/{file_name.format('X')}\n")
            o.write(f"Exception encountered: {e}\n")
            o.write(f"Only {len(models)} models were written\n")
    finally:
        logging.info(f"Saving computed models...")

        exists, prev_file = save_file_exists(save_path, file_name)
        if exists:
            # ! this does not take care of race conditions
            logging.info("File already exists")
            logging.info("Appending new models to file")
            logging.debug("Loading previous models")

            prev_models = torch.load(f"{save_path}/{prev_file}")

            models.extend(prev_models)

            with open(f"{save_path}/{prev_file}.stat", "r") as f:
                prev_stats = json.load(f)
                merge_update(stats, prev_stats)

        logging.info(f"Saving {len(models)} models...")
        models_file = f"{save_path}/{file_name.format(len(models))}"
        torch.save(models, models_file)
        with open(f"{models_file}.stat", "w") as f:
            json.dump(stats, f, sort_keys=True, indent=2)

        cleanup(exists, save_path, prev_file)


def _write_metadata(
        destination: str,
        model_config: GNNModelConfig,
        model_config_hash: str,
        formula: FOC,
        formula_hash: str,
        data_config: Dict,
        seed: int,
        **kwargs):
    formula_source = getsource(get_formula)

    """
    format is:
    formula hash, formula string, model hash, seed,
        model config, data config, others, formula source
    """
    with open(destination, "a", newline='', encoding='utf-8') as f:
        writer = csv.writer(f, quotechar="|")
        writer.writerow([
            formula_hash,
            repr(formula),
            model_config_hash,
            seed,
            json.dumps(model_config),
            json.dumps(data_config),
            json.dumps(kwargs),
            formula_source
        ])


def main():
    seed = random.randint(1, 1 << 30)
    # seed = 10
    seed_everything(seed)

    n_models = 5000
    model_name = "acgnn"

    input_dim = 4

    model_config: GNNModelConfig = {
        "name": model_name,
        "input_dim": input_dim,
        "hidden_dim": 16,
        "hidden_layers": None,
        "output_dim": 2,
        "aggregate_type": "add",
        "combine_type": "identity",
        "num_layers": 2,
        "mlp_layers": 2,  # the number of layers in A and V
        "combine_layers": 2,  # layers in the combine MLP if combine_type=mlp
        "task": "node"
    }
    model_config_hash = hashlib.md5(
        json.dumps(
            model_config,
            sort_keys=True).encode()).hexdigest()[
        :10]

    formula = get_formula()
    formula_hash = hashlib.md5(repr(formula).encode()).hexdigest()[:10]

    data_config = {
        "generator_fn": "random",
        "min_nodes": 10,
        "max_nodes": 60,
        "seed": seed,
        "n_properties": input_dim,
        "property_distribution": "uniform",
        "distribution": None,
        "verbose": 0,
        # --- generator config
        "name": "erdos",
        # ! because the generated graph is undirected, the number of average neighbors will be double `m`
        "m": 4
    }

    save_path = f"data/gnns/{model_config_hash}"
    # ! manual operation
    os.makedirs(save_path, exist_ok=True)
    # * model_name - number of models - model hash - formula hash
    file_name = f"{model_name}-" + "n{}" + \
        f"-{model_config_hash}-{formula_hash}.pt"

    iterations = 20
    stop_when: StopFormat = {
        "operation": "and",  # and or or
        "conditions": {
            "micro": 0.999,
            "macro": 0.999
        },
        "stay": 1
    }

    # I want to be able to retrieve train_batch_length graphs train_batch times
    train_batch = 50
    train_batch_length = 16
    # I want to be able to retrieve test_batch_length graphs test_batch times
    test_batch = 1
    test_batch_length = 100

    _write_metadata(
        destination=f"{save_path}/.meta.csv",
        model_config=model_config,
        model_config_hash=model_config_hash,
        formula=formula,
        formula_hash=formula_hash,
        data_config=data_config,
        iterations=iterations,
        train_batch=train_batch,
        train_batch_length=train_batch_length,
        test_batch=test_batch,
        test_batch_length=test_batch_length,
        seed=seed,
        stop_when=stop_when,
    )

    data_config["formula"] = formula

    start = timer()
    run_experiment(
        n_models=n_models,
        save_path=save_path,
        file_name=file_name,
        model_config=model_config,
        data_config=data_config,
        train_length=train_batch * train_batch_length,
        test_length=test_batch * test_batch_length,
        iterations=iterations,
        gpu_num=0,
        data_workers=2,
        batch_size=train_batch,
        test_batch_size=test_batch,
        lr=0.01,
        stop_when=stop_when
    )
    end = timer()
    logging.info(f"Took {end-start} seconds")


if __name__ == "__main__":
    _console = logging.StreamHandler()
    _console.setLevel(logging.DEBUG)
    _console_f = logging.Formatter("[%(levelname)s] %(message)s")
    _console.setFormatter(_console_f)
    # _file = logging.FileHandler("debug_log.log")
    # _file.setLevel(logging.DEBUG)
    # _file_f = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
    # _file.setFormatter(_file_f)
    logging.basicConfig(
        level=logging.DEBUG,
        handlers=[
            _console
        ]
    )

    main()
