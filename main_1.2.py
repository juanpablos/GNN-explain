import hashlib
import json
import logging
import os
import random
from collections import defaultdict
from timeit import default_timer as timer
from typing import Any, Dict

import torch

from src.data.datasets import RandomGraphDataset
from src.data.utils import SubsetSampler, clean_state
from src.generate_graphs import graph_stream
from src.graphs import *
from src.run_logic import run, seed_everything
from src.training.gnn_training import Training
from src.typing import GNNModelConfig, StopFormat
from src.utils import cleanup, merge_update, save_file_exists, write_metadata

logger = logging.getLogger("src")

"""
"RED": 0,
"BLUE": 1,
"GREEN": 2,
"BLACK": 3
"""


def get_formula():
    f = FOC(Property("BLACK"))
    return f


def run_experiment(
        n_models: int,
        save_path: str,
        file_name: str,
        model_config: GNNModelConfig,
        data_config: Dict[str, Any],
        n_graphs: int,
        total_graphs: int,
        test_size: int,
        batch_size: int = 64,
        iterations: int = 100,
        gpu_num: int = 0,
        data_workers: int = 2,
        lr: float = 0.01,
        stop_when: StopFormat = None,
        unique_test: bool = True):

    logger.debug("Initializing graph stream")
    stream = graph_stream(**data_config)

    logger.info(f"Pre-generating database of {total_graphs} graphs")
    data_pool = RandomGraphDataset(stream, limit=total_graphs)
    logger.info("Finished pre-generating")

    seed = data_config.get("seed", None)
    logger.debug("Initializing subsampler")
    data_sampler = SubsetSampler(
        dataset=data_pool,
        n_elements=n_graphs,
        test_size=test_size,
        seed=seed,
        unique_test=unique_test)

    models = []
    stats = {
        "macro": defaultdict(int),
        "micro": defaultdict(int)
    }

    try:
        for m in range(1, n_models + 1):

            logger.info(f"Training model {m}/{n_models}")

            logger.debug("Subsampling dataset")
            train_data, test_data = data_sampler()
            logger.debug("Finished Subsampling dataset")

            model, metrics = run(
                run_config=Training(),
                model_config=model_config,
                train_data=train_data,
                test_data=test_data,
                iterations=iterations,
                gpu_num=gpu_num,
                data_workers=data_workers,
                batch_size=batch_size,
                test_batch_size=test_size,
                lr=lr,
                stop_when=stop_when)

            model.cpu()
            weights = clean_state(model.state_dict())
            models.append(weights)

            stats["macro"][str(round(metrics["macro"], 3))] += 1
            stats["micro"][str(round(metrics["micro"], 3))] += 1

    except KeyboardInterrupt:
        logger.info("Manually Interrumpted")
        _error_file = f"{save_path}/{file_name.format(len(models))}.error"
        with open(_error_file, "w") as o:
            o.write(f"Interrupted work in file {save_path}\n")
            o.write(f"Only {len(models)} models were written\n")
    except Exception as e:
        logger.error(f"Exception encountered: {type(e).__name__}")
        logger.error(f"Message: {e}")
        _error_file = f"{save_path}/{file_name.format(len(models))}.error"
        with open(_error_file, "w") as o:
            o.write(f"Problem in file {save_path}/{file_name.format('X')}\n")
            o.write(f"Exception encountered: {e}\n")
            o.write(f"Only {len(models)} models were written\n")
    finally:
        logger.info(f"Saving computed models...")

        exists, prev_file = save_file_exists(save_path, file_name)
        if exists:
            # ! this does not take care of race conditions
            logger.info("File already exists")
            logger.info("Appending new models to file")

            logger.debug("Loading previous models")
            prev_models = torch.load(f"{save_path}/{prev_file}")

            models.extend(prev_models)

            with open(f"{save_path}/{prev_file}.stat", "r") as f:
                prev_stats = json.load(f)
                stats = merge_update(stats, prev_stats)

        logger.info(f"Saving {len(models)} models...")
        models_file = f"{save_path}/{file_name.format(len(models))}"
        torch.save(models, models_file)
        with open(f"{models_file}.stat", "w") as f:
            json.dump(stats, f, sort_keys=True, indent=2)

        cleanup(exists, save_path, prev_file)


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
        "hidden_dim": 8,
        "hidden_layers": None,
        "output_dim": 2,
        "aggregate_type": "add",
        "combine_type": "identity",
        "num_layers": 2,
        "mlp_layers": 1,  # the number of layers in A and V
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
        "stay": 2
    }

    # total graphs to pre-generate
    total_graphs = 300_000
    # graphs selected per training session / model
    n_graphs = 5120
    # how many graphs are selected for the testing
    test_size = 500
    # the size of the training batch
    batch_size = 128
    # if true, the test set is generated only one time and all models are
    # tested against that
    unique_test = True

    write_metadata(
        destination=f"{save_path}/.meta.csv",
        model_config=model_config,
        model_config_hash=model_config_hash,
        formula=formula,
        formula_hash=formula_hash,
        formula_fn=get_formula,
        data_config=data_config,
        iterations=iterations,
        total_graphs=total_graphs,
        n_graphs=n_graphs,
        batch_size=batch_size,
        test_size=test_size,
        seed=seed,
        stop_when=stop_when,
        unique_test=unique_test
    )

    data_config["formula"] = formula

    start = timer()
    run_experiment(
        n_models=n_models,
        save_path=save_path,
        file_name=file_name,
        model_config=model_config,
        data_config=data_config,
        n_graphs=n_graphs,
        total_graphs=total_graphs,
        test_size=test_size,
        batch_size=batch_size,
        iterations=iterations,
        gpu_num=0,
        data_workers=2,
        lr=0.01,
        stop_when=stop_when,
        unique_test=unique_test
    )
    end = timer()
    logger.info(f"Took {end-start} seconds")


if __name__ == "__main__":
    logger.setLevel(logging.DEBUG)

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    _console_f = logging.Formatter("%(levelname)-8s: %(message)s")
    ch.setFormatter(_console_f)

    # fh = logging.FileHandler("main_1.2.log")
    # fh.setLevel(logging.DEBUG)
    # _file_f = logging.Formatter(
    #     '%(asctime)s %(name)s %(levelname)s "%(message)s"')
    # fh.setFormatter(_file_f)

    logger.addHandler(ch)
    # logger.addHandler(fh)
    main()
