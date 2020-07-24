import logging
import random
from timeit import default_timer as timer

from src.data.utils import get_input_dim, load_gnn_files, train_test_dataset
from src.run_logic import run, seed_everything
from src.training.mlp_training import Training
from src.typing import MinModelConfig, NetworkDataConfig


def run_experiment(
        model_config: MinModelConfig,
        data_config: NetworkDataConfig,
        iterations: int = 100,
        gpu_num: int = 0,
        seed: int = 10,
        test_size: float = 0.25,
        stratify: bool = True,
        data_workers: int = 2,
        batch_size: int = 64,
        test_batch_size: int = 512,
        lr: float = 0.01
):

    logging.info("Loading Files")
    dataset = load_gnn_files(**data_config)

    logging.debug("Splitting data")
    train_data, test_data = train_test_dataset(dataset=dataset,
                                               test_size=test_size,
                                               random_state=seed,
                                               shuffle=True,
                                               stratify=stratify)

    input_shape = get_input_dim(train_data)
    assert len(input_shape) == 1, "The input dimension is different from 1"
    model_config["input_dim"] = input_shape[0]

    logging.debug("Running")
    logging.debug(f"Input size is {input_shape[0]}")
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
        lr=lr)


def main():
    seed = random.randint(1, 1 << 30)
    seed_everything(seed)

    model_config: MinModelConfig = {
        "num_layers": 3,
        "input_dim": None,
        "hidden_dim": 2048,
        "hidden_layers": None,
        "output_dim": 2
    }

    # TODO: implement all, to configure all the files at the same time
    # this would assign a particular label to each file
    data_config: NetworkDataConfig = {
        "root": "data/gnns",
        "model_hash": "9100982dba",
        "formula_hashes": {
            # "e7901521fb": {  # red
            #     "limit": None,
            #     "label": 0
            # },
            "ff2e1a9328": {  # blue or green
                "limit": None,
                "label": 0
            },
            "ea9c5d0ff4": {  # not blue
                "limit": None,
                "label": 0
            },
            "d747ceb5a2": {  # blue and green neighbor
                "limit": None,
                "label": 1
            }
        }
    }

    iterations = 100
    train_batch = 16
    test_batch = 512

    start = timer()
    run_experiment(
        model_config=model_config,
        data_config=data_config,
        iterations=iterations,
        gpu_num=0,
        seed=seed,
        test_size=0.20,
        stratify=True,
        data_workers=2,
        batch_size=train_batch,
        test_batch_size=test_batch,
        lr=0.01
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
        handlers=[
            _console
        ]
    )

    main()
