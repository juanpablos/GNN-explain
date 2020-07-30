import logging
import random
from timeit import default_timer as timer

from src.data.utils import get_input_dim, get_label_distribution, load_gnn_files, train_test_dataset
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

    if stratify:
        logging.debug(
            f"Dataset distribution {get_label_distribution(dataset)}")
    else:
        logging.debug(
            f"Train dataset distribution {get_label_distribution(train_data)}")
        logging.debug(
            f"Test dataset distribution {get_label_distribution(test_data)}")

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
        "hidden_layers": [4096, 512],
        "output_dim": 2
    }

    data_config: NetworkDataConfig = {
        "root": "data/gnns",
        "model_hash": "0d7e1554fa-add2",
        # * if load_all is true formula_hashes is ignored and each formula in the directory receives a different label
        "load_all": False,
        "formula_hashes": {
            # "5caab97089": {  # (black|green) and 3-5 blue neigh
            #     "limit": None,
            #     "label": 0
            # },
            # "7e24cdcffb": {  # red
            #     "limit": None,
            #     "label": 0
            # },
            # "74a0324f6e": {  # green
            #     "limit": None,
            #     "label": 0
            # },
            # "45207fda29": {  # OR(X and 4+ X neigh), X [red, blue, green, black]
            #     "limit": None,
            #     "label": 0
            # },
            # "a085814a6b": {  # blue & 2+ green neigh
            #     "limit": None,
            #     "label": 0
            # },
            # "b18de7fd2c": {  # green & ((2-4 blue neigh) | (4-6 red neigh))
            #     "limit": None,
            #     "label": 0
            # },
            "bfa11bd667": {  # red & 2-4 (black|blue) neigh
                "limit": None,
                "label": 0
            },
            "c716a094ab": {  # blue|green
                "limit": None,
                "label": 1
            }
        }
    }

    iterations = 100
    train_batch = 8
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
        lr=0.001
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
