import random
from typing import Any, Dict

from src.data.utils import get_input_dim, load_gnn_files, train_test_dataset
from src.run_logic import run, seed_everything
from src.training.mlp_training import Training


def run_experiment(
        model_config: Dict[str, Any],
        data_config: Dict[str, Any],
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

    print("Loading Files")
    dataset = load_gnn_files(**data_config)

    print("Splitting data")
    train_data, test_data = train_test_dataset(dataset,
                                               test_size=test_size,
                                               random_state=seed,
                                               shuffle=True,
                                               stratify=stratify)

    input_shape = get_input_dim(train_data)
    assert len(input_shape) == 1, "The input dimension is different from 1"
    model_config["input_dim"] = input_shape[0]

    print("Running")
    model, metrics = run(
        run_config=Training,
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

    model_config = {
        "num_layers": 2,
        "input_dim": ...,
        "hidden_dim": 64,
        "output_dim": 2
    }
    data_config = {
        "root": "data/gnns",
        "model_hash": "1dd2071cd3",
        "formula_hashes": {
            "e7901521fb": {  # red
                "limit": None,
                "label": 0
            },
            "bc152d5133": {  # green
                "limit": None,
                "label": 0
            },
            "81c2571aae": {  # black
                "limit": None,
                "label": 0
            },
            "ff2e1a9328": {  # blue or green
                "limit": None,
                "label": 0
            },
            "ea9c5d0ff4": {  # not blue
                "limit": None,
                "label": 0
            },
            "5015e55dc6": {  # red or green
                "limit": None,
                "label": 1
            },
            "8bd614c47b": {  # blue
                "limit": None,
                "label": 0
            }
        }
    }

    iterations = 100
    train_batch = 128
    test_batch = 1024

    from timeit import default_timer as timer
    start = timer()
    run_experiment(
        model_config=model_config,
        data_config=data_config,
        iterations=iterations,
        gpu_num=0,
        seed=seed,
        test_size=0.25,
        stratify=True,
        data_workers=2,
        batch_size=train_batch,
        test_batch_size=test_batch,
        lr=0.01
    )
    end = timer()
    print(f"Took {end - start} seconds")


if __name__ == "__main__":
    main()
