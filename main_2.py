import logging
import random
from timeit import default_timer as timer
from typing import List

from sklearn.metrics import classification_report

from src.data.loader import FormulaConfig, load_gnn_files
from src.data.utils import (
    get_input_dim,
    get_label_distribution,
    train_test_dataset
)
from src.formula_index import FormulaMapping
from src.run_logic import run, seed_everything
from src.training.mlp_training import Training
from src.typing import MinModelConfig, NetworkDataConfig, S
from src.visualization.confusion_matrix import plot_confusion_matrix
from src.visualization.curve_plot import plot_training

logger = logging.getLogger("src")


def run_experiment(
        model_config: MinModelConfig,
        data_config: NetworkDataConfig,
        formulas_to_load: List[FormulaConfig[S]],
        iterations: int = 100,
        gpu_num: int = 0,
        seed: int = 10,
        test_size: float = 0.25,
        stratify: bool = True,
        data_workers: int = 2,
        batch_size: int = 64,
        test_batch_size: int = 512,
        lr: float = 0.01,
        run_train_test: bool = False,
        plot_path: str = "./results",
        plot_file_name: str = None,
        plot_title: str = None,
        plot_historic: bool = True
):

    logger.info("Loading Files")
    dataset, label_mapping = load_gnn_files(
        formulas=formulas_to_load, **data_config)
    n_classes = len(dataset.label_info)
    logger.debug(f"{n_classes} classes detected")

    logger.debug("Splitting data")
    train_data, test_data = train_test_dataset(dataset=dataset,
                                               test_size=test_size,
                                               random_state=seed,
                                               shuffle=True,
                                               stratify=stratify)

    if stratify:
        logger.debug(
            f"Dataset distribution {get_label_distribution(dataset)}")
    else:
        logger.debug(
            f"Train dataset distribution {get_label_distribution(train_data)}")
        logger.debug(
            f"Test dataset distribution {get_label_distribution(test_data)}")

    input_shape = get_input_dim(train_data)
    assert len(input_shape) == 1, "The input dimension is different from 1"

    model_config["input_dim"] = input_shape[0]
    model_config["output_dim"] = n_classes

    train_state = Training(n_classes=n_classes, logging_variables="all")

    logger.debug("Running")
    logger.debug(f"Input size is {input_shape[0]}")
    model = run(
        run_config=train_state,
        model_config=model_config,
        train_data=train_data,
        test_data=test_data,
        iterations=iterations,
        gpu_num=gpu_num,
        data_workers=data_workers,
        batch_size=batch_size,
        test_batch_size=test_batch_size,
        lr=lr,
        run_train_test=run_train_test)

    # * get the last evaluation values
    _y = train_state.metrics.acc_y
    _y_pred = train_state.metrics.acc_y_pred

    try:
        formula_mapping = FormulaMapping("src/formulas.json")
        label_formula = {label: str(formula_mapping[h])
                         for h, label in label_mapping.items()}
    except Exception as e:
        logger.error("Exception encountered when using formula mapping")
        logger.error("Message:", e)
        # fallback when cannot use the formula mapping
        label_formula = {label: str(label) for label in label_mapping.values()}

    target_names = [label_formula[k] for k in sorted(label_formula)]
    print(classification_report(_y, _y_pred, target_names=target_names))
    # print(json.dumps(label_formula, indent=2, ensure_ascii=False))

    test_label_info = test_data.apply_subset().label_info
    each_label = next(iter(test_label_info.values()))

    plot_confusion_matrix(
        _y,
        _y_pred,
        save_path=plot_path,
        file_name=plot_file_name,
        title=plot_title,
        labels=target_names,
        each_label=each_label)

    if plot_historic:
        metrics = train_state.get_metric_logger()
        plot_training(
            metric_history=metrics,
            save_path=plot_path,
            file_name=plot_file_name,
            title=plot_title,
            use_selected=False)


def main(
        train_batch: int = 32,
        lr: float = 0.001,
        hidden_layers: List[int] = None):

    seed = random.randint(1, 1 << 30)
    seed_everything(seed)

    hidden_layers = [128] if hidden_layers is None else hidden_layers

    model_config: MinModelConfig = {
        "num_layers": 3,
        "input_dim": None,
        "hidden_dim": 128,
        "hidden_layers": hidden_layers,
        "output_dim": None
    }

    data_config: NetworkDataConfig = {
        "root": "data/gnns",
        "model_hash": "6106dbd778",  # "6106dbd778",
        # * if load_all is true formula_hashes is ignored and each formula in the directory receives a different label
        "load_all": False
    }
    formulas = FormulaConfig.from_hashes([
        "ea81181317",
        "9eb3668544",
        "9dfdcfb080",
        "2231100a27"
    ])

    iterations = 20
    test_batch = 512

    name = "red4+"
    hid = "".join(
        [f"{l}l{val}" for l, val in enumerate(hidden_layers, start=1)])
    msg = f"{name}-{hid}-{train_batch}b-{lr}lr"

    start = timer()
    run_experiment(
        model_config=model_config,
        data_config=data_config,
        formulas_to_load=formulas,
        iterations=iterations,
        gpu_num=0,
        seed=seed,
        test_size=0.20,
        stratify=True,
        data_workers=0,
        batch_size=train_batch,
        test_batch_size=test_batch,
        lr=lr,
        run_train_test=True,
        plot_path="./results/testing",
        plot_file_name=msg,
        plot_title=msg  # maybe a better message
    )
    end = timer()
    logger.info(f"Took {end-start} seconds")


if __name__ == "__main__":
    logger.setLevel(logging.DEBUG)
    # logger.propagate = False

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    _console_f = logging.Formatter("%(levelname)-8s: %(message)s")
    ch.setFormatter(_console_f)

    # fh = logging.FileHandler("main_2.log")
    # fh.setLevel(logging.DEBUG)
    # _file_f = logging.Formatter(
    #     '%(asctime)s %(filename) %(name)s %(levelname)s "%(message)s"')
    # fh.setFormatter(_file_f)

    logger.addHandler(ch)
    # logger.addHandler(fh)

    for __batch in [8, 16, 32, 64, 128]:
        for __lr in [0.001, 0.005, 0.01]:
            main(train_batch=__batch, lr=__lr, hidden_layers=[512])

    # main()
