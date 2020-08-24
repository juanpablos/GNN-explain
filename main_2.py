import logging
import os
import random
from timeit import default_timer as timer
from typing import List

import torch
from sklearn.metrics import classification_report

from src.data.formula_index import FormulaMapping
from src.data.formulas import *
from src.data.loader import load_gnn_files
from src.data.utils import (
    get_input_dim,
    get_label_distribution,
    train_test_dataset
)
from src.run_logic import run, seed_everything
from src.training.mlp_training import Training
from src.typing import MinModelConfig, NetworkDataConfig
from src.visualization.confusion_matrix import plot_confusion_matrix
from src.visualization.curve_plot import plot_training

logger = logging.getLogger("src")


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
        lr: float = 0.01,
        run_train_test: bool = False,
        results_path: str = "./results",
        model_name: str = None,
        plot_file_name: str = None,
        plot_title: str = None,
        _legacy_load_without_batch: bool = False
):

    logger.info("Loading Files")
    dataset, label_mapping = load_gnn_files(
        **data_config, _legacy_load_without_batch=_legacy_load_without_batch)
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

    if model_name is not None:
        model.cpu()
        os.makedirs(f"{results_path}/models/", exist_ok=True)
        torch.save(
            model.state_dict(),
            f"{results_path}/models/{model_name}.pt")

    # * get the last evaluation values
    _y = train_state.metrics.acc_y
    _y_pred = train_state.metrics.acc_y_pred

    try:
        formula_mapping = FormulaMapping("./data/formulas.json")
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

    if plot_file_name is not None:
        plot_confusion_matrix(
            _y,
            _y_pred,
            save_path=results_path,
            file_name=plot_file_name,
            title=plot_title,
            labels=target_names,
            each_label=each_label)

        metrics = train_state.get_metric_logger()
        plot_training(
            metric_history=metrics,
            save_path=results_path,
            file_name=plot_file_name,
            title=plot_title,
            use_selected=False)


def main(
        name: str = None,
        seed: int = None,
        train_batch: int = 32,
        lr: float = 0.001,
        hidden_layers: List[int] = None,
        save_model: bool = True,
        make_plots: bool = True):

    if seed is None:
        seed = random.randint(1, 1 << 30)
    seed_everything(seed)

    hidden_layers = [128] if hidden_layers is None else hidden_layers

    model_config: MinModelConfig = {
        "num_layers": 3,
        "input_dim": None,
        "hidden_dim": 128,
        "hidden_layers": hidden_layers,
        "output_dim": None,
        "use_batch_norm": True
    }

    model_hash = "f4034364ea-nosave"  # "f4034364ea"
    # filters = []
    # selector = FilterApply(condition="and")
    selector = SelectFilter(hashes=[
        "dc670b1bec",
        "4805042859",
        "688d12b701",
        "652c706f1b"
    ])
    label_logic = BinaryAtomicLabeler(atomic="RED")
    labeler = LabelerApply(labeler=label_logic)
    data_config: NetworkDataConfig = {
        "root": "data/gnns",
        "model_hash": model_hash,
        "selector": selector,
        "labeler": labeler,
        "mapping": FormulaMapping("./data/formulas.json")
    }

    iterations = 20
    test_batch = 512

    if name is None:
        name = "atomic"

    hid = "+".join(
        [f"{l}L{val}" for l, val in enumerate(hidden_layers, start=1)])
    msg = f"{name}-{hid}-{train_batch}b-{lr}lr"

    results_path = f"./results/exp2/{model_hash}"
    plot_file = None
    if make_plots:
        plot_file = msg
    model_name = None
    if save_model:
        model_name = msg

    start = timer()
    run_experiment(
        model_config=model_config,
        data_config=data_config,
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
        results_path=results_path,
        model_name=model_name,
        plot_file_name=plot_file,
        plot_title=msg,  # ? maybe a better message
        _legacy_load_without_batch=True  # ! remove eventually
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

    __layers = [2048]
    # for __batch in [128, 256, 512]:
    #     for __lr in [0.0005, 0.001, 0.005]:

    #         logger.info(f"Running NN config: batch: {__batch}, "
    #                     f"lr: {__lr}, layers: {__layers}")
    #         main(seed=42, train_batch=__batch, lr=__lr, hidden_layers=__layers)
    main(
        seed=0,
        train_batch=128,
        lr=0.005,
        hidden_layers=__layers,
        save_model=True,
        make_plots=False
    )
