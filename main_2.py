import logging
import os
import random
from timeit import default_timer as timer
from typing import List

import torch
from sklearn.metrics import classification_report

from src.data.formula_index import FormulaMapping
from src.data.formulas import *
from src.data.formulas.labeler import MultiLabelCategoricalLabeler
from src.data.loader import categorical_loader
from src.data.utils import get_input_dim, get_label_distribution, train_test_dataset
from src.eval_utils import evaluate_model
from src.run_logic import run, seed_everything
from src.training.mlp_training import MLPTrainer
from src.typing import MinModelConfig, NetworkDataConfig
from src.utils import write_result_info
from src.visualization.confusion_matrix import (
    plot_confusion_matrix,
    plot_multilabel_confusion_matrix,
)
from src.visualization.curve_plot import plot_training

logger = logging.getLogger("src")
logger_metrics = logging.getLogger("metrics")


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
    plot_filename: str = None,
    plot_title: str = None,
    info_filename: str = "info",
    multilabel: bool = True,
    _legacy_load_without_batch: bool = False,
):

    logger.info("Loading Files")
    # class_mapping: label_id -> label_name
    # hash_formula: formula_hash -> formula_object
    # hash_label:
    #   single label: formula_hash -> label_id
    #   multilabel: formula_hash -> List[label_id]
    # data_reconstruction: point_index -> formula_object
    (
        datasets,
        class_mapping,
        hash_formula,
        hash_label,
        data_reconstruction,
    ) = categorical_loader(
        **data_config, _legacy_load_without_batch=_legacy_load_without_batch
    )

    if isinstance(datasets, tuple):
        logger.debug("Using selected data as test")
        # * only here because return type problems when **[TypedDict]
        train_data, test_data = datasets
    else:
        logger.debug("Splitting data")
        train_data, test_data = train_test_dataset(
            dataset=datasets,
            test_size=test_size,
            random_state=seed,
            shuffle=True,
            stratify=stratify,
            multilabel=multilabel,
        )

    n_classes = len(class_mapping)
    logger.debug(f"{n_classes} classes detected")

    _, train_distribution = get_label_distribution(train_data)
    test_label_count, test_distribution = get_label_distribution(test_data)
    logger.debug(f"Train dataset distribution {train_distribution}")
    logger.debug(f"Test dataset distribution {test_distribution}")

    input_shape = get_input_dim(train_data)
    assert len(input_shape) == 1, "The input dimension is different from 1"

    model_config["input_dim"] = input_shape[0]
    model_config["output_dim"] = n_classes

    # --- metrics logger
    os.makedirs(os.path.join(results_path, "info"), exist_ok=True)
    fh = logging.FileHandler(
        os.path.join(results_path, "info", f"{info_filename}.log"), mode="w"
    )
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(asctime)s,%(message)s"))
    logger_metrics.addHandler(fh)
    # /--- metrics logger

    trainer = MLPTrainer(
        logging_variables="all",
        n_classes=n_classes,
        metrics_average="macro",
        multilabel=multilabel,
    )

    trainer.init_dataloader(
        train_data,
        mode="train",
        batch_size=batch_size,
        pin_memory=False,
        shuffle=True,
        num_workers=data_workers,
    )
    trainer.init_dataloader(
        test_data,
        mode="test",
        batch_size=test_batch_size,
        pin_memory=False,
        shuffle=True,
        num_workers=data_workers,
    )

    trainer.init_model(**model_config)

    logger.debug("Running")
    logger.debug(f"Input size is {input_shape[0]}")
    (model,) = run(
        trainer=trainer,
        iterations=iterations,
        gpu_num=gpu_num,
        lr=lr,
        run_train_test=run_train_test,
    )

    # mistakes is a counter for each formula for each label mistake in test:
    # mistakes: formula -> (int -> int)
    # formula_count is a counter for each formula for each label in the test set
    # formula_count: formula -> (int -> int)
    _y, _y_pred, mistakes, formula_count = evaluate_model(
        model=model,
        test_data=test_data,
        reconstruction=data_reconstruction,
        trainer=trainer,
        gpu=gpu_num,
        multilabel=multilabel,
    )

    # returns a number to put after the file name in case it already exists
    # "" or " (N)"
    ext = write_result_info(
        path=results_path,
        filename=info_filename,
        hash_formula=hash_formula,
        hash_label=hash_label,
        classes=class_mapping,
        multilabel=multilabel,
        mistakes=mistakes,
        formula_count=formula_count,
        metrics=trainer.metrics.report(),
    )

    if model_name is not None:
        logger.debug("Writing model")
        model.cpu()
        os.makedirs(f"{results_path}/models/", exist_ok=True)
        obj = {"model": model.state_dict(), "class_mapping": class_mapping}
        torch.save(obj, f"{results_path}/models/{model_name}{ext}.pt")

    # class_mapping is an ordered dict
    target_names = list(class_mapping.values())
    # * no problem with multilabel as is
    logger.debug("Printing classification report")
    print(classification_report(_y, _y_pred, target_names=target_names))

    if plot_filename is not None:
        if multilabel:
            label_numbers = [test_label_count[i] for i in class_mapping]
            plot_multilabel_confusion_matrix(
                _y,
                _y_pred,
                save_path=results_path,
                labels=list(class_mapping.values()),
                label_totals=label_numbers,
                filename=plot_filename + ext,
                title=plot_title,
            )
        else:
            cm_labels = [
                f"{label_name} ({test_label_count.get(label, 0)})"
                for label, label_name in class_mapping.items()
            ]
            plot_confusion_matrix(
                _y,
                _y_pred,
                save_path=results_path,
                filename=plot_filename + ext,
                title=plot_title,
                labels=cm_labels,
                normalize_cm=True,
            )

        metrics = trainer.metric_logger
        plot_training(
            metric_history=metrics,
            save_path=results_path,
            filename=plot_filename + ext,
            title=plot_title,
            use_selected=False,
        )


def main(
    name: str = None,
    seed: int = None,
    train_batch: int = 32,
    lr: float = 0.001,
    hidden_layers: List[int] = None,
    save_model: bool = True,
    make_plots: bool = True,
):

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
        "use_batch_norm": True,
    }

    model_hash = "40e65407aa"

    # * filters
    # selector = FilterApply(condition="or")
    # selector.add(AtomicFilter(atomic="all"))
    # selector.add(RestrictionFilter(lower=1, upper=2))
    # selector.add(RestrictionFilter(lower=None, upper=-1))
    # selector = SelectFilter(
    #     hashes=["dc670b1bec", "4805042859", "688d12b701", "652c706f1b"]
    # )
    selector = NoFilter()
    # * /filters

    # * test_filters
    # test_selector = FilterApply(condition="or")
    # test_selector.add(AtomicOnlyFilter(atomic="all"))
    # test_selector.add(RestrictionFilter(lower=4, upper=None))
    test_selector = SelectFilter(
        hashes=[
            "22609b6219",
            "d376f80fe0",
            "4865ca5688",
            "b739521345",
            "98e4690a6c",
            "fd1ede286c",
            "56dc8827b8",
            "c1eec67813",
            "8500dc307e",
            "530867a9ca",
        ]
    )
    # test_selector = NullFilter()
    # * /test_filters

    # * labelers
    # label_logic = MultiLabelAtomicLabeler()
    label_logic = MultilabelRestrictionLabeler(mode="both")
    labeler = LabelerApply(labeler=label_logic)
    # * /labelers
    data_config: NetworkDataConfig = {
        "root": "data/gnns_v2",
        "model_hash": model_hash,
        "selector": selector,
        "labeler": labeler,
        "formula_mapping": FormulaMapping("./data/formulas.json"),
        "test_selector": test_selector,
        "load_aggregated": "aggregated.pt",
        "force_preaggregated": False,
    }

    iterations = 10
    test_batch = 1024

    if name is None:
        name = f"{selector}-{labeler}-{test_selector}"

    hid = "+".join([f"{l}L{val}" for l, val in enumerate(hidden_layers, start=1)])
    msg = f"{name}-{hid}-{train_batch}b-{lr}lr"

    results_path = f"./results/v2_clean/testing/{model_hash}"
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
        plot_filename=plot_file,
        plot_title=msg,  # ? maybe a better message
        info_filename=msg,
        # * this should only be available when binary in experiment 3
        multilabel=isinstance(label_logic, MultiLabelCategoricalLabeler),
        _legacy_load_without_batch=True,  # ! remove eventually
    )
    end = timer()
    logger.info(f"Took {end-start} seconds")


if __name__ == "__main__":
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    logger_metrics.setLevel(logging.INFO)
    logger_metrics.propagate = False

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    _console_f = logging.Formatter("%(levelname)-8s: %(message)s")
    ch.setFormatter(_console_f)

    logger.addHandler(ch)

    __layers = [256, 256, 256]
    # for __batch in [128, 256, 512]:
    #     for __lr in [0.0005, 0.001, 0.005]:

    #         logger.info(f"Running NN config: batch: {__batch}, "
    #                     f"lr: {__lr}, layers: {__layers}")
    #         main(seed=42, train_batch=__batch, lr=__lr, hidden_layers=__layers)
    main(
        seed=0,
        train_batch=64,
        lr=0.005,
        hidden_layers=__layers,
        save_model=True,
        make_plots=True,
    )
