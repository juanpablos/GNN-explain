import json
import logging
import os
import random
from timeit import default_timer as timer
from typing import Dict, List, Literal, Optional, Union

import torch
from torch.functional import Tensor

from src.data.auxiliary import NetworkDatasetCollectionWrapper
from src.data.dataset_splitter import NetworkDatasetCrossFoldSplitter
from src.data.datasets import LabeledDataset, LabeledSubset
from src.data.formula_index import FormulaMapping
from src.data.formulas import *
from src.data.loader import categorical_loader
from src.data.utils import get_input_dim, get_label_distribution, train_test_dataset
from src.graphs.foc import Element
from src.run_logic import run, seed_everything
from src.training.encoder_training import EncoderTrainer
from src.typing import (
    CrossFoldConfiguration,
    MinModelConfig,
    NetworkDataConfig,
    StopFormat,
)
from src.utils import get_next_filename, prediction_data_formatting
from src.visualization.curve_plot import plot_training
from src.visualization.umap import plot_embedding_2d

logger = logging.getLogger("src")
logger_metrics = logging.getLogger("metrics")


def _run_experiment(
    train_data: Union[LabeledDataset[Tensor, int], LabeledSubset[Tensor, int]],
    test_data: Union[LabeledDataset[Tensor, int], LabeledSubset[Tensor, int]],
    class_mapping: Dict[int, str],
    hash_formula: Dict[str, Element],
    hash_label: Dict[str, int],
    data_reconstruction: NetworkDatasetCollectionWrapper,
    model_config: MinModelConfig,
    evaluate_with_train: bool,
    iterations: int = 100,
    gpu_num: int = 0,
    data_workers: int = 2,
    trainer_loss_name: Literal[
        "contrastive",
        "triplet",
        "triplet_cosine",
        "lifted_structure",
        "angular",
        "ntxent",
    ] = "contrastive",
    trainer_miner_name: Literal[
        "none", "similarity", "triplet", "triplet_cosine"
    ] = "none",
    miner_pairs: Literal["semihard", "all", None] = "all",
    use_cross_batch: bool = True,
    use_m_per_class_sampler: bool = True,
    batch_size: int = 64,
    test_batch_size: int = 512,
    lr: float = 0.01,
    seed: int = None,
    early_stopping: StopFormat = None,
    run_train_test: bool = False,
    results_path: str = "./results",
    model_name: str = None,
    plot_filename: str = None,
    plot_title: str = None,
    info_filename: str = "info",
):

    logger.info("Running experiment")
    # class_mapping: label_id -> label_name
    # hash_formula: formula_hash -> formula_object
    # hash_label:
    #   single label: formula_hash -> label_id
    #   multilabel: formula_hash -> List[label_id]
    # data_reconstruction: point_index -> formula_object

    logger.debug(f"Train dataset size {len(train_data)}")
    logger.debug(f"Test dataset size {len(test_data)}")

    _, train_distribution = get_label_distribution(train_data)
    _, test_distribution = get_label_distribution(test_data)
    logger.debug(f"Train dataset distribution {train_distribution}")
    logger.debug(f"Test dataset distribution {test_distribution}")

    input_shape = get_input_dim(train_data)
    assert len(input_shape) == 1, "The input dimension is different from 1"

    model_config["input_dim"] = input_shape[0]

    # --- metrics logger
    logger_metrics.handlers = []
    os.makedirs(os.path.join(results_path, "info"), exist_ok=True)
    fh = logging.FileHandler(
        os.path.join(results_path, "info", f"{info_filename}.log"), mode="w"
    )
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(asctime)s,%(message)s"))
    logger_metrics.addHandler(fh)
    # /--- metrics logger

    trainer = EncoderTrainer(
        evaluate_with_train=evaluate_with_train,
        logging_variables="all",
        loss_name=trainer_loss_name,
        miner_name=trainer_miner_name,
        miner_pairs=miner_pairs,
        use_cross_batch=use_cross_batch,
        use_m_per_class_sampler=use_m_per_class_sampler,
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
        shuffle=False,
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
        stop_when=early_stopping,
        run_train_test=run_train_test,
    )

    _, ext = get_next_filename(
        path=os.path.join(results_path, "models"),
        filename=model_name if model_name else "",
        ext="pt",
    )

    if model_name is not None:
        logger.debug("Writing model")
        model.cpu()

        os.makedirs(os.path.join(results_path, "models"), exist_ok=True)
        obj = {"model": model.state_dict()}
        torch.save(obj, os.path.join(results_path, "models", f"{model_name}{ext}.pt"))

    os.makedirs(os.path.join(results_path, "extra_data"), exist_ok=True)
    extra_data_dict = {}
    for (
        extra_data_label,
        extra_data,
    ) in trainer.metric_logger.extra_iteration_data.items():
        extra_data_dict[extra_data_label] = prediction_data_formatting(
            prediction_dict=dict(enumerate(extra_data, start=1)),
            label_mapping=class_mapping,
        )
    with open(
        os.path.join(results_path, "extra_data", f"{model_name}{ext}.json"), "w"
    ) as f:
        json.dump(extra_data_dict, f, indent=4)

    if plot_filename is not None:
        trainer.move_model_to_device()

        logger.debug("Generating Embedding for train set")
        embedding, labels = trainer.get_embedding_with_label(
            train_data,
            batch_size=test_batch_size,
            pin_memory=False,
            shuffle=False,
            num_workers=data_workers,
        )
        logger.debug("Plotting Embedding for train set")
        plot_embedding_2d(
            embedding,
            labels,
            labels_categorical_mapping=class_mapping,
            save_path=results_path,
            filename=plot_filename + "_train" + ext,
            seed=seed,
        )

        logger.debug("Generating Embedding for test set")
        embedding, labels = trainer.get_embedding_with_label(
            test_data,
            batch_size=test_batch_size,
            pin_memory=False,
            shuffle=False,
            num_workers=data_workers,
        )
        logger.debug("Plotting Embedding for test set")
        plot_embedding_2d(
            embedding,
            labels,
            labels_categorical_mapping=class_mapping,
            save_path=results_path,
            filename=plot_filename + "_test" + ext,
            seed=seed,
        )

        metrics = trainer.metric_logger
        plot_training(
            metric_history=metrics,
            save_path=results_path,
            filename=plot_filename + ext,
            title=plot_title,
            use_selected=False,
        )

    return ext


def run_experiment(
    model_config: MinModelConfig,
    data_config: NetworkDataConfig,
    evaluate_with_train: bool,
    crossfold_config: Optional[CrossFoldConfiguration] = None,
    crossfold_fold_file: Optional[str] = None,
    iterations: int = 100,
    gpu_num: int = 0,
    seed: int = 10,
    test_size: float = 0.25,
    stratify: bool = True,
    data_workers: int = 2,
    trainer_loss_name: Literal[
        "contrastive",
        "triplet",
        "triplet_cosine",
        "lifted_structure",
        "angular",
        "ntxent",
    ] = "contrastive",
    trainer_miner_name: Literal[
        "none", "similarity", "triplet", "triplet_cosine"
    ] = "none",
    miner_pairs: Literal["semihard", "all", None] = "all",
    use_cross_batch: bool = True,
    only_run_for_first_cv: bool = False,
    use_m_per_class_sampler: bool = True,
    batch_size: int = 64,
    test_batch_size: int = 512,
    lr: float = 0.01,
    early_stopping: StopFormat = None,
    run_train_test: bool = False,
    results_path: str = "./results",
    model_name: str = None,
    plot_filename: str = None,
    plot_title: str = None,
    info_filename: str = "info",
):

    logger.info("Loading Files")
    # class_mapping: label_id -> label_name
    # hash_formula: formula_hash -> formula_object
    # hash_label:
    #   single label: formula_hash -> label_id
    #   multilabel: formula_hash -> List[label_id]
    # data_reconstruction: point_index -> formula_object
    # serialized_labeler: arbitrary dict of a serialized labeler classes and internals
    (
        datasets,
        class_mapping,
        hash_formula,
        hash_label,
        data_reconstruction,
        serialized_labeler,
    ) = categorical_loader(
        **data_config,
        cross_fold_configuration=crossfold_config,
        _legacy_load_without_batch=True,
    )

    if isinstance(datasets, NetworkDatasetCrossFoldSplitter):

        if crossfold_fold_file is not None:
            logger.info(f"Loading CV folds from file")
            with open(crossfold_fold_file) as f:
                precalculated_folds = json.load(f)
            datasets.load_precalculated_folds(fold_dict=precalculated_folds)

        logger.info(f"Total Dataset size: {datasets.dataset_size}")

        file_ext = ""
        n_splits = datasets.n_splits
        for i, (train_data, test_data, data_reconstruction) in enumerate(
            datasets, start=1
        ):
            if only_run_for_first_cv and i > 1:
                break

            logger.info(f"Running experiment for crossfold {i}/{n_splits}")

            cf_model_name = f"{model_name}_cf{i}"
            cf_plot_filename = f"{plot_filename}_cf{i}"
            cf_info_filename = f"{info_filename}_cf{i}"

            file_ext = _run_experiment(
                train_data=train_data,
                test_data=test_data,
                class_mapping=class_mapping,
                hash_formula=hash_formula,
                hash_label=hash_label,
                data_reconstruction=data_reconstruction,
                model_config=model_config,
                evaluate_with_train=evaluate_with_train,
                iterations=iterations,
                gpu_num=gpu_num,
                data_workers=data_workers,
                trainer_loss_name=trainer_loss_name,
                trainer_miner_name=trainer_miner_name,
                use_cross_batch=use_cross_batch,
                miner_pairs=miner_pairs,
                use_m_per_class_sampler=use_m_per_class_sampler,
                batch_size=batch_size,
                test_batch_size=test_batch_size,
                lr=lr,
                seed=seed,
                early_stopping=early_stopping,
                run_train_test=run_train_test,
                results_path=results_path,
                model_name=cf_model_name,
                plot_filename=cf_plot_filename,
                plot_title=plot_title,
                info_filename=cf_info_filename,
            )

        folds_file = os.path.join(results_path, "info", f"{model_name}{file_ext}.folds")
        with open(folds_file, "w", encoding="utf-8") as f:
            json.dump(datasets.get_folds(), f, ensure_ascii=False, indent=2)
    else:
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
                multilabel=False,
            )

        logger.info(f"Total Dataset size: {len(train_data) + len(test_data)}")

        file_ext = _run_experiment(
            train_data=train_data,
            test_data=test_data,
            class_mapping=class_mapping,
            hash_formula=hash_formula,
            hash_label=hash_label,
            data_reconstruction=data_reconstruction,
            model_config=model_config,
            evaluate_with_train=evaluate_with_train,
            iterations=iterations,
            gpu_num=gpu_num,
            data_workers=data_workers,
            trainer_loss_name=trainer_loss_name,
            trainer_miner_name=trainer_miner_name,
            use_cross_batch=use_cross_batch,
            miner_pairs=miner_pairs,
            use_m_per_class_sampler=use_m_per_class_sampler,
            batch_size=batch_size,
            test_batch_size=test_batch_size,
            lr=lr,
            seed=seed,
            early_stopping=early_stopping,
            run_train_test=run_train_test,
            results_path=results_path,
            model_name=model_name,
            plot_filename=plot_filename,
            plot_title=plot_title,
            info_filename=info_filename,
        )

    labeler_data_path = os.path.join(results_path, "labelers")
    os.makedirs(labeler_data_path, exist_ok=True)
    with open(
        os.path.join(labeler_data_path, f"{model_name}{file_ext}.labeler"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(serialized_labeler, f, ensure_ascii=False, indent=2)


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

    hidden_layers = [1024]
    output_size = 16

    model_config: MinModelConfig = {
        "num_layers": 3,
        "input_dim": None,
        "hidden_dim": 128,
        "hidden_layers": hidden_layers,
        "output_dim": output_size,
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
    # ! ignored if cross validation is used
    # test_selector = FilterApply(condition="or")
    # test_selector.add(AtomicOnlyFilter(atomic="all"))
    # test_selector.add(RestrictionFilter(lower=4, upper=None))
    # test_selector = SelectFilter(
    #     hashes=[
    #         "548c9f191e",
    #         "f0c2c63b89",
    #         "2d69688180",
    #         "ac7a72db0f",
    #         "da7c072589",
    #         "ddc0cfc54b",
    #         "eecfedc45d",
    #     ]
    # )
    test_selector = NullFilter()
    # * /test_filters

    # * labelers
    # give a different label for each formula
    label_logic = MulticlassQuantifierLimitLabeler(
        any_limit="lower",
        limits=[1, 2, 3, 4, 5],
        custom_name="1-5",
    )
    # label_logic = SequentialCategoricalLabeler()
    labeler = LabelerApply(labeler=label_logic)
    # * /labelers
    data_config: NetworkDataConfig = {
        "root": os.path.join("data", "gnns_v4"),
        "model_hash": model_hash,
        "selector": selector,
        "labeler": labeler,
        "formula_mapping": FormulaMapping(os.path.join("data", "formulas.json")),
        "test_selector": test_selector,
        "load_aggregated": "aggregated_raw.pt",
        "force_preaggregated": True,
    }
    # crossfold_config = None
    crossfold_config: CrossFoldConfiguration = {
        "n_splits": 5,  # not used
        "shuffle": True,  # not used
        "random_state": None,  # not used
        "defer_loading": True,
        "required_train_hashes": [],
        "use_stratified": None,
    }
    crossfold_fold_file = os.path.join(
        "results", "v4", "crossfold_raw", model_hash, "base.folds"
    )

    early_stopping: StopFormat = {
        "operation": "early_increase",
        "conditions": {"test_acc": 0.005},
        # "conditions": {"test_loss": 0.001},
        "stay": 5,
    }

    use_cross_batch = False
    trainer_loss_name: Literal[
        "contrastive",
        "triplet",
        "triplet_cosine",
        "lifted_structure",
        "angular",
        "ntxent",
    ] = "triplet"
    trainer_miner_name: Literal[
        "none", "triplet", "triplet_cosine", "similarity"
    ] = "triplet"
    miner_pairs: Literal["semihard", "all", None] = "all"

    only_run_for_first_cv = True

    iterations = 50
    test_batch = 128

    if name is None:
        test_selector_name = "CV" if crossfold_config else str(test_selector)
        name = f"{selector}-{labeler}-{test_selector_name}"

    hid = "+".join([f"{l}L{val}" for l, val in enumerate(hidden_layers, start=1)])
    msg = f"{name}-{hid}-O{output_size}-{train_batch}b-{lr}lr"

    results_path = os.path.join(
        "results",
        "v4",
        "crossfold_raw",
        model_hash,
        "encoders",
        "encoder_lower_v2_256",
        f"{trainer_miner_name}_{miner_pairs}" if miner_pairs else trainer_miner_name,
        f"{trainer_loss_name}_cross" if use_cross_batch else trainer_loss_name,
    )
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
        evaluate_with_train=not isinstance(label_logic, SequentialCategoricalLabeler),
        crossfold_config=crossfold_config,
        crossfold_fold_file=crossfold_fold_file,
        iterations=iterations,
        gpu_num=0,
        seed=seed,
        test_size=0.20,
        stratify=True,
        data_workers=0,
        trainer_loss_name=trainer_loss_name,
        trainer_miner_name=trainer_miner_name,
        use_cross_batch=use_cross_batch,
        miner_pairs=miner_pairs,
        only_run_for_first_cv=only_run_for_first_cv,
        use_m_per_class_sampler=isinstance(label_logic, SequentialCategoricalLabeler),
        batch_size=train_batch,
        test_batch_size=test_batch,
        lr=lr,
        early_stopping=early_stopping,
        run_train_test=True,
        results_path=results_path,
        model_name=model_name,
        plot_filename=plot_file,
        plot_title=msg,  # ? maybe a better message
        info_filename=msg,
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

    main(
        seed=0,
        train_batch=256,
        lr=1e-3,
        save_model=True,
        make_plots=True,
    )
