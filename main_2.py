import json
import logging
import os
import random
from timeit import default_timer as timer
from typing import Dict, Optional, Union

import torch
from sklearn.metrics import classification_report
from torch.functional import Tensor

from src.data.auxiliary import NetworkDatasetCollectionWrapper
from src.data.dataset_splitter import NetworkDatasetCrossFoldSplitter
from src.data.datasets import LabeledDataset, LabeledSubset
from src.data.formula_index import FormulaMapping
from src.data.formulas import *
from src.data.formulas.labeler import (
    BinaryCategoricalLabeler,
    MultiLabelCategoricalLabeler,
)
from src.data.loader import categorical_loader
from src.data.utils import get_input_dim, get_label_distribution, train_test_dataset
from src.eval_utils import evaluate_model
from src.graphs.foc import Element
from src.models.encoder_model_helper import EncoderModelHelper
from src.models.utils import count_parameters
from src.run_logic import run, seed_everything
from src.training.mlp_training import MLPTrainer
from src.typing import (
    CrossFoldConfiguration,
    EncoderModelConfigs,
    MinModelConfig,
    NetworkDataConfig,
    S,
    StopFormat,
)
from src.utils import write_result_info
from src.visualization.confusion_matrix import (
    plot_confusion_matrix,
    plot_multilabel_confusion_matrix,
)
from src.visualization.curve_plot import plot_training

logger = logging.getLogger("src")
logger_metrics = logging.getLogger("metrics")


def _run_experiment(
    train_data: Union[LabeledDataset[Tensor, S], LabeledSubset[Tensor, S]],
    test_data: Union[LabeledDataset[Tensor, S], LabeledSubset[Tensor, S]],
    class_mapping: Dict[S, str],
    hash_formula: Dict[str, Element],
    hash_label: Dict[str, S],
    data_reconstruction: NetworkDatasetCollectionWrapper,
    model_config: Optional[MinModelConfig],
    encoder_model_helper: Optional[EncoderModelHelper] = None,
    iterations: int = 100,
    gpu_num: int = 0,
    data_workers: int = 2,
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
    binary_labels: bool = True,
    multilabel: bool = True,
):

    logger.info("Running experiment")
    # class_mapping: label_id -> label_name
    # hash_formula: formula_hash -> formula_object
    # hash_label:
    #   single label: formula_hash -> label_id
    #   multilabel: formula_hash -> List[label_id]
    # data_reconstruction: point_index -> formula_object

    n_classes = len(class_mapping)
    logger.debug(f"{n_classes} classes detected")

    _, train_distribution = get_label_distribution(train_data)
    test_label_count, test_distribution = get_label_distribution(test_data)
    logger.debug(f"Train dataset size {len(train_data)}")
    logger.debug(f"Train dataset distribution {train_distribution}")
    logger.debug(f"Test dataset size {len(test_data)}")
    logger.debug(f"Test dataset distribution {test_distribution}")
    logger.info(f"classes: {class_mapping}")

    input_shape = get_input_dim(train_data)
    assert len(input_shape) == 1, "The input dimension is different from 1"

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

    trainer = MLPTrainer(
        logging_variables="all",
        n_classes=n_classes,
        metrics_average="binary" if binary_labels else "macro",
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
        shuffle=False,
        num_workers=data_workers,
    )

    if encoder_model_helper is not None:
        input_size = input_shape[0]
        if model_config is not None:
            encoder_model_helper.add_simple_encoder(
                encoder_config={**model_config, "input_dim": input_size}
            )
        trainer.init_model(
            use_encoder=True,
            model_helper=encoder_model_helper,
            model_input_size=input_size,
            model_output_size=n_classes,
        )
        logger.info("Using encoder model")
    else:
        assert model_config is not None
        model_config["input_dim"] = input_shape[0]
        model_config["output_dim"] = n_classes
        trainer.init_model(use_encoder=False, **model_config)

    # log model sizes
    (_model,) = trainer.get_models()
    total_parameters, grad_parameters = count_parameters(_model)
    logger.info(f"Model Parameters: {total_parameters}")
    logger.info(f"Model Grad Parameters: {grad_parameters}")

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
                plot_precision_and_recall=True,
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
    model_config: Optional[MinModelConfig],
    data_config: NetworkDataConfig,
    crossfold_config: CrossFoldConfiguration = None,
    crossfold_fold_file: Optional[str] = None,
    only_run_for_first_cv: bool = False,
    iterations: int = 100,
    gpu_num: int = 0,
    seed: int = 10,
    test_size: float = 0.25,
    stratify: bool = True,
    data_workers: int = 2,
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
    binary_labels: bool = True,
    multilabel: bool = True,
    use_encoders: bool = False,
    encoders_configs: Optional[EncoderModelConfigs] = None,
    _legacy_load_without_batch: bool = False,
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
        _legacy_load_without_batch=_legacy_load_without_batch,
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

            encoder_helper = None
            if use_encoders:
                assert encoders_configs is not None
                encoder_helper = EncoderModelHelper(
                    encoder_model_configs=encoders_configs, current_cv_iteration=i
                )

            file_ext = _run_experiment(
                train_data=train_data,
                test_data=test_data,
                class_mapping=class_mapping,
                hash_formula=hash_formula,
                hash_label=hash_label,
                data_reconstruction=data_reconstruction,
                model_config=model_config,
                encoder_model_helper=encoder_helper,
                iterations=iterations,
                gpu_num=gpu_num,
                data_workers=data_workers,
                batch_size=batch_size,
                test_batch_size=test_batch_size,
                lr=lr,
                early_stopping=early_stopping,
                run_train_test=run_train_test,
                results_path=results_path,
                model_name=cf_model_name,
                plot_filename=cf_plot_filename,
                plot_title=plot_title,
                info_filename=cf_info_filename,
                binary_labels=binary_labels,
                multilabel=multilabel,
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
                multilabel=multilabel,
            )

        logger.info(f"Total Dataset size: {len(train_data) + len(test_data)}")

        encoder_helper = None
        if use_encoders:
            assert encoders_configs is not None
            encoder_helper = EncoderModelHelper(
                encoder_model_configs=encoders_configs, current_cv_iteration=None
            )

        file_ext = _run_experiment(
            train_data=train_data,
            test_data=test_data,
            class_mapping=class_mapping,
            hash_formula=hash_formula,
            hash_label=hash_label,
            data_reconstruction=data_reconstruction,
            model_config=model_config,
            encoder_model_helper=encoder_helper,
            iterations=iterations,
            gpu_num=gpu_num,
            data_workers=data_workers,
            batch_size=batch_size,
            test_batch_size=test_batch_size,
            lr=lr,
            early_stopping=early_stopping,
            run_train_test=run_train_test,
            results_path=results_path,
            model_name=model_name,
            plot_filename=plot_filename,
            plot_title=plot_title,
            info_filename=info_filename,
            binary_labels=binary_labels,
            multilabel=multilabel,
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
    save_model: bool = True,
    make_plots: bool = True,
):

    if seed is None:
        seed = random.randint(1, 1 << 30)
    seed_everything(seed)

    model_hash = "40e65407aa"

    hidden_layer_size = 1024
    number_of_layers = 5
    hidden_layers = [hidden_layer_size] * number_of_layers
    base_encoder_size = -1

    model_config: MinModelConfig = {
        "num_layers": 3,
        "input_dim": None,
        "hidden_dim": 128,
        "hidden_layers": hidden_layers,
        "output_dim": base_encoder_size,
        "use_batch_norm": True,
    }

    use_encoders = False
    freeze_encoders = True

    finetuning_layers = 3
    embedding_input = base_encoder_size + 16 + 16

    base_short_name = f"{hidden_layer_size}x{number_of_layers}+{base_encoder_size}"

    encoder_base_path = os.path.join(
        "results",
        "v4",
        "crossfold_raw",
        model_hash,
        "encoders",
        "{encoder_class}",
        "{miner_setting}",
        "{loss_setting}",
        "models",
        "{encoder_name}",
    )
    encoders_settings: EncoderModelConfigs = {
        "encoders": [
            {
                "encoder_path": encoder_base_path.format(
                    encoder_class="encoder_lower_v2",
                    miner_setting="triplet_all",
                    loss_setting="triplet",
                    encoder_name="NoFilter()-MulticlassQuantifierLimitLabeler(lower_1-5)-CV-1L512-O16-512b-0.001lr_cf{}.pt",
                ),
                "short_name": "lower512-1x16",
                "freeze_encoder": freeze_encoders,
                "remove_last_layer": False,
                "replace_last_layer_with": None,
                "model_config": {
                    "num_layers": 2,
                    "input_dim": 346,
                    "hidden_dim": -1,
                    "output_dim": 16,
                    "hidden_layers": [512],
                    "use_batch_norm": True,
                },
            },
            {
                "encoder_path": encoder_base_path.format(
                    encoder_class="encoder_upper_v2",
                    miner_setting="triplet_all",
                    loss_setting="triplet",
                    encoder_name="NoFilter()-MulticlassQuantifierLimitLabeler(upper_1-5)-CV-1L512-O16-512b-0.001lr_cf{}.pt",
                ),
                "short_name": "upper512-1x16",
                "freeze_encoder": freeze_encoders,
                "remove_last_layer": False,
                "replace_last_layer_with": None,
                "model_config": {
                    "num_layers": 2,
                    "input_dim": 346,
                    "hidden_dim": -1,
                    "output_dim": 16,
                    "hidden_layers": [512],
                    "use_batch_norm": True,
                },
            },
        ],
        "finetuning": {
            "num_layers": finetuning_layers,
            "input_dim": embedding_input,
            "hidden_dim": embedding_input,
            "output_dim": None,
            "hidden_layers": None,
            "use_batch_norm": True,
        },
    }

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
    # --- binary
    # label_logic = BinaryAtomicLabeler(atomic="RED", hop=1)
    # label_logic = BinaryHopLabeler(hop=1)
    # label_logic = BinaryRestrictionLabeler(lower=4, upper=-1)
    # label_logic = BinaryORHopLabeler(hop=0)
    # label_logic = BinaryDuplicatedAtomicLabeler()
    # --- multiclass
    # label_logic = MulticlassRestrictionLabeler(
    #     [
    #         (1, None),
    #         (2, None),
    #         (3, None),
    #         (4, None),
    #         (5, None),
    #         (None, 1),
    #         (None, 2),
    #         (None, 3),
    #         (None, 4),
    #         (None, 5),
    #     ],
    #     custom_name="lower-upper-open",
    # )
    # label_logic = MulticlassOpenQuantifierLabeler()
    # --- multilabel
    # label_logic = MultiLabelAtomicLabeler()
    label_logic = MultiLabelAtomicPositionLabeler()
    # label_logic = MultilabelQuantifierLabeler()
    # label_logic = MultilabelRestrictionLabeler(mode="both", class_for_no_label=False)
    # label_logic = MultilabelRestrictionLabeler(mode="upper", class_for_no_label=True)
    # label_logic = MultilabelFormulaElementLabeler()
    # label_logic = MultilabelFormulaElementWithAtomicPositionLabeler()
    labeler = LabelerApply(labeler=label_logic)
    # * /labelers
    data_config: NetworkDataConfig = {
        "root": "data/gnns_v4",
        "model_hash": model_hash,
        "selector": selector,
        "labeler": labeler,
        "formula_mapping": FormulaMapping("./data/formulas.json"),
        "test_selector": test_selector,
        "load_aggregated": "aggregated_raw.pt",
        "force_preaggregated": True,
    }
    crossfold_config: CrossFoldConfiguration = {
        "n_splits": 5,
        "shuffle": True,
        "random_state": seed,
        "defer_loading": True,
        "required_train_hashes": [],
        "use_stratified": None,
    }
    crossfold_fold_file = os.path.join(
        "results", "v4", "crossfold_raw", model_hash, "base.folds"
    )

    early_stopping: StopFormat = {
        "operation": "early_decrease",
        "conditions": {"test_loss": 0.001},
        "stay": 5,
    }

    only_run_for_first_cv = True

    iterations = 50
    test_batch = 2048

    if name is None:
        test_selector_name = "CV" if crossfold_config else str(test_selector)
        name = f"{selector}-{labeler}-{test_selector_name}"

    if use_encoders:
        encoder_names = ",".join(
            [settings["short_name"] for settings in encoders_settings["encoders"]]
        )
        if model_config is not None:
            encoder_names = f"{base_short_name},{encoder_names}"
        msg = f"{name}-F({freeze_encoders})-ENC[{encoder_names}]-FINE[{finetuning_layers}]-{train_batch}b-{lr}lr"
    else:
        msg = f"{name}-ENC[{base_short_name}]-{train_batch}b-{lr}lr"

    results_path = os.path.join(
        "results",
        "v4",
        "crossfold_raw",
        model_hash,
        "classification+color_encoder",
        "only_colors",
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
        crossfold_config=crossfold_config,
        crossfold_fold_file=crossfold_fold_file,
        only_run_for_first_cv=only_run_for_first_cv,
        iterations=iterations,
        gpu_num=0,
        seed=seed,
        test_size=0.20,
        stratify=True,
        data_workers=0,
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
        binary_labels=isinstance(label_logic, BinaryCategoricalLabeler),
        multilabel=isinstance(label_logic, MultiLabelCategoricalLabeler),
        use_encoders=use_encoders,
        encoders_configs=encoders_settings,
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

    main(
        seed=0,
        train_batch=32,
        lr=1e-3,
        save_model=True,
        make_plots=True,
    )
