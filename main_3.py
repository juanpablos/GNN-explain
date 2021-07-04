import json
import logging
import os
import random
from timeit import default_timer as timer
from typing import Any, Dict, List, Union

import torch
from torch.functional import Tensor

from src.data.auxiliary import (
    FormulaAppliedDatasetWrapper,
    NetworkDatasetCollectionWrapper,
)
from src.data.datasets import LabeledDataset, LabeledSubset, TextSequenceDataset
from src.data.formula_index import FormulaMapping
from src.data.formulas import *
from src.data.gnn.utils import clean_state
from src.data.loader import text_sequence_loader
from src.data.sampler import TextNetworkDatasetCrossFoldSampler
from src.data.utils import get_input_dim, train_test_dataset
from src.data.vocabulary import Vocabulary
from src.eval_utils import evaluate_text_model
from src.run_logic import run, seed_everything
from src.training.check_formulas import FormulaReconstruction
from src.training.sequence_training import RecurrentTrainer
from src.typing import (
    CrossFoldConfiguration,
    LSTMConfig,
    MinModelConfig,
    NetworkDataConfig,
    S,
)
from src.utils import prepare_info_dir, write_result_info_text, write_train_data
from src.visualization.curve_plot import plot_training

logger = logging.getLogger("src")
logger_metrics = logging.getLogger("metrics")


def get_model_name(hidden_layers, lstm_config, encoder_output):
    encoder = "+".join([f"{l}L{val}" for l, val in enumerate(hidden_layers, start=1)])

    embedding = f"emb{lstm_config['embedding_dim']}"
    name = lstm_config["name"]
    lstm_input = f"IN{encoder_output}"
    lstm_hidden = f"lstmH{lstm_config['hidden_dim']}"
    use_init = f"init{lstm_config['init_state_context']}"
    dropout = f"drop{lstm_config['dropout_prob']}"
    concat_input = f"cat{lstm_config['concat_encoder_input']}"

    lstm = f"{embedding}-{name}{lstm_input}-{lstm_hidden}-{use_init}-{concat_input}-{dropout}"

    if name == "lstmcell":
        compose_state = f"comp{lstm_config['compose_encoder_state']}"
        compose_dim = f"d{lstm_config['compose_dim']}"

        lstm = f"{lstm}-{compose_state}-{compose_dim}"

    return encoder, lstm


def inference(
    results_path: str,
    model_name_to_load: str,
    encoder_config: MinModelConfig,
    decoder_config: LSTMConfig,
    gpu_num: int,
    inference_filename: str,
    inference_data_file: str,
    write_representation: bool = True,
):
    logger.info("Loading inference data")
    # load the data we want to evaluate in
    data = torch.load(inference_data_file)

    inference_data = []
    for network in data:
        weights = clean_state(network)
        concat_weights = torch.cat([w.flatten() for w in weights.values()])
        inference_data.append(concat_weights)
    inference_data = torch.stack(inference_data, dim=0)
    assert len(inference_data.shape) == 2

    logger.info("Loading Pre-trained meta model")
    model_weights = torch.load(f"{results_path}/models/{model_name_to_load}.pt")
    encoder_weights = model_weights["encoder"]
    decoder_weights = model_weights["decoder"]
    vocabulary = model_weights["vocabulary"]

    # load the model in the trainer
    trainer = RecurrentTrainer(
        seed=None,
        subset_size=0.2,
        logging_variables="all",
        vocabulary=vocabulary,
        target_apply_mapping=None,
    )

    encoder_config["input_dim"] = inference_data.shape[-1]
    decoder_config["vocab_size"] = len(vocabulary)

    logger.info("Initializing trainer and metamodel")
    trainer.init_encoder(**encoder_config, model_weights=encoder_weights)
    trainer.init_decoder(**decoder_config, model_weights=decoder_weights)

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{gpu_num}")
    else:
        device = torch.device("cpu")

    trainer.set_device(device=device)
    trainer.init_trainer(inference=True)

    logger.info("Calculating inference")
    # directly run the dataset on the inference
    predictions, _ = trainer.inference(inference_data)

    logger.info("Reconstructing formulas")
    # reconstruct formulas with the scores
    formula_reconstruction = FormulaReconstruction(vocabulary)

    # write the expected and predictions in a file sequentially
    generated_formulas, _ = formula_reconstruction.batch2expression(
        batch_data=predictions.tolist()
    )

    logger.info("Writing results")
    inference_path = f"{results_path}/inference/{model_name_to_load}"
    os.makedirs(inference_path, exist_ok=True)
    with open(f"{inference_path}/{inference_filename}", "w", encoding="utf-8") as out:
        for formula in generated_formulas:
            f = repr(formula) if write_representation else str(formula)
            out.write(f"{f}\n")


def _run_experiment(
    train_data: Union[
        TextSequenceDataset[Tensor],
        LabeledDataset[Tensor, Tensor],
        LabeledSubset[Tensor, Tensor],
    ],
    test_data: Union[
        TextSequenceDataset[Tensor],
        LabeledDataset[Tensor, Tensor],
        LabeledSubset[Tensor, Tensor],
    ],
    vocabulary: Vocabulary,
    encoder_config: MinModelConfig,
    decoder_config: LSTMConfig,
    data_reconstruction: NetworkDatasetCollectionWrapper,
    formula_target: FormulaAppliedDatasetWrapper,
    iterations: int = 100,
    seed: int = 10,
    gpu_num: int = 0,
    data_workers: int = 2,
    batch_size: int = 64,
    test_batch_size: int = 512,
    lr: float = 0.01,
    run_train_test: bool = False,
    results_path: str = "./results",
    model_name: str = None,
    plot_filename: str = None,
    plot_title: str = None,
    train_file: str = None,
    info_filename: str = "info",
):
    vocab_size = len(vocabulary)
    logger.debug(f"vocab size of {vocab_size} detected")

    logger.debug(
        f"Semantic Evaluation information: {formula_target.graph_statistics()['total']} positives"
    )

    input_shape = get_input_dim(train_data)
    assert len(input_shape) == 1, "The input dimension is different from 1"

    encoder_config["input_dim"] = input_shape[0]
    decoder_config["vocab_size"] = vocab_size

    # ready to log metrics
    # returns a number to put after the file name in case it already exists
    # "" or " (N)"
    ext_filename, ext = prepare_info_dir(path=results_path, filename=info_filename)

    # --- metrics logger
    fh = logging.FileHandler(
        os.path.join(results_path, "info", f"{info_filename}{ext}.log"), mode="w"
    )
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(asctime)s,%(message)s"))

    # prevents writing to different files at the same
    # time in case of being called multiple times
    for handlers in logger_metrics.handlers[:]:
        logger_metrics.removeHandler(handlers)
    logger_metrics.addHandler(fh)
    # /--- metrics logger

    if model_name is None:
        write_checkpoint = False
        checkpoint_path = ""
    else:
        write_checkpoint = True
        checkpoint_path = os.path.join(results_path, "models", model_name)
    os.makedirs(f"{results_path}/models/{model_name}", exist_ok=True)
    trainer = RecurrentTrainer(
        seed=seed,
        subset_size=0.05,
        logging_variables="all",
        vocabulary=vocabulary,
        target_apply_mapping=formula_target,
        write_checkpoints=write_checkpoint,
        checkpoints_path=checkpoint_path,
        checkpoints_name="model",
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

    trainer.init_encoder(**encoder_config)
    trainer.init_decoder(**decoder_config)

    logger.debug("Running")
    logger.debug(f"Input size is {input_shape[0]}")

    (encoder, decoder) = run(
        trainer=trainer,
        iterations=iterations,
        gpu_num=gpu_num,
        lr=lr,
        run_train_test=run_train_test,
    )

    formula_metrics = evaluate_text_model(
        trainer=trainer, test_data=test_data, reconstruction=data_reconstruction
    )

    write_result_info_text(
        path=results_path,
        filename=ext_filename,
        formula_metrics=formula_metrics,
        semantic_eval_data=formula_target.graph_statistics(),
    )

    if model_name is not None:
        logger.debug("Writing model")
        encoder = encoder.cpu()
        decoder = decoder.cpu()
        obj = {
            "encoder": encoder.state_dict(),
            "decoder": decoder.state_dict(),
            "vocabulary": vocabulary.token2id,
        }
        os.makedirs(f"{results_path}/models/", exist_ok=True)
        torch.save(obj, f"{results_path}/models/{model_name}{ext}.pt")

    metrics = trainer.metric_logger
    if train_file is not None:
        write_train_data(
            metric_history=metrics, save_path=results_path, filename=train_file + ext
        )

    if plot_filename is not None:
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
    encoder_config: MinModelConfig,
    decoder_config: LSTMConfig,
    data_config: NetworkDataConfig,
    graph_config: Dict[str, Any],
    crossfold_config: CrossFoldConfiguration = None,
    iterations: int = 100,
    gpu_num: int = 0,
    seed: int = 10,
    test_size: float = 0.25,
    data_workers: int = 2,
    batch_size: int = 64,
    test_batch_size: int = 512,
    lr: float = 0.01,
    run_train_test: bool = False,
    results_path: str = "./results",
    model_name: str = None,
    plot_filename: str = None,
    plot_title: str = None,
    train_file: str = None,
    info_filename: str = "info",
    _legacy_load_without_batch: bool = False,
):

    logger.info("Loading Files")
    # vocabulary: object with toked_id-token mappings
    # hash_formula: formula_hash -> formula_object
    # hash_label:
    #   single label: formula_hash -> label_id
    #   multilabel: formula_hash -> List[label_id]
    # data_reconstruction: point_index -> formula_object
    # serialized_labeler: arbitrary dict of a serialized labeler classes and internals
    (
        datasets,
        vocabulary,
        hash_formula,
        hash_label,
        data_reconstruction,
        formula_target,
        serialized_labeler,
    ) = text_sequence_loader(
        **data_config,
        graph_config=graph_config,
        cross_fold_configuration=crossfold_config,
        _legacy_load_without_batch=_legacy_load_without_batch,
    )

    if isinstance(datasets, TextNetworkDatasetCrossFoldSampler):
        logger.info(f"Total Dataset size: {datasets.dataset_size}")

        file_ext = ""
        n_splits = datasets.n_splits
        for i, (train_data, test_data, data_reconstruction) in enumerate(
            datasets, start=1
        ):
            logger.info(f"Running experiment for crossfold {i}/{n_splits}")

            cf_model_name = f"{model_name}_cf{i}"
            cf_plot_filename = f"{plot_filename}_cf{i}"
            cf_info_filename = f"{info_filename}_cf{i}"

            file_ext = _run_experiment(
                train_data=train_data,
                test_data=test_data,
                vocabulary=vocabulary,
                encoder_config=encoder_config,
                decoder_config=decoder_config,
                data_reconstruction=data_reconstruction,
                formula_target=formula_target,
                iterations=iterations,
                seed=seed,
                gpu_num=gpu_num,
                data_workers=data_workers,
                batch_size=batch_size,
                test_batch_size=test_batch_size,
                lr=lr,
                run_train_test=run_train_test,
                results_path=results_path,
                model_name=cf_model_name,
                plot_filename=cf_plot_filename,
                plot_title=plot_title,
                train_file=train_file,
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
                stratify=False,
            )

        logger.info(f"Total Dataset size: {len(train_data) + len(test_data)}")

        file_ext = _run_experiment(
            train_data=train_data,
            test_data=test_data,
            vocabulary=vocabulary,
            encoder_config=encoder_config,
            decoder_config=decoder_config,
            data_reconstruction=data_reconstruction,
            formula_target=formula_target,
            iterations=iterations,
            seed=seed,
            gpu_num=gpu_num,
            data_workers=data_workers,
            batch_size=batch_size,
            test_batch_size=test_batch_size,
            lr=lr,
            run_train_test=run_train_test,
            results_path=results_path,
            model_name=model_name,
            plot_filename=plot_filename,
            plot_title=plot_title,
            train_file=train_file,
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
    mlp_hidden_layers: List[int] = None,
    save_model: bool = True,
    write_train_data: bool = True,
    make_plots: bool = True,
):

    if seed is None:
        seed = random.randint(1, 1 << 30)
    seed_everything(seed)

    mlp_hidden_layers = [128] if mlp_hidden_layers is None else mlp_hidden_layers

    encoder_output = 1024
    mlp_config: MinModelConfig = {
        "num_layers": 3,
        "input_dim": None,
        "hidden_dim": 128,
        "hidden_layers": mlp_hidden_layers,
        "output_dim": encoder_output,
        "use_batch_norm": True,
    }
    lstm_config: LSTMConfig = {
        "name": "lstmcell",
        "encoder_dim": encoder_output,
        "embedding_dim": 4,
        "hidden_dim": 256,
        "vocab_size": None,
        "dropout_prob": 0,
        # * works best with
        "init_state_context": True,
        # * works best with
        # * when too little classes and/or data, works best without
        "concat_encoder_input": True,
        # * works best without
        "compose_encoder_state": False,
        "compose_dim": 256,
    }

    graph_config = {
        "n_properties": 4,
        "seed": seed,
        "configs": [
            {"min_nodes": 10, "max_nodes": 60, "n_graphs": 5, "m": 4},
            {"min_nodes": 50, "max_nodes": 100, "n_graphs": 5, "m": 5},
            {"min_nodes": 10, "max_nodes": 100, "n_graphs": 5, "m": 5},
        ],
    }

    model_hash = "40e65407aa"

    # * filters
    # selector = FilterApply(condition="or")
    # selector.add(AtomicFilter(atomic="all"))
    # selector.add(RestrictionFilter(lower=1, upper=2))
    # selector.add(RestrictionFilter(lower=None, upper=-1))
    # selector = SelectFilter(hashes=[
    #     "0c957889eb",
    #     "1c998884a4",
    #     "4056021fb9"
    # ])
    selector = NoFilter()
    # * /filters

    # * test_filters
    # ! ignored if cross validation is used
    # test_selector = FilterApply(condition="or")
    # test_selector.add(AtomicOnlyFilter(atomic="all"))
    # test_selector.add(RestrictionFilter(lower=4, upper=None))
    # test_selector = SelectFilter(
    #     hashes=[
    #         "22609b6219",
    #         "d376f80fe0",
    #         "4865ca5688",
    #         "b739521345",
    #         "98e4690a6c",
    #         "fd1ede286c",
    #         "56dc8827b8",
    #         "c1eec67813",
    #         "8500dc307e",
    #         "530867a9ca",
    #     ]
    # )
    test_selector = NullFilter()
    # * /test_filters

    # * labelers
    label_logic = TextSequenceLabeler()
    labeler = SequenceLabelerApply(labeler=label_logic)

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
        "defer_loading": False,
    }

    iterations = 50
    test_batch = 2048

    if name is None:
        test_selector_name = "CV" if crossfold_config else str(test_selector)
        name = f"{selector}-{labeler}-{test_selector_name}"

    encoder, decoder = get_model_name(
        hidden_layers=mlp_hidden_layers,
        lstm_config=lstm_config,
        encoder_output=encoder_output,
    )

    msg = f"{name}-{encoder}-{decoder}-{train_batch}b-{lr}lr"

    results_path = f"./results/v4/crossfold_raw/{model_hash}"

    plot_file = None
    if make_plots:
        plot_file = msg
    train_file = None
    if write_train_data:
        train_file = msg
    model_name = None
    if save_model:
        model_name = msg

    start = timer()
    run_experiment(
        encoder_config=mlp_config,
        decoder_config=lstm_config,
        data_config=data_config,
        crossfold_config=crossfold_config,
        graph_config=graph_config,
        iterations=iterations,
        gpu_num=0,
        seed=seed,
        test_size=0.20,
        data_workers=0,
        batch_size=train_batch,
        test_batch_size=test_batch,
        lr=lr,
        run_train_test=True,
        results_path=results_path,
        model_name=model_name,
        plot_filename=plot_file,
        plot_title=msg,  # ? maybe a better message
        train_file=train_file,
        info_filename=msg,
        _legacy_load_without_batch=True,  # ! remove eventually
    )
    end = timer()
    logger.info(f"Took {end-start} seconds")


def main_inference():
    encoder_output = 1024
    mlp_config: MinModelConfig = {
        "num_layers": 3,
        "input_dim": None,
        "hidden_dim": 128,
        "hidden_layers": [1024, 1024, 1024],
        "output_dim": encoder_output,
        "use_batch_norm": True,
    }
    lstm_config: LSTMConfig = {
        "name": "lstmcell",
        "encoder_dim": encoder_output,
        "embedding_dim": 4,
        "hidden_dim": 256,
        "vocab_size": None,
        "dropout_prob": 0,
        # * works best with
        "init_state_context": True,
        # * works best with
        # * when too little classes and/or data, works best without
        "concat_encoder_input": True,
        # * works best without
        "compose_encoder_state": False,
        "compose_dim": 256,
    }

    model_hash = "40e65407aa"
    results_path = f"./results/v3/testing/{model_hash}"

    model_name = "NoFilter()-TextSequenceAtomic()-NullFilter()-1L1024+2L1024+3L1024-emb4-lstmcellIN1024-lstmH256-initTrue-catTrue-drop0-compFalse-d256-512b-0.005lr/model_6"

    formula_hash = "8b21f3f718"
    # formula_hash = "svd_processed_cora_l4_svd_agglomerative_d500_gnn"

    # inference_data_file = f"./data/full_gnn/{model_hash}/{formula_hash}.pt"

    inference_data_file = (
        f"./data/manual/{model_hash}/acgnn-n20-{model_hash}-{formula_hash}.pt"
    )
    inference_filename = f"{formula_hash}.txt"

    inference(
        encoder_config=mlp_config,
        decoder_config=lstm_config,
        gpu_num=0,
        results_path=results_path,
        model_name_to_load=model_name,
        inference_data_file=inference_data_file,
        inference_filename=inference_filename,
        write_representation=True,
    )


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
    main(
        seed=0,
        train_batch=32,
        lr=5e-4,
        mlp_hidden_layers=__layers,
        save_model=True,
        make_plots=True,
    )

    # main_inference()
