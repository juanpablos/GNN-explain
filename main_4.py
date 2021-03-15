import logging
import os
import random
from timeit import default_timer as timer
from typing import Any, Dict

import torch

from src.data.formula_index import FormulaMapping
from src.data.formulas import *
from src.data.loader import text_sequence_loader
from src.data.utils import train_test_dataset
from src.eval_utils import evaluate_text_model
from src.run_logic import run, seed_everything
from src.training.gnn_sequence import GraphSequenceTrainer
from src.typing import LSTMConfig, MinModelConfig, NetworkDataConfig
from src.utils import prepare_info_dir, write_result_info_text, write_train_data
from src.visualization.curve_plot import plot_training

logger = logging.getLogger("src")
logger_metrics = logging.getLogger("metrics")


def get_model_name(encoder_output, lstm_config):
    embedding = f"emb{lstm_config['embedding_dim']}"
    name = lstm_config["name"]
    lstm_input = f"IN{encoder_output}"
    lstm_hidden = f"lstmH{lstm_config['hidden_dim']}"
    use_init = f"lstmH{lstm_config['hidden_dim']}"
    dropout = f"drop{lstm_config['dropout_prob']}"
    concat_input = f"cat{lstm_config['concat_encoder_input']}"

    lstm = f"{embedding}-{name}{lstm_input}-{lstm_hidden}-{use_init}-{concat_input}-{dropout}"

    if name == "lstmcell":
        compose_state = f"comp{lstm_config['compose_encoder_state']}"
        compose_dim = f"d{lstm_config['compose_dim']}"

        lstm = f"{lstm}-{compose_state}{compose_dim}"

    return lstm


def run_experiment(
    encoder_config: MinModelConfig,
    decoder_config: LSTMConfig,
    data_config: NetworkDataConfig,
    graph_config: Dict[str, Any],
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
    (
        datasets,
        vocabulary,
        hash_formula,
        hash_label,
        data_reconstruction,
        formula_target,
    ) = text_sequence_loader(
        **data_config,
        graph_config=graph_config,
        _legacy_load_without_batch=_legacy_load_without_batch,
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
            stratify=False,
        )

    vocab_size = len(vocabulary)
    logger.debug(f"vocab size of {vocab_size} detected")

    logger.debug(
        f"Semantic Evaluation information: {formula_target.graph_statistics()['total']} positives"
    )

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

    trainer = GraphSequenceTrainer(
        seed=seed,
        subset_size=0.2,
        logging_variables="all",
        vocabulary=vocabulary,
        target_apply_mapping=formula_target,
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

    encoder, decoder = run(
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
        encoder.cpu()
        decoder.cpu()
        os.makedirs(f"{results_path}/models/", exist_ok=True)
        obj = {
            "encoder": encoder.state_dict(),
            "decoder": decoder.state_dict(),
            "vocabulary": vocabulary,
        }
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


def main(
    name: str = None,
    seed: int = None,
    train_batch: int = 32,
    lr: float = 0.001,
    save_model: bool = True,
    write_train_data: bool = True,
    make_plots: bool = True,
):

    if seed is None:
        seed = random.randint(1, 1 << 30)
    seed_everything(seed)

    hidden = 128
    n_layers = 2
    encoder_output = 1024
    encoder_config: MinModelConfig = {
        "num_layers": n_layers,
        "input_dim": None,
        "hidden_dim": hidden,
        "hidden_layers": None,
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

    model_hash = "f4034364ea-batch"

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
    # test_selector = FilterApply(condition="or")
    # test_selector.add(AtomicOnlyFilter(atomic="all"))
    # test_selector.add(RestrictionFilter(lower=4, upper=None))
    test_selector = SelectFilter(
        hashes=[
            "4805042859",
            "aae49a2efc",
            "ac4932d9e6",
            "2baa2ed86c",
            "4056021fb9",
            "548c9f191e",
            "c37cb98a75",
            "b628ede2fc",
            "f38520e138",
            "65597e2291",
            "5e65a2eaac",
            "838d8aecad",
        ]
    )
    # test_selector = NullFilter()
    # * /test_filters

    # * labelers
    label_logic = TextSequenceLabeler()
    labeler = SequenceLabelerApply(labeler=label_logic)

    undirected = False

    # * /labelers
    data_config: NetworkDataConfig = {
        "root": "data/gnns",
        "model_hash": model_hash,
        "selector": selector,
        "labeler": labeler,
        "formula_mapping": FormulaMapping("./data/formulas.json"),
        "test_selector": test_selector,
        "load_aggregated": f"graph_gnns{'_undirected' if undirected else ''}.pt",
        "force_preaggregated": True,
    }

    iterations = 20
    test_batch = 1024

    if name is None:
        name = f"{selector}-{labeler}-{test_selector}"

    encoder = f"UND{undirected}-GraphNN-h{hidden}-L{n_layers}"
    decoder = get_model_name(encoder_output, lstm_config)

    msg = f"{name}-{encoder}-{decoder}-{train_batch}b-{lr}lr"

    results_path = f"./results/testing/{model_hash}"

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
        encoder_config=encoder_config,
        decoder_config=lstm_config,
        data_config=data_config,
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

    main(seed=0, train_batch=512, lr=0.001, save_model=True, make_plots=True)
