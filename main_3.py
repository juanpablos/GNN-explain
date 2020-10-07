import logging
import os
import random
from timeit import default_timer as timer
from typing import List

import torch

from src.data.formula_index import FormulaMapping
from src.data.formulas import *
from src.data.loader import text_sequence_loader
from src.data.utils import get_input_dim, train_test_dataset
from src.eval_utils import evaluate_text_model
from src.run_logic import run, seed_everything
from src.training.sequence_training import RecurrentTrainer
from src.typing import LSTMConfig, MinModelConfig, NetworkDataConfig
from src.utils import write_result_info_text
from src.visualization.curve_plot import plot_training

logger = logging.getLogger("src")


def run_experiment(
        encoder_config: MinModelConfig,
        decoder_config: LSTMConfig,
        data_config: NetworkDataConfig,
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
        info_filename: str = "info",
        _legacy_load_without_batch: bool = False
):

    logger.info("Loading Files")
    # vocabulary: object with toked_id-token mappings
    # hash_formula: formula_hash -> formula_object
    # hash_label:
    #   single label: formula_hash -> label_id
    #   multilabel: formula_hash -> List[label_id]
    # data_reconstruction: point_index -> formula_object
    (datasets, vocabulary,
     hash_formula, hash_label,
     data_reconstruction) = text_sequence_loader(
        **data_config,
        _legacy_load_without_batch=_legacy_load_without_batch)

    if isinstance(datasets, tuple):
        logger.debug("Using selected data as test")
        # * only here because return type problems when **[TypedDict]
        train_data, test_data = datasets
    else:
        logger.debug("Splitting data")
        train_data, test_data = train_test_dataset(dataset=datasets,
                                                   test_size=test_size,
                                                   random_state=seed,
                                                   shuffle=True,
                                                   stratify=False)

    vocab_size = len(vocabulary)
    logger.debug(f"vocab size of {vocab_size} detected")

    input_shape = get_input_dim(train_data)
    assert len(input_shape) == 1, "The input dimension is different from 1"

    encoder_config["input_dim"] = input_shape[0]
    decoder_config["vocab_size"] = vocab_size

    trainer = RecurrentTrainer(logging_variables="all",
                               vocabulary=vocabulary)

    trainer.init_dataloader(
        train_data,
        mode="train",
        batch_size=batch_size,
        pin_memory=False,
        shuffle=True,
        num_workers=data_workers)
    trainer.init_dataloader(
        test_data,
        mode="test",
        batch_size=test_batch_size,
        pin_memory=False,
        shuffle=True,
        num_workers=data_workers)

    trainer.init_encoder(**encoder_config)
    trainer.init_decoder(**decoder_config)

    logger.debug("Running")
    logger.debug(f"Input size is {input_shape[0]}")
    encoder, decoder = run(
        trainer=trainer,
        iterations=iterations,
        gpu_num=gpu_num,
        lr=lr,
        run_train_test=run_train_test)

    formula_metrics = evaluate_text_model(
        trainer=trainer,
        test_data=test_data,
        reconstruction=data_reconstruction
    )

    # returns a number to put after the file name in case it already exists
    # "" or " (N)"
    ext = write_result_info_text(
        path=results_path,
        filename=info_filename,
        formula_metrics=formula_metrics)

    if model_name is not None:
        logger.debug("Writing model")
        encoder.cpu()
        decoder.cpu()
        os.makedirs(f"{results_path}/models/", exist_ok=True)
        obj = {
            "encoder": encoder.state_dict(),
            "decoder": decoder.state_dict(),
            "vocabulary": vocabulary
        }
        torch.save(obj, f"{results_path}/models/{model_name}{ext}.pt")

    if plot_filename is not None:
        metrics = trainer.metric_logger
        plot_training(
            metric_history=metrics,
            save_path=results_path,
            filename=plot_filename + ext,
            title=plot_title,
            use_selected=False)


def main(
        name: str = None,
        seed: int = None,
        train_batch: int = 32,
        lr: float = 0.001,
        mlp_hidden_layers: List[int] = None,
        save_model: bool = True,
        make_plots: bool = True):

    if seed is None:
        seed = random.randint(1, 1 << 30)
    seed_everything(seed)

    mlp_hidden_layers = [128] if mlp_hidden_layers is None \
        else mlp_hidden_layers

    encoder_output = 1024
    mlp_config: MinModelConfig = {
        "num_layers": 3,
        "input_dim": None,
        "hidden_dim": 128,
        "hidden_layers": mlp_hidden_layers,
        "output_dim": encoder_output,
        "use_batch_norm": True
    }
    lstm_config: LSTMConfig = {
        "name": "lstmcell",
        "encoder_dim": encoder_output,
        "embedding_dim": 4,
        "hidden_dim": 256,
        "vocab_size": None,
        "context_hidden_init": True,
        "dropout_prob": 0
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
    test_selector = SelectFilter(hashes=[
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
        "838d8aecad"
    ])
    # test_selector = NullFilter()
    # * /test_filters

    # * labelers
    label_logic = TextSequenceLabeler()
    labeler = SequenceLabelerApply(labeler=label_logic)
    # * /labelers
    data_config: NetworkDataConfig = {
        "root": "data/gnns",
        "model_hash": model_hash,
        "selector": selector,
        "labeler": labeler,
        "formula_mapping": FormulaMapping("./data/formulas.json"),
        "test_selector": test_selector,
        "load_aggregated": "aggregated.pt"
    }

    iterations = 20
    test_batch = 1024

    if name is None:
        name = f"{selector}-{labeler}-{test_selector}"

    mlp = "+".join(
        [f"{l}L{val}" for l, val in enumerate(mlp_hidden_layers, start=1)])
    lstm = f"emb{lstm_config['embedding_dim']}-{lstm_config['name']}IN{encoder_output}-lstmH{lstm_config['hidden_dim']}-init{lstm_config['context_hidden_init']}-drop{lstm_config['dropout_prob']}"

    msg = f"{name}-{mlp}-{lstm}-{train_batch}b-{lr}lr"

    results_path = f"./results/testing/{model_hash}"
    plot_file = None
    if make_plots:
        plot_file = msg
    model_name = None
    if save_model:
        model_name = msg

    start = timer()
    run_experiment(
        encoder_config=mlp_config,
        decoder_config=lstm_config,
        data_config=data_config,
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
        info_filename=msg,
        _legacy_load_without_batch=True  # ! remove eventually
    )
    end = timer()
    logger.info(f"Took {end-start} seconds")


if __name__ == "__main__":
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

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

    __layers = [1024, 1024, 1024]
    main(
        seed=0,
        train_batch=512,
        lr=0.005,
        mlp_hidden_layers=__layers,
        save_model=True,
        make_plots=True
    )
