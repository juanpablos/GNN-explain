import csv
import json
import logging
import os
from timeit import default_timer as timer
from typing import List, Optional, Union

import torch
from nltk.translate.bleu_score import corpus_bleu

from src.data.auxiliary import PreloadedSingleFormulaEvaluationWrapper
from src.data.datasets import NetworkDataset
from src.data.formula_index import FormulaMapping
from src.data.formulas import *
from src.data.loader import text_sequence_loader
from src.data.vocabulary import Vocabulary
from src.eval_heuristics import *
from src.generate_graphs import graph_data_stream_pregenerated_graphs_test
from src.graphs.foc import *
from src.models.encoder_model_helper import EncoderModelHelper
from src.training.check_formulas import FormulaReconstruction
from src.training.metrics import semantic_evaluation_for_formula
from src.training.sequence_training import RecurrentTrainer
from src.typing import LSTMConfig, MinModelConfig, NetworkDataConfig

logger = logging.getLogger("src")


def _write_predicted_formulas(
    cv_fold: int,
    generated_formulas: List[Union[FOC, None]],
    decoded_expected_formula: Optional[FOC],
    cv_evaluation_results_path: str,
    expected_formula_hash: str,
    bleu_score: float,
):
    with open(
        os.path.join(cv_evaluation_results_path, expected_formula_hash + ".txt"),
        "w",
        newline="",
    ) as evaluation_file:
        expected_header = "\n".join(
            [
                "-" * 20,
                repr(decoded_expected_formula),
                f"bleu score: {bleu_score}",
                "-" * 20,
            ]
        )
        evaluation_file.writelines([expected_header, "\n\n"])

        csv_writer = csv.writer(evaluation_file, delimiter=";")
        csv_writer.writerow(["formula"])

        logger.debug(f"[CV{cv_fold}] Running evaluation")
        for generated_formula in generated_formulas:
            csv_writer.writerow([repr(generated_formula)])


def _write_evaluate_for_formulas(
    cv_fold: int,
    generated_formulas: List[Union[FOC, None]],
    decoded_expected_formula: Optional[FOC],
    cv_evaluation_results_path: str,
    expected_formula_hash: str,
    bleu_score: float,
):
    logger.debug(f"[CV{cv_fold}] Loading evaluation graphs")
    test_stream = graph_data_stream_pregenerated_graphs_test(
        formula=decoded_expected_formula,
        graphs_path=os.path.join("data", "graphs"),
        graphs_filename="test_graphs_v2_10626.pt",
        pregenerated_labels_file=f"{expected_formula_hash}_labels_test.pt",
    )

    semantic_formula_evaluator = PreloadedSingleFormulaEvaluationWrapper(
        formula=decoded_expected_formula, graphs_data=test_stream
    )
    cached_semantic_results = {}

    with open(
        os.path.join(cv_evaluation_results_path, expected_formula_hash + ".txt"),
        "w",
        newline="",
    ) as evaluation_file:
        expected_header = "\n".join(
            [
                "-" * 20,
                repr(decoded_expected_formula),
                f"bleu score: {bleu_score}",
                "-" * 20,
            ]
        )
        evaluation_file.writelines([expected_header, "\n\n"])

        csv_writer = csv.writer(evaluation_file, delimiter=";")
        csv_writer.writerow(["formula", "precision", "recall", "accuracy"])

        logger.debug(f"[CV{cv_fold}] Running evaluation")
        for generated_formula in generated_formulas:
            precision, recall, accuracy = semantic_evaluation_for_formula(
                formula=generated_formula,
                semantic_formula=semantic_formula_evaluator,
                cached_evaluation=cached_semantic_results,
            )

            csv_writer.writerow(
                [
                    repr(generated_formula),
                    precision,
                    recall,
                    accuracy,
                ]
            )


def evaluate_crossfolds(
    encoder_config: Optional[MinModelConfig],
    decoder_config: LSTMConfig,
    data_config: NetworkDataConfig,
    labeler: TextSequenceLabeler,
    model_file_path: str,
    model_filename: str,
    encoder_configs_path: Optional[str],
    encoder_configs_filename: Optional[str],
    encoder_target_cf: Optional[int],
    labeler_path: str,
    labeler_filename: str,
    evaluation_results_path: str,
    skip_semantic_evaluation: bool,
    run_only_bleu: bool = False,
):
    logger.info("Loading labeler stored state")
    with open(os.path.join(labeler_path, labeler_filename)) as f:
        labeler_data = json.load(f)

    logger.info("Loading Files")
    # vocabulary: object with toked_id-token mappings
    # hash_formula: formula_hash -> formula_object
    # hash_label:
    #   single label: formula_hash -> label_id
    #   multilabel: formula_hash -> List[label_id]
    # data_reconstruction: point_index -> formula_object
    # serialized_labeler: arbitrary dict of a serialized labeler classes and internals
    (test_datasets, vocabulary, *_,) = text_sequence_loader(
        **data_config,
        labeler_stored_state=labeler_data,
        return_list_of_datasets=True,
        graph_config={"configs": [], "_ignore": True},
        _legacy_load_without_batch=True,
    )
    labeler.preload_vocabulary(vocabulary.token2id)

    assert isinstance(test_datasets, list), "Datasets are not a list"
    assert all(
        isinstance(dataset, NetworkDataset) for dataset in test_datasets
    ), "Not all datasets are network datasets"

    input_shape = test_datasets[0][0][0].shape
    assert len(input_shape) == 1, "The input dimension is different from 1"

    vocab_size = len(vocabulary)
    logger.debug(f"vocab size of {vocab_size} detected")

    logger.info(f"Total Dataset size: {sum(len(dataset) for dataset in test_datasets)}")

    trainer = RecurrentTrainer(
        seed=None,
        subset_size=1,
        logging_variables="all",
        vocabulary=vocabulary,
        target_apply_mapping=None,
    )

    decoder_config["vocab_size"] = vocab_size

    if encoder_configs_filename is None:
        assert encoder_config is not None
        encoder_config["input_dim"] = input_shape[0]

        logger.info("Using base encoder")
        trainer.init_encoder(use_encoder=False, **encoder_config)
    else:
        assert encoder_configs_path is not None
        assert encoder_configs_filename is not None
        logger.info("Using complex encoder")

        input_size = input_shape[0]

        with open(
            os.path.join(
                encoder_configs_path, encoder_configs_filename.format(encoder_target_cf)
            )
        ) as f:
            encoder_helper_configs = json.load(f)
        encoder_helper = EncoderModelHelper.load_helper(configs=encoder_helper_configs)
        trainer.init_encoder(
            use_encoder=True,
            model_helper=encoder_helper,
            model_input_size=input_size,
        )

    trainer.init_decoder(**decoder_config)

    device = torch.device("cuda:0")
    trainer.set_device(device=device)
    trainer.init_trainer(inference=True)

    for i in range(1, 6):
        if encoder_target_cf is not None and encoder_target_cf != i:
            continue

        logger.info(f"Running eval for crossfold {i}/{5}")

        cf_model_name = model_filename.format(i)

        model_info = torch.load(os.path.join(model_file_path, cf_model_name))
        encoder_weights = model_info["encoder"]
        decoder_weights = model_info["decoder"]
        vocabulary = Vocabulary()
        vocabulary.load_vocab(model_info["vocabulary"])

        trainer.encoder.load_state_dict(encoder_weights)
        trainer.decoder.load_state_dict(decoder_weights)

        cv_evaluation_results_path = os.path.join(evaluation_results_path, f"CV{i}")
        os.makedirs(cv_evaluation_results_path, exist_ok=True)

        # references
        # list of references
        all_expected_formula_tokens: List[List[List[int]]] = []
        # hypotheses
        # list of hypotheses
        all_predicted_formula_tokens: List[List[int]] = []

        for test_dataset in test_datasets:
            expected_formula_hash = test_dataset.formula_hash
            expected_formula = test_dataset.formula
            encoded_formula = labeler(expected_formula)

            if isinstance(test_dataset.dataset, torch.Tensor):
                dataset = test_dataset.dataset
            else:
                dataset = torch.stack(test_dataset.dataset)

            formula_dataset = dataset.to(device)

            logger.info(f"[CV{i}] Calculating inference for {expected_formula}")
            predictions, _ = trainer.inference(formula_dataset)

            logger.info(f"[CV{i}] Reconstructing formulas from gnns")
            # reconstruct formulas with the scores
            formula_reconstruction = FormulaReconstruction(vocabulary)

            encoded_formula_cleaned_tokens = encoded_formula[1:]
            (decoded_expected_formula,), _ = formula_reconstruction.batch2expression(
                batch_data=[encoded_formula_cleaned_tokens]
            )
            predicted_formula_tokens = predictions.tolist()
            generated_formulas, _ = formula_reconstruction.batch2expression(
                batch_data=predicted_formula_tokens
            )
            predicted_formula_cleaned_tokens = formula_reconstruction.batch2clean(
                batch_data=predicted_formula_tokens
            )

            n_formulas = len(predicted_formula_cleaned_tokens)
            all_expected_formula_tokens.extend(
                [[encoded_formula_cleaned_tokens]] * n_formulas
            )
            all_predicted_formula_tokens.extend(predicted_formula_cleaned_tokens)

            formula_bleu_score = corpus_bleu(
                [[encoded_formula_cleaned_tokens]] * n_formulas,
                predicted_formula_cleaned_tokens,
            )

            if not run_only_bleu:
                if skip_semantic_evaluation:
                    _write_predicted_formulas(
                        cv_fold=i,
                        generated_formulas=generated_formulas,
                        decoded_expected_formula=decoded_expected_formula,
                        cv_evaluation_results_path=cv_evaluation_results_path,
                        expected_formula_hash=expected_formula_hash,
                        bleu_score=formula_bleu_score,
                    )
                else:
                    _write_evaluate_for_formulas(
                        cv_fold=i,
                        generated_formulas=generated_formulas,
                        decoded_expected_formula=decoded_expected_formula,
                        cv_evaluation_results_path=cv_evaluation_results_path,
                        expected_formula_hash=expected_formula_hash,
                        bleu_score=formula_bleu_score,
                    )

            logger.info(f"[CV{i}] Calculating total bleu")

        total_bleu_score = corpus_bleu(
            all_expected_formula_tokens, all_predicted_formula_tokens
        )
        with open(
            os.path.join(cv_evaluation_results_path, ".bleu.txt"),
            "w",
            newline="",
        ) as bleu_file:
            json.dump({"bleu4": total_bleu_score}, bleu_file)


def evaluate_crossfolds_heuristics(
    data_config: NetworkDataConfig,
    heuristic: EvalHeuristic,
    evaluation_results_path: str,
    cf_target: Optional[int],
):
    logger.info("Loading Files")
    # hash_formula: formula_hash -> formula_object
    (test_datasets, _, hash_formula, *_) = text_sequence_loader(
        **data_config,
        return_list_of_datasets=True,
        graph_config={"configs": [], "_ignore": True},
        _legacy_load_without_batch=True,
    )

    assert isinstance(test_datasets, list), "Datasets are not a list"
    assert all(
        isinstance(dataset, NetworkDataset) for dataset in test_datasets
    ), "Not all datasets are network datasets"

    logger.info(f"Total Dataset size: {sum(len(dataset) for dataset in test_datasets)}")

    for i in range(1, 6):
        if cf_target is not None and cf_target != i:
            continue
        logger.info(f"Running eval for crossfold {i}/5")

        cv_evaluation_heuristic_results_path = os.path.join(
            evaluation_results_path, f"CV{i}"
        )
        os.makedirs(cv_evaluation_heuristic_results_path, exist_ok=True)

        for test_dataset in test_datasets:
            expected_formula_hash = test_dataset.formula_hash
            # expected_formula = test_dataset.formula

            expected_formula = FOC(hash_formula[expected_formula_hash])

            logger.debug(f"[CV{i}] Loading evaluation graphs")
            test_stream = graph_data_stream_pregenerated_graphs_test(
                formula=expected_formula,
                graphs_path=os.path.join("data", "graphs"),
                graphs_filename="test_graphs_v2_10626.pt",
                pregenerated_labels_file=f"{expected_formula_hash}_labels_test.pt",
            )

            semantic_formula_evaluator = PreloadedSingleFormulaEvaluationWrapper(
                formula=expected_formula, graphs_data=test_stream
            )
            cached_semantic_results = {}

            heuristic_allowed_values = heuristic.extract_valid_elements(
                formula=expected_formula
            )

            with open(
                os.path.join(
                    cv_evaluation_heuristic_results_path,
                    expected_formula_hash + ".txt",
                ),
                "w",
                newline="",
            ) as evaluation_file:
                expected_header = "\n".join(
                    ["-" * 20, repr(expected_formula), "-" * 20]
                )
                evaluation_file.writelines([expected_header, "\n\n"])

                csv_writer = csv.writer(evaluation_file, delimiter=";")
                csv_writer.writerow(
                    ["formula", "precision", "recall", "accuracy", "match"]
                )

                logger.debug(f"[CV{i}] Running evaluation")
                for gnn in test_dataset.dataset:
                    generated_formula = heuristic.predict(weights=gnn)
                    precision, recall, accuracy = semantic_evaluation_for_formula(
                        formula=generated_formula,
                        semantic_formula=semantic_formula_evaluator,
                        cached_evaluation=cached_semantic_results,
                    )

                    heuristic_matches = heuristic.match(
                        candicate_value=generated_formula,
                        allowed_values=heuristic_allowed_values,
                    )

                    csv_writer.writerow(
                        [
                            repr(generated_formula),
                            precision,
                            recall,
                            accuracy,
                            int(heuristic_matches),
                        ]
                    )


def main():
    model_hash = "40e65407aa"
    selector = NoFilter()
    test_selector = NullFilter()
    label_logic = TextSequenceLabeler()
    labeler = SequenceLabelerApply(labeler=label_logic)

    skip_semantic_evaluation = False

    hidden_layer_size = 1024
    number_of_layers = 5
    mlp_hidden_layers = [hidden_layer_size] * number_of_layers

    encoder_output_size = 8
    lstm_hidden = 8

    encoder_config = None
    # encoder_config: MinModelConfig = {
    #     "num_layers": 3,
    #     "input_dim": None,
    #     "hidden_dim": 128,
    #     "hidden_layers": mlp_hidden_layers,
    #     "output_dim": encoder_output_size,
    #     "use_batch_norm": True,
    # }
    decoder_config: LSTMConfig = {
        "name": "lstmcell",
        "encoder_dim": encoder_output_size,
        "embedding_dim": 4,
        "hidden_dim": lstm_hidden,
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

    data_config: NetworkDataConfig = {
        "root": "data/gnns_v4_test",
        "model_hash": model_hash,
        "selector": selector,
        "labeler": labeler,
        "formula_mapping": FormulaMapping("./data/formulas.json"),
        "test_selector": test_selector,
        "load_aggregated": "aggregated_raw.pt",
        "force_preaggregated": True,
    }

    base_folder = "text+encoder_v2+color(rem,rep)"
    model_filename = "NoFilter()-TextSequenceAtomic()-CV-F(True)-ENC[color1024x4+16,lower512x1+16,upper512x1+16]-FINE[2]-emb4-lstmcellIN8-lstmH8-initTrue-catTrue-drop0-compFalse-d256-32b-0.001lr"

    model_path = os.path.join(
        "results",
        "v4",
        "crossfold_raw",
        model_hash,
        base_folder,
        "models",
    )
    base_model_filename = f"{model_filename}_cf{{}}.pt"

    encoder_configs_path = os.path.join(
        "results", "v4", "crossfold_raw", model_hash, base_folder, "enc_conf"
    )
    # encoder_configs_filename = None
    encoder_configs_filename = f"{model_filename}_cf{{}}.conf.json"

    labeler_path = os.path.join(
        "results",
        "v4",
        "crossfold_raw",
        model_hash,
        base_folder,
        "labelers",
    )
    labeler_filename = f"{model_filename}.labeler"

    evaluation_results_path = os.path.join(
        "results",
        "v4",
        "crossfold_raw",
        model_hash,
        base_folder,
        "evaluation - delete",
        model_filename,
    )
    os.makedirs(evaluation_results_path, exist_ok=True)

    start = timer()
    evaluate_crossfolds(
        encoder_config=encoder_config,
        decoder_config=decoder_config,
        data_config=data_config,
        labeler=label_logic,
        model_file_path=model_path,
        model_filename=base_model_filename,
        encoder_configs_path=encoder_configs_path,
        encoder_configs_filename=encoder_configs_filename,
        encoder_target_cf=1,
        labeler_path=labeler_path,
        labeler_filename=labeler_filename,
        evaluation_results_path=evaluation_results_path,
        skip_semantic_evaluation=skip_semantic_evaluation,
        run_only_bleu=True,
    )

    end = timer()
    logger.info(f"Took {end-start} seconds")


def main_heuristic():
    model_hash = "40e65407aa"
    selector = NoFilter()
    test_selector = NullFilter()
    label_logic = TextSequenceLabeler()
    labeler = SequenceLabelerApply(labeler=label_logic)

    # heuristic = SingleFormulaHeuristic(formula=Property("RED"))
    heuristic = MinSumFormulaHeuristic()
    # heuristic = MaxSumFormulaHeuristic()
    # heuristic = MaxDiffSumFormulaHeuristic()

    data_config: NetworkDataConfig = {
        "root": "data/gnns_v4_test",
        "model_hash": model_hash,
        "selector": selector,
        "labeler": labeler,
        "formula_mapping": FormulaMapping("./data/formulas.json"),
        "test_selector": test_selector,
        "load_aggregated": "aggregated_raw.pt",
        "force_preaggregated": True,
    }

    evaluation_results_path = os.path.join(
        "results",
        "v4",
        "crossfold_raw",
        model_hash,
        "heuristics",
        str(heuristic),
        "test",
    )
    os.makedirs(evaluation_results_path, exist_ok=True)

    start = timer()
    evaluate_crossfolds_heuristics(
        data_config=data_config,
        heuristic=heuristic,
        evaluation_results_path=evaluation_results_path,
        cf_target=1,
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

    logger.addHandler(ch)

    # main()
    main_heuristic()
