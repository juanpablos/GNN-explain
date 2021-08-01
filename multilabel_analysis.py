import json
import logging
import os
from timeit import default_timer as timer

import torch
from tabulate import tabulate

from src.data.dataset_splitter import NetworkDatasetCrossFoldSplitter
from src.data.formula_index import FormulaMapping
from src.data.formulas import *
from src.data.loader import categorical_loader
from src.data.utils import get_input_dim
from src.training.mlp_training import MLPTrainer
from src.typing import CrossFoldConfiguration, MinModelConfig, NetworkDataConfig

logger = logging.getLogger("src")


def _get_multilabel_analysis(y_true, y_pred, class_mapping):
    per_class_mistakes = {}

    trues_per_label = y_true.sum(dim=0)

    torch.set_printoptions(precision=4)
    torch.set_printoptions(sci_mode=False)
    for label_id in class_mapping:
        # select when i should be A and i said B

        # indices where label_id should be positive
        true_pos_idx = y_true[:, label_id] == 1
        # indices where label_id was marked negative
        pred_neg_idx = y_pred[:, label_id] == 0

        # indices where should be positive but it was negative for label_id
        wrong_indices = true_pos_idx & pred_neg_idx
        true_id = y_true[wrong_indices]
        pred_id = y_pred[wrong_indices]

        if len(true_id) == 0:
            continue

        # for the other labels

        should_be_0 = true_id == 0
        said_1 = pred_id == 1

        said_1_and_wrong = said_1 & should_be_0

        # number of indices where it should be 1 but it was 0
        # but also it was 0 for eveything
        all_0 = torch.all(pred_id == 0, dim=1).sum().item()

        sum_every_1 = said_1.sum(dim=0)
        sum_every_1[sum_every_1 == 0] = 1
        # this shows the times it incorrectly labeled label i, the percentage
        # it incorrectly labeled 1 other labels
        # print(said_1_and_wrong.sum(dim=0), sum_every_1, len(said_1))
        percentage = said_1_and_wrong.float().sum(dim=0) / len(said_1)
        per_class_mistakes[label_id] = [
            percentage,
            float(f"{float(len(true_id)) / trues_per_label[label_id].item():.4f}"),
            len(true_id),
            float(f"{float(all_0) / trues_per_label[label_id].item():.4f}"),
            int(all_0),
            int(trues_per_label[label_id].item()),
        ]

    mistakes_data_header = [
        "Class",
        *[
            f"{class_label} ({class_index})"
            for class_index, class_label in class_mapping.items()
        ],
        "% mistakes",
        "N mistakes",
        "% All 0s",
        "N All 0s",
        "Total Trues",
    ]
    mistakes_data_list = []
    for class_index, mistake_data in per_class_mistakes.items():
        mistakes_data_list.append(
            [
                f"{class_mapping[class_index]} ({class_index})",
                *[f"{v:.2%}" for v in mistake_data[0].tolist()],
                f"{mistake_data[1]:.0%}",
                f"{mistake_data[2]}",
                f"{mistake_data[3]:.0%}",
                f"{mistake_data[4]}",
                f"{mistake_data[5]}",
            ]
        )

    summary_data_list = []
    for class_index, class_label in class_mapping.items():
        if class_index in per_class_mistakes:
            mistakes = per_class_mistakes[class_index]

            p = round(mistakes[1] * 100)

            _, order_indices = torch.sort(mistakes[0], descending=True)
            order_indices = order_indices.tolist()
            values = []
            for i in order_indices:
                val = mistakes[0][i].item()
                if val >= 0.0:
                    label = class_mapping[i]

                    s = f"{label} ({val:.0%})"
                    values.append(s)

            summary_data_list.append(
                [f"{class_label} ({class_index})", f"{p}%", "->", *values]
            )
        else:
            summary_data_list.append([f"{class_label} ({class_index})", "X"])

    return mistakes_data_header, mistakes_data_list, summary_data_list


def _predict(model, data_loader, trainer):
    model = model.eval()

    y_true = []
    y_pred = []
    for x, y in data_loader:
        x = x.to(torch.device("cuda:0"))

        with torch.no_grad():
            output = model(x)

        output = trainer.activation(output)
        _y_pred = trainer.inference(output)

        y_true.extend(y.tolist())
        y_pred.extend(_y_pred.tolist())

    y_true = torch.tensor(y_true)
    y_pred = torch.tensor(y_pred)

    return y_true, y_pred


def analyze_crossfolds(
    model_config: MinModelConfig,
    data_config: NetworkDataConfig,
    crossfold_config: CrossFoldConfiguration,
    model_file_path: str,
    model_filename: str,
    folds_file_path: str,
    folds_filename: str,
    labeler_path: str,
    labeler_filename: str,
    analysis_path: str,
    analysis_filename: str,
):
    logger.info("Loading labeler stored state")
    with open(os.path.join(labeler_path, labeler_filename)) as f:
        labeler_data = json.load(f)

    logger.info("Loading Files")
    # class_mapping: label_id -> label_name
    # hash_formula: formula_hash -> formula_object
    # hash_label:
    #   single label: formula_hash -> label_id
    #   multilabel: formula_hash -> List[label_id]
    # data_reconstruction: point_index -> formula_object
    (cv_data_splitter, class_mapping, *_) = categorical_loader(
        **data_config,
        cross_fold_configuration=crossfold_config,
        labeler_stored_state=labeler_data,
        _legacy_load_without_batch=True,
    )

    print(class_mapping)

    if not isinstance(cv_data_splitter, NetworkDatasetCrossFoldSplitter):
        raise NotImplementedError("Only cross fold splitter is supported")

    with open(os.path.join(folds_file_path, folds_filename)) as f:
        precalculated_folds = json.load(f)
    cv_data_splitter.load_precalculated_folds(fold_dict=precalculated_folds)

    n_classes = len(class_mapping)
    logger.debug(f"{n_classes} classes detected")

    input_shape = get_input_dim(next(iter(cv_data_splitter))[0])
    assert len(input_shape) == 1, "The input dimension is different from 1"

    model_config["input_dim"] = input_shape[0]
    model_config["output_dim"] = n_classes

    logger.info(f"Total Dataset size: {cv_data_splitter.dataset_size}")

    trainer = MLPTrainer(
        logging_variables="all",
        n_classes=n_classes,
        metrics_average="macro",
        multilabel=True,
    )

    with open(os.path.join(analysis_path, analysis_filename), "w") as analysis:

        n_splits = cv_data_splitter.n_splits
        for i, (_, test_data, _) in enumerate(cv_data_splitter, start=1):
            logger.info(f"Running eval for crossfold {i}/{n_splits}")

            cf_model_name = model_filename.format(i)

            info = torch.load(os.path.join(model_file_path, cf_model_name))
            model_weights = info["model"]
            class_mapping = info["class_mapping"]

            model = trainer.init_model(**model_config)
            model.load_state_dict(model_weights)
            model = model.to(torch.device("cuda:0"))

            data_loader = trainer.init_dataloader(
                test_data,
                mode="test",
                batch_size=1024,
                pin_memory=False,
                shuffle=False,
                num_workers=0,
            )

            y_true, y_pred = _predict(model, data_loader, trainer)

            (
                mistakes_data_header,
                mistakes_data_list,
                summary_data_list,
            ) = _get_multilabel_analysis(
                y_true=y_true, y_pred=y_pred, class_mapping=class_mapping
            )

            analysis.writelines([f"CV {i}", "\n"])
            analysis.writelines(
                [
                    tabulate(
                        mistakes_data_list, mistakes_data_header, tablefmt="plain"
                    ),
                    "\n\n",
                ]
            )
            analysis.writelines(
                [tabulate(summary_data_list, tablefmt="plain"), "\n\n\n\n"]
            )


def main():
    model_hash = "40e65407aa"
    selector = NoFilter()
    test_selector = NullFilter()

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
    #         (None, 1),
    #         (None, 2),
    #         (None, 3),
    #         (None, 4),
    #     ]
    # )
    # label_logic = MulticlassOpenQuantifierLabeler()
    # --- multilabel
    # label_logic = MultiLabelAtomicLabeler()
    # label_logic = MultilabelQuantifierLabeler()
    # label_logic = MultilabelRestrictionLabeler(mode="both", class_for_no_label=False)
    # label_logic = MultilabelRestrictionLabeler(mode="upper", class_for_no_label=True)
    # label_logic = MultilabelFormulaElementLabeler()
    label_logic = MultilabelFormulaElementWithAtomicPositionLabeler()
    labeler = LabelerApply(labeler=label_logic)
    # * /labelers

    model_config: MinModelConfig = {
        "num_layers": 3,
        "input_dim": None,
        "hidden_dim": 128,
        "hidden_layers": [256, 256, 256],
        "output_dim": None,
        "use_batch_norm": True,
    }
    crossfold_config: CrossFoldConfiguration = {
        "n_splits": 5,  # not used
        "shuffle": True,  # not used
        "random_state": None,  # not used
        "defer_loading": True,
        "use_stratified": None,
    }

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

    model_path = os.path.join(
        "results",
        "v4",
        "crossfold_raw",
        model_hash,
        "models",
    )
    base_model_filename = "NoFilter()-MultilabelFormulaElementWithAtomicPositionLabeler()-CV-1L256+2L256+3L256-32b-0.0005lr_cf{}.pt"

    crossfold_path = os.path.join(
        "results",
        "v4",
        "crossfold_raw",
        model_hash,
        "info",
    )
    crossfold_filename = "NoFilter()-MultilabelFormulaElementWithAtomicPositionLabeler()-CV-1L256+2L256+3L256-32b-0.0005lr.folds"

    labeler_path = os.path.join(
        "results",
        "v4",
        "crossfold_raw",
        model_hash,
        "labelers",
    )
    labeler_filename = "NoFilter()-MultilabelFormulaElementWithAtomicPositionLabeler()-CV-1L256+2L256+3L256-32b-0.0005lr.labeler"

    analysis_path = os.path.join(
        "results",
        "v4",
        "crossfold_raw",
        model_hash,
        "analysis",
    )
    os.makedirs(analysis_path, exist_ok=True)
    analysis_filename = crossfold_filename.split(".folds")[0] + ".txt"

    start = timer()
    analyze_crossfolds(
        model_config=model_config,
        data_config=data_config,
        crossfold_config=crossfold_config,
        model_file_path=model_path,
        model_filename=base_model_filename,
        folds_file_path=crossfold_path,
        folds_filename=crossfold_filename,
        labeler_path=labeler_path,
        labeler_filename=labeler_filename,
        analysis_path=analysis_path,
        analysis_filename=analysis_filename,
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
    main()
