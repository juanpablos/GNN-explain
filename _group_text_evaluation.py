import csv
import json
import os
import re
from collections import defaultdict
from typing import Any, Dict, Tuple

import numpy as np

model_hash = "40e65407aa"
model_name = "NoFilter()-TextSequenceAtomic()-CV-1L256+2L256+3L256-emb4-lstmcellIN256-lstmH256-initTrue-catTrue-drop0-compFalse-d256-32b-0.0005lr"
evaluation_path = os.path.join(
    "results",
    "v4",
    "crossfold_raw",
    model_hash,
    "text",
    "evaluation",
    model_name,
)


def extract_metrics(
    formula_correct_count: Dict[str, Dict[str, Any]], total_formula_number: float
) -> Tuple[float, float, float, float]:
    model_accuracy_count = 0.0
    model_semantic_precision_micro_sum = 0.0
    model_semantic_recall_micro_sum = 0.0
    model_semantic_accuracy_micro_sum = 0.0

    for metrics in formula_correct_count.values():
        model_accuracy_count += metrics["count"]
        model_semantic_precision_micro_sum += (
            metrics["semantic_precision"] * metrics["number_of_gnns"]
        )
        model_semantic_recall_micro_sum += (
            metrics["semantic_recall"] * metrics["number_of_gnns"]
        )
        model_semantic_accuracy_micro_sum += (
            metrics["semantic_accuracy"] * metrics["number_of_gnns"]
        )

    model_accuracy = model_accuracy_count / total_formula_number
    model_semantic_precision = model_semantic_precision_micro_sum / total_formula_number
    model_semantic_recall = model_semantic_recall_micro_sum / total_formula_number
    model_semantic_accuracy = model_semantic_accuracy_micro_sum / total_formula_number

    return (
        model_accuracy,
        model_semantic_precision,
        model_semantic_recall,
        model_semantic_accuracy,
    )


for i in range(1, 5 + 1):
    cv_path = os.path.join(evaluation_path, f"CV{i}")
    cv_grouped_path = os.path.join(evaluation_path, f"CV{i}_grouped")
    os.makedirs(cv_grouped_path, exist_ok=True)

    formula_original_correct_count = {}
    formula_compressed_correct_count = {}
    total_formulas = 0

    for evaluation_filename in os.listdir(cv_path):
        original_grouped_formulas = defaultdict(int)
        compressed_grouped_formulas = defaultdict(int)
        grouped_metrics = {"precision": [], "recall": [], "accuracy": []}
        original_formula_metrics = defaultdict(dict)
        compressed_grouped_metrics = defaultdict(lambda: defaultdict(list))

        total_output_formulas = 0
        with open(os.path.join(cv_path, evaluation_filename)) as f:
            next(f)  # ----
            correct_formula = next(f).strip()
            correct_compressed_formula = re.sub(r"\d", "*", correct_formula)
            next(f)  # ----
            next(f)  # blank

            reader = csv.DictReader(f, delimiter=";")

            for eval_line in reader:
                formula = eval_line["formula"].strip()
                precision = float(eval_line["precision"])
                recall = float(eval_line["recall"])
                accuracy = float(eval_line["accuracy"])

                original_grouped_formulas[formula] += 1

                compressed_formula = re.sub(r"\d", "*", formula)
                compressed_grouped_formulas[compressed_formula] += 1

                original_formula_metrics[formula]["precision"] = precision
                original_formula_metrics[formula]["recall"] = recall
                original_formula_metrics[formula]["accuracy"] = accuracy

                grouped_metrics["precision"].append(precision)
                grouped_metrics["recall"].append(recall)
                grouped_metrics["accuracy"].append(accuracy)

                compressed_grouped_metrics[compressed_formula]["precision"].append(
                    precision
                )
                compressed_grouped_metrics[compressed_formula]["recall"].append(recall)
                compressed_grouped_metrics[compressed_formula]["accuracy"].append(
                    accuracy
                )

                total_output_formulas += 1

        formula_precision = np.mean(grouped_metrics["precision"])
        formula_recall = np.mean(grouped_metrics["recall"])
        formula_accuracy = np.mean(grouped_metrics["accuracy"])

        grouped_filename, file_extension = os.path.splitext(evaluation_filename)
        with open(
            os.path.join(
                cv_grouped_path, f"{grouped_filename}_original{file_extension}"
            ),
            "w",
        ) as o:
            correct_count = original_grouped_formulas.get(correct_formula, 0)
            correct_ratio = correct_count / total_output_formulas

            formula_original_correct_count[correct_formula] = {
                "count": correct_count,
                "accuracy_ratio": correct_ratio,
                "semantic_precision": formula_precision,
                "semantic_recall": formula_recall,
                "semantic_accuracy": formula_accuracy,
                "number_of_gnns": total_output_formulas,
            }

            expected_header = "\n".join(
                [
                    "-" * 20,
                    correct_formula,
                    f"\n\tCorrect %: {correct_ratio:.0%}",
                    f"\tAvg Precision: {formula_precision:.2}",
                    f"\tAvg Recall: {formula_recall:.2}",
                    f"\tAvg Accuracy: {formula_accuracy:.2}",
                    "-" * 20,
                ]
            )
            o.writelines([expected_header, "\n\n"])

            for formula, count in sorted(
                original_grouped_formulas.items(), key=lambda x: x[1], reverse=True
            ):
                gnn_formula_precision = original_formula_metrics[formula]["precision"]
                gnn_formula_recall = original_formula_metrics[formula]["recall"]
                gnn_formula_accuracy = original_formula_metrics[formula]["accuracy"]

                o.writelines(
                    [
                        formula,
                        f"\n\t{count}: {count/total_output_formulas:>12.0%}",
                        f"\n\tPrecision: {gnn_formula_precision:.4}",
                        f"\n\tRecall: {gnn_formula_recall:.4}",
                        f"\n\tAccuracy: {gnn_formula_accuracy:.4}",
                        "\n",
                    ]
                )
        with open(
            os.path.join(
                cv_grouped_path, f"{grouped_filename}_compressed{file_extension}"
            ),
            "w",
        ) as o:
            correct_count = compressed_grouped_formulas.get(
                correct_compressed_formula, 0
            )
            correct_ratio = correct_count / total_output_formulas

            max_precision_for_compressed_raw = 0.0
            max_recall_for_compressed_raw = 0.0
            max_accuracy_for_compressed_raw = 0.0
            total_weight = 0.0
            for _metrics in compressed_grouped_metrics.values():
                weight = float(len(_metrics["precision"]))
                max_precision_for_compressed_raw += max(_metrics["precision"]) * weight
                max_recall_for_compressed_raw += max(_metrics["recall"]) * weight
                max_accuracy_for_compressed_raw += max(_metrics["accuracy"]) * weight

                total_weight += weight

            # we are registering values for 'correct_formula'
            formula_compressed_correct_count[correct_formula] = {
                "count": correct_count,
                "accuracy_ratio": correct_ratio,
                "semantic_precision": max_precision_for_compressed_raw / total_weight,
                "semantic_recall": max_recall_for_compressed_raw / total_weight,
                "semantic_accuracy": max_accuracy_for_compressed_raw / total_weight,
                "number_of_gnns": total_output_formulas,
            }

            expected_header = "\n".join(
                [
                    "-" * 20,
                    correct_formula,
                    f"\n\tCorrect %: {correct_ratio:.0%}",
                    f"\tAvg Precision: {formula_precision:.2}",
                    f"\tAvg Recall: {formula_recall:.2}",
                    f"\tAvg Accuracy: {formula_accuracy:.2}",
                    "-" * 20,
                ]
            )
            o.writelines([expected_header, "\n\n"])

            for formula, count in sorted(
                compressed_grouped_formulas.items(), key=lambda x: x[1], reverse=True
            ):
                gnn_formula_precision = max(
                    compressed_grouped_metrics[formula]["precision"]
                )
                gnn_formula_recall = max(compressed_grouped_metrics[formula]["recall"])
                gnn_formula_accuracy = max(
                    compressed_grouped_metrics[formula]["accuracy"]
                )

                o.writelines(
                    [
                        formula,
                        f"\n\t{count}: {count/total_output_formulas:>12.0%}",
                        f"\n\tPrecision: {gnn_formula_precision:.4}",
                        f"\n\tRecall: {gnn_formula_recall:.4}",
                        f"\n\tAccuracy: {gnn_formula_accuracy:.4}",
                        "\n",
                    ]
                )

        total_formulas += total_output_formulas

    (
        original_model_accuracy,
        original_model_semantic_precision,
        original_model_semantic_recall,
        original_model_semantic_accuracy,
    ) = extract_metrics(
        formula_correct_count=formula_original_correct_count,
        total_formula_number=total_formulas,
    )
    (
        compressed_model_accuracy,
        compressed_model_semantic_precision,
        compressed_model_semantic_recall,
        compressed_model_semantic_accuracy,
    ) = extract_metrics(
        formula_correct_count=formula_compressed_correct_count,
        total_formula_number=total_formulas,
    )

    summary_dict = {
        "original_accuracy": original_model_accuracy,
        "compressed_accuracy": compressed_model_accuracy,
        "original_semantic_precision": original_model_semantic_precision,
        "compressed_semantic_precision": compressed_model_semantic_precision,
        "original_semantic_recall": original_model_semantic_recall,
        "compressed_semantic_recall": compressed_model_semantic_recall,
        "original_semantic_accuracy": original_model_semantic_accuracy,
        "compressed_semantic_accuracy": compressed_model_semantic_accuracy,
        "compressed": {
            formula: [
                metrics["accuracy_ratio"],
                metrics["semantic_precision"],
                metrics["semantic_recall"],
                metrics["semantic_accuracy"],
            ]
            for formula, metrics in formula_compressed_correct_count.items()
        },
        "original": {
            formula: [
                metrics["accuracy_ratio"],
                metrics["semantic_precision"],
                metrics["semantic_recall"],
                metrics["semantic_accuracy"],
            ]
            for formula, metrics in formula_original_correct_count.items()
        },
    }
    with open(
        os.path.join(cv_grouped_path, f".summary_file.txt"),
        "w",
    ) as o:
        json.dump(summary_dict, o, indent=2)
