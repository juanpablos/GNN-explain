import csv
import json
import os
from collections import defaultdict
from typing import Dict

import numpy as np
from zss import Node, simple_distance

from src.data.formulas.copy_tree import FOC2Node
from src.graphs.foc import *

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

formula_object_to_node_transformator = FOC2Node()
repr_to_node_cache: Dict[str, Node] = {}
edit_distance_cache: Dict[str, float] = {}


def get_or_set_node(node_repr: str):
    try:
        return repr_to_node_cache[node_repr]
    except KeyError:
        formula_object = eval(node_repr)
        _formula_node = formula_object_to_node_transformator(formula_object)
        repr_to_node_cache[node_repr] = _formula_node
        return _formula_node


def edit_distance_for_trees(
    tree_a: Node, repr_a: str, tree_b: Node, repr_b: str
) -> float:
    key = f"{repr_a};{repr_b}"
    try:
        return edit_distance_cache[key]
    except KeyError:
        edit_distance: float = simple_distance(tree_a, tree_b)
        edit_distance_cache[key] = edit_distance
        return edit_distance


for i in range(1, 5 + 1):
    cv_path = os.path.join(evaluation_path, f"CV{i}")
    cv_tree_edit_path = os.path.join(evaluation_path, f"CV{i}_tree")
    os.makedirs(cv_tree_edit_path, exist_ok=True)

    formula_correct_counts = {}
    total_formulas = 0

    for evaluation_filename in os.listdir(cv_path):
        grouped_formula_counter = defaultdict(int)
        grouped_formula_distance = defaultdict(float)

        total_output_formulas = 0
        with open(os.path.join(cv_path, evaluation_filename)) as f:
            next(f)  # ----
            expected_formula = next(f).strip()
            next(f)  # ----
            next(f)  # blank

            expected_formula_node = get_or_set_node(expected_formula)

            reader = csv.DictReader(f, delimiter=";")

            for eval_line in reader:
                pred_formula = eval_line["formula"].strip()
                pred_formula_node = get_or_set_node(eval_line["formula"].strip())

                predicted_edit_distance = edit_distance_for_trees(
                    tree_a=expected_formula_node,
                    repr_a=expected_formula,
                    tree_b=pred_formula_node,
                    repr_b=pred_formula,
                )

                grouped_formula_counter[pred_formula] += 1
                grouped_formula_distance[pred_formula] = predicted_edit_distance

                total_output_formulas += 1

        avg_edit_distance = np.mean(list(grouped_formula_distance.values()))

        grouped_filename, file_extension = os.path.splitext(evaluation_filename)
        with open(
            os.path.join(cv_tree_edit_path, f"{grouped_filename}_tree{file_extension}"),
            "w",
        ) as o:
            correct_count = grouped_formula_counter[expected_formula]
            correct_ratio = correct_count / total_output_formulas

            formula_correct_counts[expected_formula] = {
                "count": correct_count,
                "accuracy_ratio": correct_ratio,
                "edit_distance": avg_edit_distance,
                "number_of_gnns": total_output_formulas,
            }

            expected_header = "\n".join(
                [
                    "-" * 20,
                    expected_formula,
                    f"\n\tCorrect %: {correct_ratio:.0%}",
                    f"\tAvg edit distance: {avg_edit_distance}",
                    "-" * 20,
                ]
            )
            o.writelines([expected_header, "\n\n"])

            for formula, count in sorted(
                grouped_formula_counter.items(), key=lambda x: x[1], reverse=True
            ):
                formula_edit_distance = grouped_formula_distance[formula]

                o.writelines(
                    [
                        formula,
                        f"\n\t{count}: {count/total_output_formulas:>12.0%}",
                        f"\n\tEdit distance: {formula_edit_distance}",
                        "\n",
                    ]
                )

        total_formulas += total_output_formulas

    model_accuracy_count = 0.0
    model_edit_distance_sum = 0.0

    for metrics in formula_correct_counts.values():
        model_accuracy_count += metrics["count"]
        model_edit_distance_sum += metrics["edit_distance"] * metrics["number_of_gnns"]

    model_accuracy = model_accuracy_count / total_formulas
    model_edit_distance = model_edit_distance_sum / total_formulas

    summary_dict = {
        "accuracy": model_accuracy,
        "avg_edit_distance": model_edit_distance,
        "predictions": {
            formula: [
                metrics["accuracy_ratio"],
                metrics["edit_distance"],
            ]
            for formula, metrics in formula_correct_counts.items()
        },
    }
    with open(
        os.path.join(cv_tree_edit_path, f".summary_file.txt"),
        "w",
    ) as o:
        json.dump(summary_dict, o, indent=2)
