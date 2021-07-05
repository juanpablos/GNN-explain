import json
import os
import re
from collections import defaultdict

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

        total_output_formulas = 0
        with open(os.path.join(cv_path, evaluation_filename)) as f:
            next(f)  # ----
            correct_formula = next(f).strip()
            correct_compressed_formula = re.sub(r"\d", "*", correct_formula)
            next(f)  # ----
            next(f)  # blank

            for formula in f:
                formula = formula.strip()

                original_grouped_formulas[formula] += 1

                compressed_formula = re.sub(r"\d", "*", formula)
                compressed_grouped_formulas[compressed_formula] += 1

                total_output_formulas += 1

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
                "ratio": correct_ratio,
            }

            expected_header = "\n".join(
                [
                    "-" * 20,
                    correct_formula,
                    f"\n\tCorrect %: {correct_ratio:.0%}",
                    "-" * 20,
                ]
            )
            o.writelines([expected_header, "\n\n"])

            for formula, count in sorted(
                original_grouped_formulas.items(), key=lambda x: x[1], reverse=True
            ):
                o.writelines(
                    [
                        formula,
                        "\n",
                        f"\t{count}: {count/total_output_formulas:>8.0%}",
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

            formula_compressed_correct_count[correct_compressed_formula] = {
                "count": correct_count,
                "ratio": correct_ratio,
            }

            expected_header = "\n".join(
                [
                    "-" * 20,
                    correct_formula,
                    f"\n\tCorrect %: {correct_ratio:.0%}",
                    "-" * 20,
                ]
            )
            o.writelines([expected_header, "\n\n"])

            for formula, count in sorted(
                compressed_grouped_formulas.items(), key=lambda x: x[1], reverse=True
            ):
                o.writelines(
                    [
                        formula,
                        "\n",
                        f"\t{count}: {count/total_output_formulas:>8.0%}",
                        "\n",
                    ]
                )

        total_formulas += total_output_formulas

    summary_dict = {
        "original_accuracy": sum(
            metrics["count"] for metrics in formula_original_correct_count.values()
        )
        / float(total_formulas),
        "compressed_accuracy": sum(
            metrics["count"] for metrics in formula_compressed_correct_count.values()
        )
        / float(total_formulas),
        "compressed": {
            formula: metrics["ratio"]
            for formula, metrics in formula_compressed_correct_count.items()
        },
        "original": {
            formula: metrics["ratio"]
            for formula, metrics in formula_original_correct_count.items()
        },
    }
    with open(
        os.path.join(cv_grouped_path, f".summary_file.txt"),
        "w",
    ) as o:
        json.dump(summary_dict, o, indent=2)
