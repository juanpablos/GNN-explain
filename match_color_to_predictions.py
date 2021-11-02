import csv
import json
import os
from collections import defaultdict

from src.data.formulas.extract import ColorsExtractor
from src.graphs.foc import *

model_hash = "40e65407aa"
model_name = "NoFilter()-TextSequenceAtomic()-CV-F(True)-ENC[color1024x4+16,lower512x1+16,upper512x1+16]-FINE[2]-emb4-lstmcellIN8-lstmH8-initTrue-catTrue-drop0-compFalse-d256-32b-0.001lr"
evaluation_path = os.path.join(
    "results",
    "v4",
    "crossfold_raw",
    model_hash,
    "text+encoder_v2+color(rem,rep)",
    "evaluation - test",
    model_name,
)

limit_iterations = 1


for i in range(1, 5 + 1):
    if i > limit_iterations:
        break

    cv_path = os.path.join(evaluation_path, f"CV{i}")
    target_path = os.path.join(evaluation_path, f"CV{i}_matches")
    os.makedirs(target_path, exist_ok=True)

    color_extractor = ColorsExtractor(hop=0)

    global_matches = 0.0
    global_formulas = 0.0

    global_formula_matches = {}

    for evaluation_filename in os.listdir(cv_path):
        if ".bleu" in evaluation_filename:
            continue

        n_formulas = defaultdict(int)
        match_for_formula = defaultdict(bool)
        total_matches = 0.0
        total_formulas = 0.0

        with open(os.path.join(cv_path, evaluation_filename)) as f:
            next(f)  # ----
            expected_formula = next(f).strip()
            next(f)  # bleu
            next(f)  # ----
            next(f)  # blank

            expected_formula_object = eval(expected_formula)
            allowed_values = set(color_extractor(expected_formula_object))

            reader = csv.DictReader(f, delimiter=";")
            for eval_line in reader:
                formula = eval_line["formula"].strip()

                n_formulas[formula] += 1

                if formula not in match_for_formula:
                    formula_object = eval(formula)

                    if formula_object:
                        predicted_values = set(color_extractor(formula_object))
                        match_for_formula[formula] = not allowed_values.isdisjoint(
                            predicted_values
                        )
                    else:
                        match_for_formula[formula] = False

                total_matches += int(match_for_formula[formula])
                total_formulas += 1

            global_formula_matches[expected_formula] = total_matches / total_formulas

        global_matches += total_matches
        global_formulas += total_formulas

        matches_ratio = total_matches / total_formulas

        filename, file_extension = os.path.splitext(evaluation_filename)
        with open(
            os.path.join(target_path, f"{filename}_matches{file_extension}"),
            "w",
        ) as o:
            expected_header = "\n".join(
                [
                    "-" * 20,
                    expected_formula,
                    f"\n\tMatches %: {matches_ratio:.2%}",
                    f"\tMatches: {total_matches:.0f}",
                    f"\tTotal: {total_formulas:.0f}",
                    "-" * 20,
                ]
            )
            o.writelines([expected_header, "\n\n"])

            for formula, count in sorted(
                n_formulas.items(), key=lambda x: x[1], reverse=True
            ):
                formula_matchs = match_for_formula[formula]

                o.writelines(
                    [
                        formula,
                        f"\n\t{count}: {count/total_formulas:>12.0%}",
                        f"\n\tMatch: {formula_matchs}",
                        "\n",
                    ]
                )

    summary_dict = {
        "total_matches_ratio": global_matches / global_formulas,
        "formulas": {
            formula: [matches_ratio]
            for formula, matches_ratio in global_formula_matches.items()
        },
    }
    with open(
        os.path.join(target_path, f".summary_file.txt"),
        "w",
    ) as o:
        json.dump(summary_dict, o, indent=2)
