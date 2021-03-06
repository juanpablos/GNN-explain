import json
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.data.formula_index import FormulaMapping
from src.data.formulas import *
from src.graphs import *


def get_formula_hashes(path: str):
    hashes = []
    for file in os.listdir(path):
        if file.endswith(".pt") and "labels_train" in file:
            _hash = file.split(".")[0].split("_")[0]
            hashes.append(_hash)
    return hashes


def get_existing_hashes(path: str):
    hashes = set()
    for file in os.listdir(path):
        if file.endswith(".png"):
            _hash = file.split(".")[0]
            hashes.add(_hash)
    return hashes


results_path = os.path.join("results", "analysis", "graph_distributions")
results_path_1 = os.path.join(results_path, "1up")
results_path_10 = os.path.join(results_path, "10up")
results_path_20 = os.path.join(results_path, "20up")
results_path_distributions = os.path.join(results_path, "distributions")

os.makedirs(results_path, exist_ok=True)
os.makedirs(results_path_1, exist_ok=True)
os.makedirs(results_path_10, exist_ok=True)
os.makedirs(results_path_20, exist_ok=True)
os.makedirs(results_path_distributions, exist_ok=True)

graph_labels_path = os.path.join("data", "graphs", "labels")
mapping = FormulaMapping(os.path.join("data", "formulas.json"))

all_hashes = get_formula_hashes(path=graph_labels_path)

existing_hashes = (
    get_existing_hashes(path=results_path_1)
    & get_existing_hashes(path=results_path_10)
    & get_existing_hashes(path=results_path_20)
)

stats_mapping = {}
percent_1 = {}
percent_10 = {}
percent_20 = {}

fig, ax1 = plt.subplots(figsize=(15, 10))

for formula_hash in all_hashes:
    formula = mapping[formula_hash]

    if formula_hash in existing_hashes:
        print("skipping", formula_hash, str(formula))
        continue

    print("working on formula", formula_hash, str(formula))

    train_label_file = os.path.join(
        graph_labels_path, f"{formula_hash}_labels_train.pt"
    )
    train_labels = torch.load(train_label_file)

    total = 0
    all_0s = 0
    percent_1up = 0
    percent_10up = 0
    percent_20up = 0

    all_distributions = set()
    relevant_distributions1 = defaultdict(int)
    relevant_distributions10 = defaultdict(int)
    relevant_distributions20 = defaultdict(int)
    for distribution, distribution_labels in train_labels.items():
        for labels in distribution_labels:
            all_0s += int(np.sum(labels) == 0)

            relevant1 = bool(np.mean(labels) >= 0.01)
            relevant10 = bool(np.mean(labels) >= 0.1)
            relevant20 = bool(np.mean(labels) >= 0.2)

            percent_1up += int(relevant1)
            percent_10up += int(relevant10)
            percent_20up += int(relevant20)

            if relevant1:
                relevant_distributions1[str(distribution)] += int(relevant1)
            if relevant10:
                relevant_distributions10[str(distribution)] += int(relevant10)
            if relevant20:
                relevant_distributions20[str(distribution)] += int(relevant20)

        all_distributions.add(str(distribution))
        total += len(distribution_labels)

    stats_mapping[formula_hash] = {
        "formula": str(formula),
        "formula_repr": repr(formula),
        "all_0s%": float(all_0s) / total,
        "all_0s": all_0s,
        "1%+%": float(percent_1up) / total,
        "1%+": percent_1up,
        "10%+%": float(percent_10up) / total,
        "10%+": percent_10up,
        "20%+%": float(percent_20up) / total,
        "20%+": percent_20up,
        "total": total,
    }
    percent_1[formula_hash] = percent_1up
    percent_10[formula_hash] = percent_10up
    percent_20[formula_hash] = percent_20up

    relevant_count_1up = [
        relevant_distributions1[distribution] for distribution in all_distributions
    ]
    relevant_count_10up = [
        relevant_distributions10[distribution] for distribution in all_distributions
    ]
    relevant_count_20up = [
        relevant_distributions20[distribution] for distribution in all_distributions
    ]

    n, bins, patches = ax1.hist(relevant_count_1up, 30, facecolor="g")
    ax1.set_title(
        f"{str(formula)}: total {percent_1up} graphs ({float(percent_1up) / total:.2%})"
    )
    ax1.set_xlim((0, 200))  # 200 graphs per distribution
    ax1.set_xlabel("N graphs with 1%+ positives for formula")
    fig.tight_layout()
    fig.savefig(os.path.join(results_path_1, f"{formula_hash}.png"))

    ax1.clear()

    n, bins, patches = ax1.hist(relevant_count_10up, 30, facecolor="g")
    ax1.set_title(
        f"{str(formula)}: total {percent_10up} graphs ({float(percent_10up) / total:.2%})"
    )
    ax1.set_xlim((0, 200))  # 200 graphs per distribution
    ax1.set_xlabel("N graphs with 10%+ positives for formula")
    fig.tight_layout()
    fig.savefig(os.path.join(results_path_10, f"{formula_hash}.png"))

    ax1.clear()

    n, bins, patches = ax1.hist(relevant_count_20up, 30, facecolor="g")
    ax1.set_title(
        f"{str(formula)}: total {percent_20up} graphs ({float(percent_20up) / total:.2%})"
    )
    ax1.set_xlim((0, 200))  # 200 graphs per distribution
    ax1.set_xlabel("N graphs with 20%+ positives for formula")
    fig.tight_layout()
    fig.savefig(os.path.join(results_path_20, f"{formula_hash}.png"))

    ax1.clear()

    with open(
        os.path.join(results_path_distributions, f"{formula_hash}_1up.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(relevant_distributions1, f, ensure_ascii=False, indent=2)
    with open(
        os.path.join(results_path_distributions, f"{formula_hash}_10up.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(relevant_distributions10, f, ensure_ascii=False, indent=2)
    with open(
        os.path.join(results_path_distributions, f"{formula_hash}_20up.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(relevant_distributions20, f, ensure_ascii=False, indent=2)

with open(
    os.path.join(results_path, "graph_label_statistics.json"), "w", encoding="utf-8"
) as f:
    json.dump(stats_mapping, f, ensure_ascii=False, indent=2)


total = next(iter(stats_mapping.values()))["total"]

_min = min(percent_1.values())
_max = max(percent_1.values())

percent_1["min"] = _min
percent_1["min_complement"] = total - _min
percent_1["max"] = _max
percent_1["max_complement"] = total - _max
percent_1["total"] = total
with open(os.path.join(results_path, "percent_1.json"), "w", encoding="utf-8") as f:
    json.dump(percent_1, f, ensure_ascii=False, indent=2)

_min = min(percent_10.values())
_max = max(percent_10.values())

percent_10["min"] = _min
percent_10["min_complement"] = total - _min
percent_10["max"] = _max
percent_10["max_complement"] = total - _max
percent_10["total"] = total
with open(os.path.join(results_path, "percent_10.json"), "w", encoding="utf-8") as f:
    json.dump(percent_10, f, ensure_ascii=False, indent=2)

_min = min(percent_20.values())
_max = max(percent_20.values())

percent_20["min"] = _min
percent_20["min_complement"] = total - _min
percent_20["max"] = _max
percent_20["max_complement"] = total - _max
percent_20["total"] = total
with open(os.path.join(results_path, "percent_20.json"), "w", encoding="utf-8") as f:
    json.dump(percent_20, f, ensure_ascii=False, indent=2)
