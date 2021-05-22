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
results_path_10 = os.path.join(results_path, "10up")
results_path_20 = os.path.join(results_path, "20up")
results_path_distributions = os.path.join(results_path, "distributions")

os.makedirs(results_path, exist_ok=True)
os.makedirs(results_path_10, exist_ok=True)
os.makedirs(results_path_20, exist_ok=True)
os.makedirs(results_path_distributions, exist_ok=True)

graph_labels_path = os.path.join("data", "graphs", "labels")
mapping = FormulaMapping(os.path.join("data", "formulas.json"))

all_hashes = get_formula_hashes(path=graph_labels_path)

existing_hashes = get_existing_hashes(path=results_path_10) & get_existing_hashes(
    path=results_path_20
)

stats_mapping = {}
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
    percent_10up = 0
    percent_20up = 0

    relevant_distributions10 = defaultdict(int)
    relevant_distributions20 = defaultdict(int)
    for distribution, distribution_labels in train_labels.items():
        for labels in distribution_labels:
            all_0s += int(np.sum(labels) == 0)

            relevant10 = bool(np.mean(labels) >= 0.1)
            relevant20 = bool(np.mean(labels) >= 0.2)

            percent_10up += int(relevant10)
            percent_20up += int(relevant20)

            if relevant10:
                relevant_distributions10[str(distribution)] += int(relevant10)
            if relevant20:
                relevant_distributions20[str(distribution)] += int(relevant20)
        total += len(distribution_labels)

    stats_mapping[formula_hash] = {
        "formula": str(formula),
        "formula_repr": repr(formula),
        "all_0s%": float(all_0s) / total,
        "all_0s": all_0s,
        "10%+%": float(percent_10up) / total,
        "10%+": percent_10up,
        "20%+%": float(percent_20up) / total,
        "20%+": percent_20up,
        "total": total,
    }
    percent_10[formula_hash] = percent_10up
    percent_20[formula_hash] = percent_20up

    relevant_count_10up = list(relevant_distributions10.values())
    relevant_count_20up = list(relevant_distributions20.values())

    n, bins, patches = ax1.hist(relevant_count_10up, 30, facecolor="g")
    ax1.set_title(
        f"{str(formula)}: total {percent_10up} graphs ({float(percent_10up) / total:.2%})"
    )
    ax1.set_xlabel("N graphs with 10%+ positives for formula")
    fig.tight_layout()
    fig.savefig(os.path.join(results_path_10, f"{formula_hash}.png"))

    ax1.clear()

    n, bins, patches = ax1.hist(relevant_count_20up, 30, facecolor="g")
    ax1.set_title(
        f"{str(formula)}: total {percent_20up} graphs ({float(percent_20up) / total:.2%})"
    )
    ax1.set_xlabel("N graphs with 20%+ positives for formula")
    fig.tight_layout()
    fig.savefig(os.path.join(results_path_20, f"{formula_hash}.png"))

    ax1.clear()

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

percent_10["min"] = min(percent_10.values())
percent_10["complement"] = total - percent_10["min"]
percent_10["total"] = total
with open(os.path.join(results_path, "percent_10.json"), "w", encoding="utf-8") as f:
    json.dump(percent_10, f, ensure_ascii=False, indent=2)
percent_20["min"] = min(percent_20.values())
percent_20["complement"] = total - percent_20["min"]
percent_20["total"] = total
with open(os.path.join(results_path, "percent_20.json"), "w", encoding="utf-8") as f:
    json.dump(percent_20, f, ensure_ascii=False, indent=2)
