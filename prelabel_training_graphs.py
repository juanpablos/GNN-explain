import json
import os
from collections import defaultdict

import torch
from torch_geometric.utils import to_networkx

from src.data.formula_index import FormulaMapping
from src.data.gnn.utils import prepare_files
from src.graphs import FOC


def generate_graph_labels(graphs, use_formula, train: bool):
    graph_labels = []

    total_graphs = len(graphs)
    for i, data in enumerate(graphs):
        if i % 10_000 == 0 and not train:
            print(f"{i}/{total_graphs}")
        graph = to_networkx(data, node_attrs=["properties"], to_undirected=True)

        labels = use_formula(graph)
        graph_labels.append(labels)
    return graph_labels


def list_existing_labels(labels_path):
    existing_hashes = set()
    for label_file in os.listdir(labels_path):
        if label_file.endswith(".pt"):
            _hash = label_file.split("_labels")[0]
            existing_hashes.add(_hash)
    return existing_hashes


def generate_train_labels(train_data, use_formula):
    train_labels_data = defaultdict(list)
    counter = 0
    total_graphs = len(train_data) * len(next(iter(train_data.values())))
    for distribution, graphs in train_data.items():
        if counter % 10_000 == 0:
            print(f"{counter}/{total_graphs}")
        train_graph_labels = generate_graph_labels(
            graphs=graphs, use_formula=use_formula, train=True
        )
        train_labels_data[distribution].extend(train_graph_labels)

        counter += len(graphs)

    return train_labels_data


def generate_test_labels(test_data, use_formula):
    test_graph_labels = generate_graph_labels(
        graphs=test_data, use_formula=use_formula, train=False
    )
    return test_graph_labels


graphs_path = os.path.join("data", "graphs")
graph_labels_path = os.path.join(graphs_path, "labels")
os.makedirs(graph_labels_path, exist_ok=True)

train_filename = "train_graphs_v2_354200.pt"
test_filename = "test_graphs_v2_10626.pt"

gnn_path = os.path.join("data", "full_gnn", "40e65407aa")
formula_files = prepare_files(path=gnn_path, model_hash="40e65407aa")
# with open(os.path.join("data", "formulas_v3.json"), encoding="utf-8") as f:
#     formula_files = json.load(f)

formula_mapping = FormulaMapping(os.path.join("data", "formulas.json"))

train_graphs_data = torch.load(os.path.join(graphs_path, train_filename))
test_graphs_data = torch.load(os.path.join(graphs_path, test_filename))

pre_existing_labels = list_existing_labels(graph_labels_path)
manual_run_hashes = []
# manual_run_hashes = [
#     "688d12b701",
#     "676d3c83b1",
#     "dc670b1bec",
#     "4805042859",
#     "652c706f1b",
#     "a8c45da01a",
#     "40a6f530d2",
#     "6aa72b4580",
# ]

run_hashes = manual_run_hashes if manual_run_hashes else formula_files

for formula_hash in run_hashes:
    formula = FOC(formula_mapping[formula_hash])
    if formula_hash in pre_existing_labels:
        print("Skipping", formula_hash, str(formula))
        continue

    print("Currently labeling for", formula_hash, str(formula))

    train_graph_labels = generate_train_labels(
        train_data=train_graphs_data, use_formula=formula
    )
    torch.save(
        train_graph_labels,
        os.path.join(graph_labels_path, f"{formula_hash}_labels_train.pt"),
    )

    test_graph_labels = generate_test_labels(
        test_data=test_graphs_data, use_formula=formula
    )
    torch.save(
        test_graph_labels,
        os.path.join(graph_labels_path, f"{formula_hash}_labels_test.pt"),
    )
