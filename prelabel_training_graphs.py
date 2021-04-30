import os

import torch
from torch_geometric.utils import to_networkx

from src.data.formula_index import FormulaMapping
from src.data.gnn.utils import prepare_files
from src.graphs import FOC


def generate_graph_labels(graphs, use_formula):
    graph_labels = []

    total_graphs = len(graphs)
    for i, data in enumerate(graphs):
        if i % 10_000 == 0:
            print(f"{i}/{total_graphs}")
        graph = to_networkx(data, node_attrs=["properties"], to_undirected=True)

        labels = use_formula(graph)
        graph_labels.append(labels)
    return graph_labels


graphs_path = os.path.join("data", "graphs")
graph_labels_path = os.path.join(graphs_path, "labels")
os.makedirs(graph_labels_path, exist_ok=True)

train_filename = "train_graphs_177100.pt"
test_filename = "test_graphs_5313.pt"

gnn_path = os.path.join("data", "full_gnn", "40e65407aa")
formula_files = prepare_files(path=gnn_path, model_hash="40e65407aa")

formula_mapping = FormulaMapping(os.path.join("data", "formulas.json"))

train_graphs_data = torch.load(os.path.join(graphs_path, train_filename))
test_graphs_data = torch.load(os.path.join(graphs_path, test_filename))

for formula_hash, formula_file in formula_files.items():
    formula = FOC(formula_mapping[formula_hash])
    print("Currently labeling for", formula_hash, str(formula))

    train_graph_labels = generate_graph_labels(
        graphs=train_graphs_data, use_formula=formula
    )
    torch.save(
        train_graph_labels,
        os.path.join(graph_labels_path, f"{formula_hash}_labels_train.pt"),
    )

    test_graph_labels = generate_graph_labels(
        graphs=test_graphs_data, use_formula=formula
    )
    torch.save(
        test_graph_labels,
        os.path.join(graph_labels_path, f"{formula_hash}_labels_test.pt"),
    )
