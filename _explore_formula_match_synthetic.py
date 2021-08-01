import csv
import json
import os

import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support

from src.data.gnn.utils import prepare_files
from src.data.graph_transform import graph_labeled_data_to_graph
from src.generate_graphs import graph_data_stream_pregenerated_graphs_test
from src.graphs.foc import *
from src.models.ac_gnn import ACGNN

with open(os.path.join("data", "formulas.json")) as f:
    formulas = json.load(f)

# try to find the formula that the GNN is representing

gpu_num = 0

model_hash = "40e65407aa"

to_search_formula = FOC(
    AND(Property("RED"), Exist(AND(Role("EDGE"), Property("BLACK")), None, 5))
)
formula_graphs_hash = to_search_formula.get_hash()

graphs_path = os.path.join("data", "graphs")

gnns_path = os.path.join("data", "gnns_v4", model_hash, "original")
formula_hash_gnn_file = prepare_files(path=gnns_path, model_hash=model_hash)[
    formula_graphs_hash
]

eval_dir = os.path.join(
    "results", "validity_checks", "formula_search", model_hash, formula_graphs_hash
)
os.makedirs(eval_dir, exist_ok=True)

print("loading testing graphs data")
graphs_data_data = list(
    graph_data_stream_pregenerated_graphs_test(
        formula=to_search_formula,
        graphs_path=graphs_path,
        graphs_filename="test_graphs_v2_10626.pt",
        n_properties=4,
        pregenerated_labels_file=f"{formula_graphs_hash}_labels_test.pt",
    )
)
print("loading testing graphs")
graphs_data = [
    graph_labeled_data_to_graph(graph_data=graph_data)
    for graph_data in graphs_data_data
]
# only the first one, there might be more, but just take the first
model_weights = torch.load(os.path.join(gnns_path, formula_hash_gnn_file))[0][0]

device = torch.device(f"cuda:{gpu_num}")

model = (
    ACGNN(
        input_dim=4,
        hidden_dim=8,
        output_dim=2,
        aggregate_type="add",
        combine_type="identity",
        num_layers=2,
        combine_layers=1,
        mlp_layers=1,
        task="node",
        use_batch_norm=False,
    )
    .to(device)
    .eval()
)
model.load_state_dict(model_weights)

_gnn_predictions = []
with torch.no_grad():
    # to try to find a formula that could explain what the gnn is doing
    for data in graphs_data_data:
        gnn_pred = (
            model(
                x=data.x.to(device),
                edge_index=data.edge_index.to(device),
                batch=None,
            )
            .max(1)[1]
            .detach()
            .cpu()
            .numpy()
        )
        _gnn_predictions.append(gnn_pred)

gnn_predictions = np.concatenate(_gnn_predictions)

# try to also find the formula that could explain the dataset's labels
graphs_labels = np.concatenate(
    [_graphs_data_data.y.numpy() for _graphs_data_data in graphs_data_data]
)

dataset_formulas_evaluation = []
gnn_formulas_evaluation = []

for i, (formula_hash, formula) in enumerate(formulas.items(), start=1):
    formula = eval(formula)

    print(f"{i}/{len(formulas)}, {formula_hash}: {formula}")

    _attempt_labels_for_dataset_graph = []
    for graph in graphs_data:
        _attempt_labels_for_dataset_graph.append(FOC(formula)(graph=graph))
    attempt_labels_for_dataset_graph = np.concatenate(_attempt_labels_for_dataset_graph)

    gnn_precision, gnn_recall, gnn_f1, _ = precision_recall_fscore_support(
        gnn_predictions, attempt_labels_for_dataset_graph, average="binary"
    )
    cora_precision, cora_recall, cora_f1, _ = precision_recall_fscore_support(
        graphs_labels, attempt_labels_for_dataset_graph, average="binary"
    )

    dataset_formulas_evaluation.append(
        (formula_hash, repr(formula), (cora_f1, cora_precision, cora_recall))
    )
    gnn_formulas_evaluation.append(
        (formula_hash, repr(formula), (gnn_f1, gnn_precision, gnn_recall))
    )


with open(
    os.path.join(eval_dir, "dataset.csv"), "w", encoding="utf-8", newline=""
) as f:
    csv_writer = csv.writer(f, delimiter=";")

    for formula_hash, formula_repr, (f1, precision, recall) in sorted(
        dataset_formulas_evaluation, reverse=True, key=lambda x: x[2]
    ):
        csv_writer.writerow([formula_hash, formula_repr, precision, recall, f1])


with open(os.path.join(eval_dir, "gnn.csv"), "w", encoding="utf-8", newline="") as f:
    csv_writer = csv.writer(f, delimiter=";")

    for formula_hash, formula_repr, (f1, precision, recall) in sorted(
        gnn_formulas_evaluation, reverse=True, key=lambda x: x[2]
    ):
        csv_writer.writerow([formula_hash, formula_repr, precision, recall, f1])
