import csv
import json
import os

import torch
from sklearn.metrics import precision_recall_fscore_support
from torch_geometric.utils import to_networkx

from src.graphs.foc import *
from src.models.ac_gnn import ACGNN

with open(os.path.join("data", "formulas.json")) as f:
    formulas = json.load(f)

# try to find the formula that the GNN is representing

gpu_num = 0
cora_label = 6

model_hash = "40e65407aa"
cora_base_path = os.path.join(
    "data",
    "cora_data",
)


eval_dir = os.path.join("results", "cora", "formula_search", model_hash, "svd")
os.makedirs(eval_dir, exist_ok=True)

cora_dataset_filename = f"processed_cora_l{cora_label}_svd_agglomerative_d500.pt"
model_filename = f"svd_processed_cora_l{cora_label}_svd_agglomerative_d500_gnn.pt"

eval_file_cora = f"gnn_extract_cora_l{cora_label}_svd_agglomerative_d500.csv"
eval_file_gnn = f"cora_extract_cora_l{cora_label}_svd_agglomerative_d500.csv"


cora_data = torch.load(
    os.path.join(
        cora_base_path,
        "svd",
        cora_dataset_filename,
    )
)
model_weights = torch.load(
    os.path.join(
        cora_base_path,
        model_hash,
        model_filename,
    )
)[0]

properties = torch.argmax(cora_data.x, dim=1)
cora_data.properties = properties
graph = to_networkx(cora_data, to_undirected=True, node_attrs=["properties"])

device = torch.device(f"cuda:{gpu_num}")

model = (
    ACGNN(
        input_dim=cora_data.num_features,
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

with torch.no_grad():
    # try to find a formula that could explain what the gnn is doing
    gnn_pred = (
        model(
            x=cora_data.x.to(device),
            edge_index=cora_data.edge_index.to(device),
            batch=None,
        )
        .max(1)[1]
        .detach()
        .cpu()
        .numpy()
    )

# try to also find the formula that could explain the dataset's labels
cora_labels = cora_data.y.numpy()

cora_formulas_evaluation = []
gnn_formulas_evaluation = []

for i, (formula_hash, formula) in enumerate(formulas.items(), start=1):
    formula = eval(formula)

    if i % 1000 == 0:
        print(f"{i}/{len(formulas)}, {formula_hash}: {formula}")

    attempt_labels_for_cora_graph = FOC(formula)(graph=graph)

    gnn_precision, gnn_recall, gnn_f1, _ = precision_recall_fscore_support(
        gnn_pred, attempt_labels_for_cora_graph, average="binary"
    )
    cora_precision, cora_recall, cora_f1, _ = precision_recall_fscore_support(
        cora_labels, attempt_labels_for_cora_graph, average="binary"
    )

    cora_formulas_evaluation.append(
        (formula_hash, repr(formula), (cora_f1, cora_precision, cora_recall))
    )
    gnn_formulas_evaluation.append(
        (formula_hash, repr(formula), (gnn_f1, gnn_precision, gnn_recall))
    )


with open(
    os.path.join(eval_dir, eval_file_cora), "w", encoding="utf-8", newline=""
) as f:
    csv_writer = csv.writer(f, delimiter=";")

    for formula_hash, formula_repr, (f1, precision, recall) in sorted(
        cora_formulas_evaluation, reverse=True, key=lambda x: x[2]
    ):
        csv_writer.writerow([formula_hash, formula_repr, precision, recall, f1])


with open(
    os.path.join(eval_dir, eval_file_gnn), "w", encoding="utf-8", newline=""
) as f:
    csv_writer = csv.writer(f, delimiter=";")

    for formula_hash, formula_repr, (f1, precision, recall) in sorted(
        gnn_formulas_evaluation, reverse=True, key=lambda x: x[2]
    ):
        csv_writer.writerow([formula_hash, formula_repr, precision, recall, f1])
