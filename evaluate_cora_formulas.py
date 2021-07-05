import os

import networkx as nx
import numpy as np
import torch
from sklearn.metrics import classification_report, precision_score, recall_score
from torch_geometric.utils import to_networkx

from src.graphs.foc import *
from src.models.ac_gnn import ACGNN

model_hash = "40e65407aa"
model_name = "NoFilter()-TextSequenceAtomic()-CV-1L512+2L512+3L512-emb4-lstmcellIN512-lstmH256-initTrue-catTrue-drop0-compFalse-d256-32b-0.0005lr_cf1"
cora_method = "svd"

cora_data_path = os.path.join("data", "cora_data")
cora_dataset_path = os.path.join(cora_data_path, cora_method)
cora_model_path = os.path.join(cora_data_path, model_hash)

cora_dataset_name = "processed_cora_l0_svd_agglomerative_d500"
cora_model_name = f"{cora_method}_{cora_dataset_name}_gnn"

formula_inference_path = os.path.join(
    "results",
    "v4",
    "crossfold_raw",
    model_hash,
    "text",
    "inference",
    model_name,
)


cora_dataset = torch.load(os.path.join(cora_dataset_path, f"{cora_dataset_name}.pt"))
cora_models = torch.load(os.path.join(cora_model_path, f"{cora_model_name}.pt"))

inference_file = os.path.join(formula_inference_path, f"{cora_model_name}.txt")
inference_comparison_file = os.path.join(
    formula_inference_path, f"{cora_model_name}_eval.txt"
)

with open(inference_file) as f:
    inference_formulas = [eval(formula) for formula in f.readlines()]

properties = torch.argmax(cora_dataset.x, dim=1)
cora_dataset.properties = properties
graph = to_networkx(cora_dataset, to_undirected=True, node_attrs=["properties", "y"])

true_expected = np.array(list(nx.get_node_attributes(graph, "y").values()))

base_model = ACGNN(
    input_dim=cora_dataset.num_features,
    hidden_dim=8,
    output_dim=2,
    aggregate_type="add",
    combine_type="identity",
    num_layers=2,
    combine_layers=1,
    mlp_layers=1,
    task="node",
    use_batch_norm=False,
).cuda()

reports = []
for i, (model_weights, formula) in enumerate(zip(cora_models, inference_formulas)):
    if formula is None:
        print(f"model {i} is gave None as answer")
        continue
    base_model.load_state_dict(model_weights)
    model = base_model

    with torch.no_grad():
        gnn_pred = (
            model(
                x=cora_dataset.x.cuda(),
                edge_index=cora_dataset.edge_index.cuda(),
                batch=None,
            )
            .max(1)[1]
            .detach()
            .cpu()
            .numpy()
        )

    extracted_pred = FOC(formula)(graph=graph)

    extracted_precision = precision_score(gnn_pred, extracted_pred)
    extracted_recall = recall_score(gnn_pred, extracted_pred)

    report = classification_report(gnn_pred, extracted_pred)
    print(f"model {i}, with formula {formula}")
    print(report)
    print(
        "Macro precision:",
        extracted_precision,
        "\nMacro recall:",
        extracted_recall,
        end="\n\n",
    )

    reports.append(
        f"model {i}, with formula {formula}\n{str(report)}\n"
        f"Macro precision: {extracted_precision}\nMacro recall: {extracted_recall}\n\n"
    )

with open(inference_comparison_file, "w", encoding="utf-8") as f:
    for report in reports:
        f.write(report)
