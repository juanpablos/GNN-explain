import os

import networkx as nx
import numpy as np
import torch
from sklearn.metrics import classification_report, precision_score, recall_score
from torch_geometric.utils import to_networkx
from src.models.ac_gnn import ACGNN
from src.graphs.foc import *


use_random = False

cora_dataset = "processed_cora_ae_h32-mid512-p01_agglomerative"

data_path = os.path.join("data", "cora_data")
model_path = os.path.join("data", "gnns_v2", "40e65407aa")
inference_path = os.path.join(
    "results",
    "v2",
    "exp5 - text - flat encoder",
    "40e65407aa",
    "inference",
    "NoFilter()-TextSequenceAtomic()-ManualFilter(10)-1L1024+2L1024+3L1024-emb4-lstmcellIN1024-lstmH256-initTrue-catTrue-drop0-compFalse-d256-512b-0.005lr.pt",
)

data = torch.load(os.path.join(data_path, f"{cora_dataset}.pt"))
models = torch.load(os.path.join(model_path, f"{cora_dataset}.pt"))
inference_file = os.path.join(inference_path, f"{cora_dataset}.txt")
with open(inference_file, "r") as f:
    inference_formulas = [eval(formula) for formula in f.readlines()]


properties = torch.argmax(data.x, dim=1)
data.properties = properties
graph = to_networkx(data, to_undirected=True, node_attrs=["properties", "y"])

true_expected = np.array(list(nx.get_node_attributes(graph, "y").values()))

base_model = ACGNN(
    input_dim=data.num_features,
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

for i, (model_weights, formula) in enumerate(zip(models, inference_formulas)):
    base_model.load_state_dict(model_weights)
    model = base_model

    with torch.no_grad():
        gnn_pred = (
            model(
                x=data.x.cuda(),
                edge_index=data.edge_index.cuda(),
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

    print(f"model {i}, with formula {formula}")
    print(classification_report(gnn_pred, extracted_pred))
    print(
        "Macro precision:",
        extracted_precision,
        "\nMacro recall:",
        extracted_recall,
        end="\n\n",
    )

# formula = AND(
#     Property("RED"), Exist(AND(Role("EDGE"), Property("RED")), lower=1, upper=None)
# )
# formula = AND(
#     OR(Property("RED"), Property("BLACK")),
#     Exist(AND(Role("EDGE"), Property("GREEN")), lower=None, upper=3),
# )
# formula = OR(Property("BLUE"), Property("BLACK"))
# formula = Property("BLACK")
