import os

import networkx as nx
import numpy as np
import torch
from sklearn.metrics import classification_report, precision_score, recall_score
from torch_geometric.utils import to_networkx

from src.graphs.foc import *

label = 6
path = os.path.join("data", "cora_data", "original")
cora_dataset = f"original_reduced_cora_l{label}"

data = torch.load(os.path.join(path, f"{cora_dataset}.pt"))

properties = torch.argmax(data.x, dim=1)
data.properties = properties
graph = to_networkx(data, to_undirected=True, node_attrs=["properties", "y"])

true_expected = np.array(list(nx.get_node_attributes(graph, "y").values()))
random_pred = np.random.choice([0, 1], size=true_expected.size, replace=True)

extracted_precision = precision_score(true_expected, random_pred)
extracted_recall = recall_score(true_expected, random_pred)
report = classification_report(true_expected, random_pred)

print(report)
print(
    "Macro precision:",
    extracted_precision,
    "\nMacro recall:",
    extracted_recall,
    end="\n\n",
)

line = (
    f"{str(report)}\n"
    f"Macro precision: {extracted_precision}\nMacro recall: {extracted_recall}\n\n"
)

with open(
    os.path.join("results", "cora", f"cora_random_{label}.pt"), "w", encoding="utf-8"
) as f:
    f.write(line)
