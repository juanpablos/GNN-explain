import networkx as nx
import numpy as np
import torch
from sklearn.metrics import classification_report, precision_score, recall_score
from torch_geometric.utils import to_networkx

from temp_chem import MoleculeNet
from src.graphs.foc import *

use_random = True
# dataset = [torch.load("./data/gnns_v2/40e65407aa/reduced_cora.pt")]
dataset = MoleculeNet("./data/chem", "esol")

graphs = []
for data in dataset:
    properties = torch.argmax(data.x, dim=1)
    data.properties = properties

    try:
        graph = to_networkx(data, to_undirected=True, node_attrs=["properties", "y"])
    except TypeError:
        pass

    graphs.append(graph)

# formula = AND(
#     Property("GREEN"), Exist(AND(Role("EDGE"), Property("BLACK")), lower=None, upper=1)
# )
formula = OR(Property("RED"), Property("GREEN"))


total_expected = []
total_pred = []

graph_stats = []

for graph in graphs:
    expected = np.array(list(nx.get_node_attributes(graph, "y").values()))

    if use_random:
        actual = np.random.choice([0, 1], size=expected.size, replace=True)
    else:
        actual = FOC(formula)(graph=graph)

    total_expected.extend(expected.tolist())
    total_pred.extend(actual.tolist())

    precision = precision_score(expected, actual)
    recall = recall_score(expected, actual)

    graph_stats.append((precision, recall))

print(graph_stats)

print(classification_report(total_expected, total_pred))

avg_precision = 0.0
avg_recall = 0.0
for pre, rec in graph_stats:
    avg_precision += pre
    avg_recall += rec
avg_precision /= len(graph_stats)
avg_recall /= len(graph_stats)

print("Macro precision:", avg_precision, "\nMacro recall:", avg_recall)
