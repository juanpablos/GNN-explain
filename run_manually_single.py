import networkx as nx
import numpy as np
import torch
from sklearn.metrics import classification_report
from torch_geometric.utils import to_networkx

from src.graphs.foc import *

cora = torch.load("./data/gnns/f4034364ea-batch/reduced_cora.pt")
properties = torch.argmax(cora.x, dim=1)
cora.properties = properties

graph = to_networkx(cora, to_undirected=True, node_attrs=["properties", "y"])

# print(graph.nodes(data=True))

formula = AND(
    Property("GREEN"), Exist(AND(Role("EDGE"), Property("BLUE")), lower=None, upper=2)
)

expected = np.array(list(nx.get_node_attributes(graph, "y").values()))
actual = FOC(formula)(graph=graph)

print(classification_report(expected, actual))
