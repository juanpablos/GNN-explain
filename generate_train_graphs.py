import networkx as nx
import torch
import torch.nn.functional
from torch_geometric.data import Data, InMemoryDataset

from src.data.auxiliary import GNNGraphDataset
from src.generate_graphs import graph_object_stream
from src.graphs.foc import *
import numpy as np
import os


save_path = os.path.join("data", "graphs")
os.makedirs(save_path, exist_ok=True)
filename = "test_graphs.pt"


def graph_to_data(graph):
    graph = graph.to_directed()
    edges = torch.tensor(list(graph.edges), dtype=torch.long)
    features = torch.tensor(
        list(nx.get_node_attributes(graph, "properties").values()), dtype=torch.long
    )

    _data = Data(properties=features, edge_index=edges.t().contiguous())
    _data.num_nodes = len(graph)
    return _data


color_tuples = []
for a in range(5, 81, 5):
    for b in range(5, 81, 5):
        for c in range(5, 81, 5):
            for d in range(5, 81, 5):
                if a + b + c + d != 100:
                    continue
                color_tuples.append((a / 100.0, b / 100.0, c / 100.0, d / 100.0))

data_config = {
    "generator_fn": "random",
    "min_nodes": 80,  # 20
    "max_nodes": 120,  # 80
    "seed": 0,
    "n_properties": 4,
    "property_distribution": "manual",
    # "distribution": [v / 100.0 for v in tuples[0]],
    "verbose": 0,
    "number_of_graphs": 1,
    # --- generator config
    "name": "erdos",
    # ! because the generated graph is undirected,
    # ! the number of average neighbors will be double `m`
    "m": 4,
}

total = len(color_tuples)
graph_data = []
for i, color_tuple in enumerate(color_tuples):
    print(f"{i + 1}/{total}", color_tuple)
    graph_generator = graph_object_stream(**data_config, distribution=color_tuple)

    graph_data.extend(graph_to_data(g) for g in graph_generator)

np.random.shuffle(graph_data)
data, slices = InMemoryDataset.collate(graph_data)
dataset = GNNGraphDataset(data=data, slices=slices)

torch.save(dataset, os.path.join(save_path, filename))
print(dataset)
