import os
import random

import networkx as nx
import numpy as np
import torch
import torch.nn.functional
from torch_geometric.data import Data, InMemoryDataset

from src.data.auxiliary import GNNGraphDataset
from src.generate_graphs import graph_object_stream
from src.graphs.foc import *

rand = random.Random(42)


save_path = os.path.join("data", "graphs")
os.makedirs(save_path, exist_ok=True)
filename = "test_graphs"


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
for a in range(0, 101, 5):
    for b in range(0, 101, 5):
        for c in range(0, 101, 5):
            for d in range(0, 101, 5):
                if a + b + c + d != 100:
                    continue
                color_tuples.append((a / 100.0, b / 100.0, c / 100.0, d / 100.0))

data_config = {
    "generator_fn": "random",
    "min_nodes": 80,  # 20 80
    "max_nodes": 120,  # 80 120
    # "seed": 0,
    "n_properties": 4,
    "property_distribution": "manual",
    # "distribution": [v / 100.0 for v in tuples[0]],
    "verbose": 0,
    "number_of_graphs": 1,
    # --- generator config
    "name": "erdos",
    # ! because the generated graph is undirected,
    # ! the number of average neighbors will be double `m`
    # "m": 4,
}

total = len(color_tuples)
graph_data = []


# train: 50 graphs, m=[3, 4], color distributions 0->100/5, 20/80
# test: 1 graph, m=[2, 4, 6], color distributions 0->100/5, 80/120


for m in [2, 4, 6]:
    for i, color_tuple in enumerate(color_tuples):
        print(f"m={m}, {i + 1}/{total}", color_tuple)
        graph_generator = graph_object_stream(
            **data_config, distribution=color_tuple, m=m, seed=rand.randint(1, 1 << 30)
        )

        graph_data.extend(graph_to_data(g) for g in graph_generator)

np.random.shuffle(graph_data)
data, slices = InMemoryDataset.collate(graph_data)
dataset = GNNGraphDataset(data=data, slices=slices)

torch.save(dataset, os.path.join(save_path, f"{filename}_{len(dataset)}.pt"))
print(dataset)
