import os
import random
from collections import defaultdict

import networkx as nx
import torch
import torch.nn.functional
from torch_geometric.data import Data, InMemoryDataset

from src.data.auxiliary import GNNGraphDataset
from src.generate_graphs import graph_object_stream
from src.graphs.foc import *

rand = random.Random(42)


save_path = os.path.join("data", "graphs")
os.makedirs(save_path, exist_ok=True)


def graph_to_data(graph):
    graph = graph.to_directed()
    edges = torch.tensor(list(graph.edges), dtype=torch.long)
    features = torch.tensor(
        list(nx.get_node_attributes(graph, "properties").values()), dtype=torch.long
    )

    _data = Data(properties=features, edge_index=edges.t().contiguous())
    _data.num_nodes = len(graph)
    return _data


def generate_tuples():
    color_tuples = []
    for a in range(0, 101, 5):
        for b in range(0, 101, 5):
            for c in range(0, 101, 5):
                for d in range(0, 101, 5):
                    if a + b + c + d != 100:
                        continue
                    color_tuples.append((a / 100.0, b / 100.0, c / 100.0, d / 100.0))
    return color_tuples


def generate_test_graphs(filename, color_tuples):
    data_config = {
        "generator_fn": "random",
        "min_nodes": 80,
        "max_nodes": 200,
        "n_properties": 4,
        "property_distribution": "manual",
        "verbose": 0,
        "number_of_graphs": 1,
        # --- generator config
        "name": "erdos",
        # ! because the generated graph is undirected,
        # ! the number of average neighbors will be double `m`
    }

    total = len(color_tuples)
    test_graph_data = []
    for m in [1, 2, 3, 4, 5, 6]:
        for i, color_tuple in enumerate(color_tuples):
            print(f"m={m}, {i + 1}/{total}", color_tuple)
            graph_generator = graph_object_stream(
                **data_config,
                distribution=color_tuple,
                m=m,
                seed=rand.randint(1, 1 << 30),
            )

            test_graph_data.extend(graph_to_data(g) for g in graph_generator)

    data, slices = InMemoryDataset.collate(test_graph_data)
    dataset = GNNGraphDataset(data=data, slices=slices)

    torch.save(dataset, os.path.join(save_path, f"{filename}_{len(dataset)}.pt"))
    print(dataset)


def generate_train_graphs(filename, color_tuples):
    data_config = {
        "generator_fn": "random",
        "min_nodes": 20,
        "max_nodes": 80,
        "n_properties": 4,
        "property_distribution": "manual",
        "verbose": 0,
        "number_of_graphs": 100,
        # --- generator config
        "name": "erdos",
    }

    total = len(color_tuples)
    train_graph_data = defaultdict(list)
    for m in [2, 4]:
        for i, color_tuple in enumerate(color_tuples):
            print(f"m={m}, {i + 1}/{total}", color_tuple)
            graph_generator = graph_object_stream(
                **data_config,
                distribution=color_tuple,
                m=m,
                seed=rand.randint(1, 1 << 30),
            )

            train_graph_data[color_tuple].extend(
                graph_to_data(g) for g in graph_generator
            )

    total_graphs = 0
    train_data = {}
    for color_distribution, train_graphs in train_graph_data.items():
        data, slices = InMemoryDataset.collate(train_graphs)
        dataset = GNNGraphDataset(data=data, slices=slices)

        train_data[color_distribution] = dataset
        total_graphs += len(train_graphs)

    torch.save(train_data, os.path.join(save_path, f"{filename}_{total_graphs}.pt"))
    print("Finished training graphs")


if __name__ == "__main__":
    _color_tuples = generate_tuples()
    # generate_train_graphs(filename="train_graphs_v2", color_tuples=_color_tuples)
    generate_test_graphs(filename="test_graphs_v2", color_tuples=_color_tuples)
