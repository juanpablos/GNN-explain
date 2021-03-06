from typing import Any, List

import networkx as nx
import torch
import torch.nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx


def stream_transform(
    graph: nx.Graph,
    node_labels: List[List[Any]],
    n_node_features: int,
    feature_type: str = "categorical",
) -> Data:
    """
    Generates a stream of torch_geometric:Data objects containing the generated graph and the label associated with each node.

    Args:
        graph (nx.Graph): the graph to work on
        node_labels (List[List[Any]]): the list of labels for each node
        n_node_features (int, optional): the number of features available.
        feature_type (str, optional): whether the features are categorical or numerical. Defaults to "categorical".

    Returns:
        torch_geometric.Data: a Data object representing a single graph to train on.
    """

    graph = graph.to_directed()
    edges = torch.tensor(list(graph.edges), dtype=torch.long)
    labels = torch.tensor(node_labels, dtype=torch.long)
    features = torch.tensor(
        list(nx.get_node_attributes(graph, "properties").values()), dtype=torch.long
    )

    if feature_type == "categorical":
        x = F.one_hot(features, n_node_features).float()
    else:
        x = features

    return Data(x=x, edge_index=edges.t().contiguous(), y=labels)


def graph_data_to_labeled_data(
    graph_data: Data,
    node_labels: List[List[Any]],
    n_node_features: int,
    feature_type: str = "categorical",
) -> Data:
    features = graph_data.properties.long()
    if feature_type == "categorical":
        x = F.one_hot(features, n_node_features).float()
    else:
        x = features
    labels = torch.tensor(node_labels, dtype=torch.long)

    return Data(x=x, edge_index=graph_data.edge_index, y=labels)


def graph_data_to_graph(graph_data: Data) -> nx.Graph:
    return to_networkx(graph_data, node_attrs=["properties"], to_undirected=True)


def graph_labeled_data_to_graph(graph_data: Data) -> nx.Graph:
    G = nx.Graph()
    G.add_nodes_from(range(graph_data.num_nodes))

    for u, v in graph_data.edge_index.t().tolist():
        if v > u:
            continue
        G.add_edge(u, v)

    properties = graph_data["x"].int().argmax(1).tolist()
    nx.set_node_attributes(G, dict(zip(G, properties)), name="properties")

    return G
