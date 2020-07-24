from typing import Any, List

import networkx as nx
import torch
from torch_geometric.data import Data


def stream_transform(graph: nx.Graph,
                     node_labels: List[List[Any]],
                     n_node_features: int = 2,
                     #  n_node_labels=2,
                     feature_type: str = "categorical") -> Data:
    """Generates a stream of torch_geometric:Data objects containing the generated graph and the label associated with each node.

    Args:
        graph (nx.Graph): the graph to work on
        node_labels (List[List[Any]]): the list of labels for each node
        n_node_features (int, optional): the number of features available. Choose from 2 by default. Defaults to 2.
        feature_type (str, optional): whether the features are categorical or numerical. Defaults to "categorical".

    Returns:
        torch_geometric.Data: a Data object representing a single graph to train on.
    """

    graph = graph.to_directed()
    edges = torch.tensor(list(graph.edges), dtype=torch.long)
    labels = torch.tensor(node_labels, dtype=torch.long)
    features = torch.tensor(
        list(
            nx.get_node_attributes(
                graph,
                "properties").values()),
        dtype=torch.long)

    if feature_type == "categorical":
        x = torch.nn.functional.one_hot(
            features, n_node_features).float()
    else:
        x = features

    return Data(
        x=x,
        edge_index=edges.t().contiguous(),
        y=labels
    )


def graph_file_loader():

    # TODO
    pass
