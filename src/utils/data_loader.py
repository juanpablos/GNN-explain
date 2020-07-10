import networkx as nx
from torch_geometric.data import Data
import torch


def graph_loader(graph,
                 node_labels,
                 n_node_features=2,
                 #  n_node_feature_types=1,
                 #  n_node_labels=2,
                 feature_type="categorical"):

    edges = torch.tensor(list(graph.edges), dtype=torch.long)
    labels = torch.tensor(node_labels)
    features = torch.tensor(
        list(
            nx.get_node_attributes(
                graph,
                "properties").values()), dtype=torch.int64).squeeze()

    if feature_type == "categorical":
        x = torch.nn.functional.one_hot(
            features, n_node_features).type(torch.FloatTensor)
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
