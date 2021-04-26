import os
from typing import List

import torch

from src.data.graph_transform import graph_data_to_graph, stream_transform
from src.graphs import *


def graph_object_stream(
    generator_fn: str,
    min_nodes: int,
    max_nodes: int,
    seed: int = None,
    n_properties: int = 10,
    property_distribution: str = "uniform",
    distribution: List[float] = None,
    verbose: int = 0,
    number_of_graphs: int = None,
    **kwargs
):

    _generator = graph_generator(
        generator_fn=generator_fn,
        min_nodes=min_nodes,
        max_nodes=max_nodes,
        seed=seed,
        **kwargs
    )
    _properties = property_generator(
        graph_generator=_generator,
        number_of_graphs=number_of_graphs,
        n_properties=n_properties,
        property_distribution=property_distribution,
        distribution=distribution,
        seed=seed,
        verbose=verbose,
    )

    return _properties


def graph_data_stream(
    formula: FOC,
    generator_fn: str,
    min_nodes: int,
    max_nodes: int,
    seed: int = None,
    n_properties: int = 10,
    property_distribution: str = "uniform",
    distribution: List[float] = None,
    verbose: int = 0,
    number_of_graphs: int = None,
    **kwargs
):
    """

    Args:
        formula (FOC): the formula used to label each node of the graph
        generator_fn (str): generation fuction of a graph. Only random is accepted as of now.
        min_nodes (int): the minimum amount of node in each graph
        max_nodes (int): the maximum amount of node in each graph
        seed (int, optional): Defaults to None.
        n_properties (int, optional): the number of properties to be assigned to the graph batch. Defaults to 10.
        property_distribution (str, optional): uniform states for each property choosen with an uniform distribution. Otherwise the distribution is in distribution is used. Defaults to "uniform".
        distribution (List[float], optional): the distribution used for each property, must be the same length as the number of properties. Defaults to None.
        verbose (int, optional): 0 is silent, 1 is only 10 messages, 2 is a message for each graph. Defaults to 0.

    Yields:
        torch_geometric.Data: a single data object representing a graph and its labels
    """

    graphs = graph_object_stream(
        generator_fn=generator_fn,
        min_nodes=min_nodes,
        max_nodes=max_nodes,
        seed=seed,
        n_properties=n_properties,
        property_distribution=property_distribution,
        distribution=distribution,
        verbose=verbose,
        number_of_graphs=number_of_graphs,
        **kwargs
    )

    for graph in graphs:
        labels = formula(graph)
        yield stream_transform(
            graph=graph,
            node_labels=labels,
            n_node_features=n_properties,
            feature_type="categorical",
        )


def graph_data_stream_pregenerated_graphs(
    formula: FOC,
    graphs_path: str,
    graphs_filename: str,
    n_properties: int = 10,
    **kwargs
):
    graph_data_path = os.path.join(graphs_path, graphs_filename)
    graphs_data = torch.load(graph_data_path)

    for graph_data in graphs_data:
        graph = graph_data_to_graph(graph_data)
        labels = formula(graph)
        yield stream_transform(
            graph=graph,
            node_labels=labels,
            n_node_features=n_properties,
            feature_type="categorical",
        )
