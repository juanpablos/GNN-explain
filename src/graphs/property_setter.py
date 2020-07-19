from itertools import repeat
from typing import Generator, Iterable, List

import networkx as nx
import numpy as np


# TODO: in the case of mutiple types of properties
# TODO: what to do when there can be more than one property per node
def property_generator(graph_generator: Iterable[nx.Graph],
                       number_of_graphs: int = None,
                       n_properties: int = 10,
                       n_property_types: int = 1,
                       property_distribution: str = "uniform",
                       distribution: List[float] = None,
                       seed: int = None,
                       verbose: int = 0) -> Iterable[nx.Graph]:
    """Assign properties to each graph generated by the argument graph_generator

    Args:
        graph_generator (Generator[nx.Graph, None, None]): the graph generator
        number_of_graphs (int, optional): the number of graphs required. If None generate an infinite number of graphs.
        n_properties (int, optional): the number of properties to be assigned to the graph batch. Defaults to 10.
        n_property_types (int, optional): number of different entries in the property attribute.
        property_distribution (str, optional): uniform states for each property choosen with an uniform distribution. Otherwise the distribution is in distribution is used. Defaults to "uniform".
        distribution (List[float], optional): the distribution used for each property, must be the same length as the number of properties. Defaults to None.
        seed (int, optional): Defaults to None.
        verbose (int, optional): 0 is silent, 1 is only 10 messages, 2 is a message for each graph. Defaults to 0.

    Yields:
        the graph generated by the graph generator plus the assigned properties
    """

    if property_distribution == "uniform":
        distribution = None
    else:
        # in the case the properties are not uniformly distributed
        assert distribution, "distribution cannot be empty nor None"
        assert len(
            distribution) == n_properties, "distribution has to have the same number of elements as n_properties"
        assert 0 <= 1 - \
            sum(distribution) < 1e5, "probabilities do not sum to 1"

    if n_property_types != 1:
        raise NotImplementedError

    # ! representation of properties as intergers
    # TODO: check if we need to change this
    properties = list(range(n_properties))

    rand = np.random.default_rng(seed)

    if number_of_graphs is None:
        times = repeat(None)
        verbose = 0
    else:
        times = range(number_of_graphs)

    for it in times:
        # get the next graph
        graph = next(graph_generator)

        graph_properties = rand.choice(
            properties,
            size=(len(graph), n_property_types),
            replace=True,
            p=distribution)

        nx.set_node_attributes(graph, dict(
            zip(graph, graph_properties)), name="properties")

        if verbose == 1:
            if it % (number_of_graphs // 10) == 0:
                print(f"{it + 1}/{number_of_graphs} graphs colored")
        elif verbose == 2:
            print(f"{it + 1}/{number_of_graphs} graphs colored")

        yield graph