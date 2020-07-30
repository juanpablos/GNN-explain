import random
from functools import partial
from typing import Any, Iterator

import networkx as nx

# TODO: support different types of graphs and configurations


def __generate_random_graph(
        n_nodes: int,
        p: float = None,
        m: int = None,
        seed: Any = None,
        name="erdos",
        **kwargs) -> nx.Graph:
    """
    Args:
        n_nodes (int): number of nodes
        p (float, optional): probability that a new edge is generated between two nodes. Defaults to None.
        m (int, optional): number of edges in the graph. Defaults to None.
        seed (Any, optional): Defaults to None.
        name (str, optional): Use "erdos" or "barabasi" generators. Defaults to "erdos".

    Raises:
        ValueError: when selecting a generator not in [erdos, barabasi]

    Returns:
        nx.Graph: a new graph
    """

    if name == "erdos":
        if m is not None:
            return nx.gnm_random_graph(n=n_nodes, m=n_nodes * m, seed=seed)
        elif p is not None:
            return nx.fast_gnp_random_graph(n=n_nodes, p=p, seed=seed)
        else:
            raise ValueError(
                "`erdos` generator needs either arguments `m` or `p`")

    elif name == "barabasi":
        if m is None:
            raise ValueError("`barabasi` generator must define argument `m`")
        return nx.barabasi_albert_graph(n=n_nodes, m=m, seed=seed)

    else:
        raise ValueError(
            f"Invalid generator value, not `erdos` or `barabasi`: {name}")


def graph_generator(generator_fn: str,
                    min_nodes: int,
                    max_nodes: int,
                    seed: int = None,
                    **kwargs) -> Iterator[nx.Graph]:
    """
    Generator that creates an unlimited amount of random graphs given a generator function

    Args:
        generator_fn (str): generation fuction of a graph. Only random is accepted as of now.
        min_nodes (int): the minimum amount of node in each graph
        max_nodes (int): the maximum amount of node in each graph
        seed (int, optional): Defaults to None.

    Raises:
        ValueError: the generation function is invalid

    Yields:
        A generated graph
    """

    rand = random.Random(seed)

    fn = None
    if generator_fn == "random":
        fn = partial(__generate_random_graph, seed=rand, **kwargs)
    else:
        raise ValueError("Generator function not supported")

    while True:
        # randomly choose the number of nodes for the graph
        n_nodes = rand.randint(min_nodes, max_nodes)
        yield fn(n_nodes=n_nodes)
