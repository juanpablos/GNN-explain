from typing import List

from .data.graph_transform import stream_transform
from .graphs import *

# from timeit import default_timer as timer
# temp = 0


def graph_stream(formula: FOC,
                 generator_fn: str,
                 min_nodes: int,
                 max_nodes: int,
                 seed: int = None,
                 n_properties: int = 10,
                 n_property_types: int = 1,
                 property_distribution: str = "uniform",
                 distribution: List[float] = None,
                 verbose: int = 0,
                 **kwargs):
    """

    Args:
        formula (FOC): the formula used to label each node of the graph
        generator_fn (str): generation fuction of a graph. Only random is accepted as of now.
        min_nodes (int): the minimum amount of node in each graph
        max_nodes (int): the maximum amount of node in each graph
        seed (int, optional): Defaults to None.
        n_properties (int, optional): the number of properties to be assigned to the graph batch. Defaults to 10.
        n_property_types (int, optional): number of different entries in the property attribute.
        property_distribution (str, optional): uniform states for each property choosen with an uniform distribution. Otherwise the distribution is in distribution is used. Defaults to "uniform".
        distribution (List[float], optional): the distribution used for each property, must be the same length as the number of properties. Defaults to None.
        verbose (int, optional): 0 is silent, 1 is only 10 messages, 2 is a message for each graph. Defaults to 0.

    Yields:
        torch_geometric.Data: a single data object representing a graph and its labels
    """

    _generator = graph_generator(
        generator_fn=generator_fn,
        min_nodes=min_nodes,
        max_nodes=max_nodes,
        seed=seed,
        **kwargs)
    _properties = property_generator(
        graph_generator=_generator,
        number_of_graphs=None,
        n_properties=n_properties,
        n_property_types=n_property_types,
        property_distribution=property_distribution,
        distribution=distribution,
        seed=seed,
        verbose=verbose)

    for graph in _properties:
        # t = timer()
        labels = formula(graph)
        # global temp
        # temp += timer() - t
        yield stream_transform(graph=graph,
                               node_labels=labels,
                               n_node_features=n_properties,
                               feature_type="categorical")


def generate_graphs(formula,
                    generator_fn,
                    n_graphs,
                    min_nodes,
                    max_nodes,
                    seed,
                    n_properties,
                    n_property_types,
                    property_distribution,
                    distribution,
                    verbose,
                    **kwargs):

    _generator = graph_generator(
        generator_fn=generator_fn,
        min_nodes=min_nodes,
        max_nodes=max_nodes,
        seed=seed,
        **kwargs)
    _properties = property_generator(
        graph_generator=_generator,
        number_of_graphs=n_graphs,
        n_properties=n_properties,
        n_property_types=n_property_types,
        property_distribution=property_distribution,
        distribution=distribution,
        seed=seed,
        verbose=0)

    # TODO: offline graph reading
    raise NotImplementedError


if __name__ == "__main__":

    a0 = Property("RED", "x")

    a1 = Property("BLUE", "y")
    a2 = NEG(Role(relation="EDGE", variable1="x", variable2="y"))
    a3 = AND(a1, a2)
    a4 = Exist(variable="y", expression=a3, lower=2, upper=6)
    a5 = AND(a0, a4)
    _formula = FOC(a5)

    _seed = 11

    stream = graph_stream(formula=_formula,
                          generator_fn="random",
                          min_nodes=40,
                          max_nodes=50,
                          seed=_seed,
                          n_properties=5,
                          n_property_types=1,
                          property_distribution="uniform",
                          distribution=None,
                          verbose=0,
                          m=8)

    # aa = []
    # from timeit import default_timer as timer
    # a = timer()
    # for _ in range(1024):
    #     aa.append(next(stream))
    # print("total", timer() - a)
    # print("label", temp)
    # import torch
    # class Iterable(torch.utils.data.IterableDataset):
    #     def __init__(self, iterable):
    #         self.data = iterable

    #     def __iter__(self):
    #         return self.data

    # from torch_geometric.data import DataLoader
    # import time

    # d = DataLoader(Iterable(stream), batch_size=512, num_workers=0)

    # a = time.time()
    # for i, data in enumerate(d):
    #     print(time.time() - a)
    #     a = time.time()

    dt = next(stream)

    print(dt.y)
    print(dt.y.unsqueeze(1))

    print(dt.x.dtype)
    print(dt.edge_index.dtype)
    print(dt.y.dtype)
