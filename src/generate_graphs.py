import torch

from graphs import *
from utils.data_loader import graph_loader

# from timeit import default_timer as timer
# temp = 0


def graph_stream(formula,
                 generator_fn,
                 min_nodes,
                 max_nodes,
                 seed,
                 n_properties,
                 n_property_types,
                 property_distribution,
                 distribution,
                 verbose,
                 **kwargs):

    # TODO: generate directed graphs
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
        verbose=0)

    for graph in _properties:
        # t = timer()
        labels = formula(graph)
        # global temp
        # temp += timer() - t
        yield graph_loader(graph=graph,
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
                          seed=10,
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

    class Iterable(torch.utils.data.IterableDataset):
        def __init__(self, iterable):
            self.data = iterable

        def __iter__(self):
            return self.data

    from torch_geometric.data import DataLoader
    import time

    d = DataLoader(Iterable(stream), batch_size=512, num_workers=0)

    a = time.time()
    for i, data in enumerate(d):
        print(time.time() - a)
        a = time.time()
