from graphs import *


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
        yield graph.nodes(data=True), formula(graph)


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

    # a1 = Property("BLUE", "y")
    # a2 = NEG(Role(relation="EDGE", variable1="x", variable2="y"))
    # a3 = AND(a1, a2)
    # a4 = Exist(variable="y", expression=a3, lower=1, upper=1)
    # a5 = AND(a0, a4)
    _formula = FOC(NEG(a0))

    _seed = 11

    stream = graph_stream(formula=_formula,
                          generator_fn="random",
                          min_nodes=3,
                          max_nodes=10,
                          seed=10,
                          n_properties=2,
                          n_property_types=1,
                          property_distribution="uniform",
                          distribution=None,
                          verbose=0,
                          m=2)

    print(next(stream))
    print(next(stream))
    print(next(stream))
    print(next(stream))
