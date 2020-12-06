import matplotlib.pyplot as plt
import networkx as nx

if __name__ == "__main__":
    import os
    import sys
    sys.path.insert(
        1, os.path.dirname(
            os.path.dirname(
                os.path.dirname(
                    os.path.abspath(__file__)))))

from src.data.gnn.convert import convert_gnn, prepare_gnn


def filter_name(layer_name):
    return 'linear' in layer_name


def color_map(node):
    if node['color'] == 0:
        return 'red'
    elif node['color'] == 1:
        return 'blue'
    elif node['color'] == 2:
        return 'green'
    elif node['color'] == 3:
        return 'orange'
    elif node['color'] == 4:
        return 'black'
    elif node['color'] == 5:
        return 'grey'
    else:
        return 'yellow'


def draw(graph):
    fig, ax = plt.subplots(1, 1, figsize=(20, 10))
    colors = [color_map(graph.nodes[node]) for node in graph.nodes]
    labels = {node: graph.nodes[node]['value'] for node in graph.nodes}
    edge_labels = {(u, v): attrs['weight']
                   for u, v, attrs in graph.edges(data=True)}

    pos = nx.multipartite_layout(graph, subset_key='subset')  # type: ignore
    nx.draw(graph, pos, with_labels=True, node_color=colors,
            labels=labels, ax=ax)
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, ax=ax)
    plt.show()


def draw_gnn(gnn):
    gnn_layers = prepare_gnn(gnn)
    graph, *_ = convert_gnn(gnn_layers, draw=True)

    for node in graph.nodes:
        graph.nodes[node]['value'] = round(graph.nodes[node]['value'], 3)
    for edge in graph.edges:
        graph.edges[edge]['weight'] = round(graph.edges[edge]['weight'], 3)

    draw(graph)


if __name__ == "__main__":
    from src.models.ac_gnn import ACGNN

    draw_gnn(ACGNN(
        input_dim=6,
        hidden_dim=4,
        output_dim=3,
        aggregate_type='add',
        combine_type='identity',
        num_layers=3,
        combine_layers=-1,
        mlp_layers=3,
        task='node'))
