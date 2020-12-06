import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.utils import from_networkx

if __name__ == "__main__":
    import os
    import sys
    sys.path.insert(
        1, os.path.dirname(
            os.path.dirname(
                os.path.dirname(
                    os.path.abspath(__file__)))))

from src.models.ac_gnn import ACGNN


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


def name_convert(layer_name):
    name = layer_name.split('.')
    if 'linear_prediction' in name:
        return tuple(name)
    return tuple(name[1:3] + name[4:])


def group_layers(layers):
    grouped = {}
    for layer_name, value in layers.items():
        if 'linear_prediction' in layer_name:
            grouped.setdefault('output', {})[layer_name[1]] = value
        else:
            gnn_layer, mlp, mlp_layer, param_type = layer_name
            grouped.setdefault(
                gnn_layer,
                {}).setdefault(
                mlp,
                {}).setdefault(
                mlp_layer,
                {})[param_type] = value

    return grouped


def draw(graph):
    fig, ax = plt.subplots(1, 1, figsize=(20, 10))
    colors = [color_map(graph.nodes[node]) for node in graph.nodes]
    labels = {node: graph.nodes[node]['value'] for node in graph.nodes}
    edge_labels = {(u, v): attrs['weight']
                   for u, v, attrs in graph.edges(data=True)}

    pos = nx.multipartite_layout(graph, subset_key='type')  # type: ignore
    nx.draw(graph, pos, with_labels=True, node_color=colors,
            labels=labels, ax=ax)
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, ax=ax)
    plt.show()


###########################


def convert_mlp(
        state_dict,
        graph=None,
        last_output=None,
        node_counter=0,
        pos=0,
        node_type=0):

    weights = [layer_values['weight']
               for _, layer_values in state_dict.items()]
    biases = [layer_values['bias']
              for _, layer_values in state_dict.items()]

    if graph is None:
        graph = nx.DiGraph()

    if last_output is None:
        input_nodes = []
        for _ in range(weights[0].size(1)):
            graph.add_node(node_counter, type=pos, value=0., color=2)
            input_nodes.append(node_counter)
            node_counter += 1
    else:
        input_nodes = last_output

    last = input_nodes

    max_pos = pos

    for l in range(len(weights)):
        current = []
        for i in range(weights[l].size(0)):
            local_pos = pos + l + 1
            max_pos = max(max_pos, local_pos)

            graph.add_node(node_counter,
                           type=local_pos,
                           value=round(biases[l][i].item(), 3),
                           color=node_type)
            current.append(node_counter)

            ws = weights[l][i]
            for j in range(ws.size(0)):
                graph.add_edge(
                    last[j],
                    node_counter,
                    weight=round(ws[j].item(), 3)
                )

            node_counter += 1

        last = current

    return graph, input_nodes, last, max_pos


def convert_conv(state_dict, graph=None, last_output=None, pos=0):
    if graph is None:
        graph = nx.DiGraph()

    input_nodes_all = []
    output_nodes_all = []

    max_pos = pos + 1

    for layer_name, layer_values in state_dict.items():
        mlp_graph, input_nodes, output_nodes, local_pos = convert_mlp(
            layer_values,
            graph=None,
            node_counter=len(graph),
            pos=pos + 1,
            node_type=10 if layer_name == 'V' else 4)

        max_pos = max(max_pos, local_pos)

        input_nodes_all.append(input_nodes)
        output_nodes_all.append(output_nodes)
        graph = nx.disjoint_union(graph, mlp_graph)

    if last_output is None:
        first_nodes = [i + len(graph) for i in range(len(input_nodes_all[0]))]
        graph.add_nodes_from(first_nodes, type=pos, value=0., color=0)
    else:
        first_nodes = last_output

    for n, *ns in zip(first_nodes, *input_nodes_all):
        for _nx in ns:
            graph.add_edge(n, _nx, weight=1.)

    output_nodes = [i + len(graph) for i in range(len(output_nodes_all[0]))]

    graph.add_nodes_from(output_nodes, type=max_pos + 1, value=0., color=1)
    for n, *ns in zip(output_nodes, *output_nodes_all):
        for _nx in ns:
            graph.add_edge(_nx, n, weight=1.)

    return graph, first_nodes, output_nodes, max_pos + 1


def convert_gnn(state_dict, pos=0):
    graph = nx.DiGraph()

    max_pos = pos + 1

    first_nodes = None
    last_nodes = None
    last_output = None

    for layer_name, layer in state_dict.items():
        if layer_name == 'output':
            graph, input_nodes, output_nodes, local_pos = convert_mlp(
                {'0': layer},
                graph=graph,
                last_output=last_output,
                node_type=5,
                pos=max_pos,
                node_counter=len(graph))

            output = [n + len(graph) for n in range(len(output_nodes))]
            graph.add_nodes_from(output, type=local_pos + 1, value=0., color=3)
            for n1, n2 in zip(output_nodes, output):
                graph.add_edge(n1, n2, weight=1.)

            last_nodes = output

        else:
            graph, input_nodes, output_nodes, local_pos = convert_conv(
                layer,
                graph=graph,
                last_output=last_output,
                pos=max_pos)

            if first_nodes is None:
                first_nodes = input_nodes
            last_output = output_nodes

        max_pos = max(max_pos, local_pos)

    return graph, first_nodes, last_nodes, max_pos


def draw_gnn(gnn):
    gnn_layers = {name_convert(name): value for name,
                  value in gnn.items() if filter_name(name)}
    gnn_layers = group_layers(gnn_layers)

    graph, *_ = convert_gnn(gnn_layers)
    draw(graph)

    data = from_networkx(graph)


if __name__ == "__main__":
    draw_gnn(ACGNN(
        input_dim=6,
        hidden_dim=4,
        output_dim=3,
        aggregate_type='add',
        combine_type='identity',
        num_layers=3,
        combine_layers=-1,
        mlp_layers=3,
        task='node').state_dict())
