import networkx as nx
import torch
from torch_geometric.utils import from_networkx


def filter_name(layer_name):
    return 'linear' in layer_name


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


###########################


def convert_mlp(
        state_dict,
        graph=None,
        last_output=None,
        node_counter=0,
        # only if draw is True
        draw=False,
        pos=0,
        node_type=0):

    weights = [layer_values['weight']
               for _, layer_values in state_dict.items()]
    biases = [layer_values['bias']
              for _, layer_values in state_dict.items()]

    if graph is None:
        graph = nx.DiGraph()

    # if last_output is passed, use it as input, otherwise create new inputs
    if last_output is None:
        input_nodes = []
        # generate new nodes
        for _ in range(weights[0].size(1)):
            graph.add_node(node_counter, x=0.)

            if draw:
                graph.nodes[node_counter].update({'subset': pos, 'color': 2})

            input_nodes.append(node_counter)
            node_counter += 1
    else:
        input_nodes = last_output

    # use the input nodes as base to append new layers
    last = input_nodes

    # position to plot if needed
    max_pos = pos

    # for each layer
    for l in range(len(weights)):
        current = []
        for i in range(weights[l].size(0)):
            local_pos = pos + l + 1
            max_pos = max(max_pos, local_pos)

            graph.add_node(node_counter, x=biases[l][i].item())

            if draw:
                graph.nodes[node_counter].update(
                    {'subset': local_pos, 'color': node_type})

            current.append(node_counter)

            ws = weights[l][i]
            for j in range(ws.size(0)):
                graph.add_edge(
                    last[j],
                    node_counter,
                    weight=ws[j].item()
                )

            node_counter += 1

        last = current

    return graph, input_nodes, last, max_pos


def convert_conv(
        state_dict,
        graph=None,
        last_output=None,
        # only if draw is True
        draw=False,
        pos=0):
    if graph is None:
        graph = nx.DiGraph()

    # input nodes for each MLP
    input_nodes_all = []
    # output nodes for each MLP
    output_nodes_all = []

    max_pos = pos + 1

    for layer_name, layer_values in state_dict.items():
        mlp_graph, input_nodes, output_nodes, local_pos = convert_mlp(
            layer_values,
            graph=None,
            node_counter=len(graph),
            draw=draw,
            pos=pos + 1,
            node_type=10 if layer_name == 'V' else 4)

        max_pos = max(max_pos, local_pos)

        # store the input and output nodes
        input_nodes_all.append(input_nodes)
        output_nodes_all.append(output_nodes)

        # merge the MLP graph to the main graph
        graph = nx.disjoint_union(graph, mlp_graph)

    # create new nodes only if no previous output
    if last_output is None:
        first_nodes = [i + len(graph) for i in range(len(input_nodes_all[0]))]
        graph.add_nodes_from(first_nodes, x=0.)

        if draw:
            for n in first_nodes:
                graph.nodes[n].update({'subset': pos, 'color': 0})
    else:
        first_nodes = last_output

    # link base input nodes to MLP input nodes
    for n, *ns in zip(first_nodes, *input_nodes_all):
        for _nx in ns:
            graph.add_edge(n, _nx, weight=1.)

    # generate output nodes
    output_nodes = [i + len(graph) for i in range(len(output_nodes_all[0]))]
    graph.add_nodes_from(output_nodes, x=0.)

    if draw:
        for n in output_nodes:
            graph.nodes[n].update({'subset': max_pos + 1, 'color': 1})

    # link MLP output nodes to network output nodes
    for n, *ns in zip(output_nodes, *output_nodes_all):
        for _nx in ns:
            graph.add_edge(_nx, n, weight=1.)

    return graph, first_nodes, output_nodes, max_pos + 1


def convert_gnn(
        state_dict,
        # only if draw is True
        draw=False,
        pos=0):
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
                node_counter=len(graph),
                draw=draw,
                pos=max_pos,
                node_type=5)

            output = [n + len(graph) for n in range(len(output_nodes))]
            graph.add_nodes_from(output, x=0.)

            if draw:
                for n in output:
                    graph.nodes[n].update(
                        {'subset': local_pos + 1, 'color': 3})

            for n1, n2 in zip(output_nodes, output):
                graph.add_edge(n1, n2, weight=1.)

            last_nodes = output

        else:
            graph, input_nodes, output_nodes, local_pos = convert_conv(
                layer,
                graph=graph,
                last_output=last_output,
                draw=draw,
                pos=max_pos)

            if first_nodes is None:
                first_nodes = input_nodes
            last_output = output_nodes

        max_pos = max(max_pos, local_pos)

    return graph, first_nodes, last_nodes, max_pos


def prepare_gnn(gnn):
    if isinstance(gnn, torch.nn.Module):
        gnn = gnn.state_dict()

    gnn_layers = {name_convert(name): value for name,
                  value in gnn.items() if filter_name(name)}
    gnn_layers = group_layers(gnn_layers)
    return gnn_layers


def gnn2data(gnn):
    gnn_layers = prepare_gnn(gnn)
    graph, *_ = convert_gnn(gnn_layers, draw=False)
    return from_networkx(graph)


if __name__ == "__main__":
    import os
    import sys
    sys.path.insert(
        1,
        os.path.dirname(
            os.path.dirname(
                os.path.dirname(
                    os.path.dirname(
                        os.path.abspath(__file__))))))
    from torch_geometric.data import DataLoader

    from src.models import ACGNN

    d1 = gnn2data(ACGNN(
        input_dim=6,
        hidden_dim=4,
        output_dim=3,
        aggregate_type='add',
        combine_type='identity',
        num_layers=3,
        combine_layers=-1,
        mlp_layers=3,
        task='node'))

    dd = [d1, d1]
    ddd = DataLoader(dd, batch_size=2)

    print(next(iter(ddd)))
