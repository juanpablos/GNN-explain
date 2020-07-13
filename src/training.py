import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import DataLoader
from torch_scatter import scatter_mean


def __loss_aux(output, loss, data, binary):
    if binary:
        labels = torch.zeros_like(output).scatter_(dim=1,
                                                   index=data.y.unsqueeze(1),
                                                   value=1.)
    else:
        # TODO: missing option when not just predicting 0/1
        raise NotImplementedError()

    return loss(output, labels)


def __accuracy_aux(node_labels, predicted_labels, batch, device):
    results = torch.eq(
        predicted_labels,
        node_labels).float().to(device)

    micro = torch.sum(results)
    macro = torch.sum(scatter_mean(results, batch))

    return micro, macro


def train(
        model: nn.Module,
        training_data: DataLoader,
        criterion,
        device,
        optimizer,
        scheduler,
        binary_prediction: bool) -> float:

    #!########
    model.train()
    #!########

    accum_loss = []

    for data in training_data:
        data = data.to(device)

        # TODO: for a more generic application, include the edge features
        output = model(x=data.x,
                       edge_index=data.edge_index,
                       batch=data.batch)

        loss = __loss_aux(
            output=output,
            loss=criterion,
            data=data,
            binary=binary_prediction
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        accum_loss.append(loss.detach().cpu().numpy())

    average_loss = np.mean(accum_loss)

    return average_loss


def evaluate(
        model: nn.Module,
        test_data: DataLoader,
        criterion,
        device,
        binary_prediction: bool = True):

    #!########
    model.eval()
    #!########

    micro_avg = 0.
    macro_avg = 0.
    n_nodes = 0
    n_graphs = 0
    accum_loss = []

    for data in test_data:
        data = data.to(device)

        with torch.no_grad():
            output = model(
                x=data.x,
                edge_index=data.edge_index,
                batch=data.batch
            )

        loss = __loss_aux(
            output=output,
            loss=criterion,
            data=data,
            binary=binary_prediction
        )

        accum_loss.append(loss.detach().cpu().numpy())

        output = torch.sigmoid(output)
        _, predicted_labels = output.max(dim=1)

        micro, macro = __accuracy_aux(
            node_labels=data.y,
            predicted_labels=predicted_labels,
            batch=data.batch,
            device=device
        )

        micro_avg += micro.cpu().numpy()
        macro_avg += macro.cpu().numpy()

        n_nodes += data.num_nodes
        n_graphs += data.num_graphs

    average_loss = np.mean(accum_loss)
    micro_avg = micro_avg / n_nodes
    macro_avg = macro_avg / n_graphs

    return average_loss, micro_avg, macro_avg
