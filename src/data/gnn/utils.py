import os
from collections import defaultdict
from typing import Any, Dict, Generator, List, Tuple

import torch
from torch_geometric.data import InMemoryDataset

try:
    from .convert import gnn2graph, gnn2tensordict
except ImportError:
    from convert import gnn2graph, gnn2tensordict  # type: ignore


def prepare_files(path: str):
    files: Dict[str, str] = {}
    # reproducibility, always sorted files
    for file in sorted(os.listdir(path)):
        if file.endswith(".pt"):
            _hash = file.split(".")[0].split("-")[-1]
            files[_hash] = file
    return files


def clean_state(model_dict):
    """Removes the weights associated with batchnorm"""
    return {k: v for k, v in model_dict.items() if "batch" not in k}


def stack_dict_tensors(tensor_dict_list):
    temp_dict = defaultdict(list)
    for tensor_dict in tensor_dict_list:
        for k, tensor in tensor_dict.items():
            temp_dict[k].append(tensor)

    aggregator_dict: Dict[str, torch.Tensor] = {}
    for k, tensors in temp_dict.items():
        aggregator_dict[k] = torch.stack(tensors)

    return aggregator_dict


def network_loader_generator(
        root: str,
        model_hash: str,
        filename: str) -> Generator[Tuple[List[torch.Tensor], str], Any, None]:

    if model_hash not in os.listdir(root):
        raise FileExistsError(
            f"No directory for the current model hash: {root}/{model_hash}")

    model_path = os.path.join(root, model_hash)
    available_formulas = prepare_files(model_path)

    big_dataset = {}

    for formula_hash, formula_file in available_formulas.items():
        file_path = os.path.join(model_path, formula_file)
        print(f"Loading formula {formula_hash}")

        yield (torch.load(file_path), formula_hash)
        data = yield  # type: ignore

        big_dataset[formula_hash] = {
            "file": formula_file,
            "data": data
        }

    save_file = os.path.join(model_path, "processed", filename)
    print(f"Saving whole dataset")
    torch.save(big_dataset, save_file)


def aggregate_formulas(
        root: str,
        model_hash: str,
        filename: str,
        nobatch: bool):

    network_loader = network_loader_generator(
        root=root,
        model_hash=model_hash,
        filename=filename)
    network_loader.send(None)

    try:
        for networks, formula_hash in network_loader:
            dataset = []
            for network in networks:
                # legacy
                if nobatch:
                    network = clean_state(network)
                # /legacy

                concat_weights = torch.cat(
                    [w.flatten() for w in network.values()])
                dataset.append(concat_weights)

            print(f"Stacking formula {formula_hash}")
            dataset = torch.stack(dataset)

            network_loader.send(dataset)
    except StopIteration:
        print("Finished processing")


def stack_gnn_graphs(
        root: str,
        model_hash: str,
        filename: str,
        nobatch: bool,
        as_undirected: bool = False):

    network_loader = network_loader_generator(
        root=root,
        model_hash=model_hash,
        filename=filename)
    network_loader.send(None)

    try:
        for networks, formula_hash in network_loader:
            data_list = []
            for network in networks:
                # legacy
                if nobatch:
                    network = clean_state(network)
                # /legacy

                data_list.append(gnn2graph(network, undirected=as_undirected))

            print(f"Collating formula/network {formula_hash}")
            dataset, slices = InMemoryDataset.collate(data_list)

            network_loader.send((dataset, slices))
    except StopIteration:
        print("Finished processing")


def tensor_dict_gnn(
        root: str,
        model_hash: str,
        filename: str,
        nobatch: bool):

    network_loader = network_loader_generator(
        root=root,
        model_hash=model_hash,
        filename=filename)
    network_loader.send(None)

    try:
        for networks, formula_hash in network_loader:
            data_list = []
            for network in networks:
                # legacy
                if nobatch:
                    network = clean_state(network)
                # /legacy

                data_list.append(gnn2tensordict(network))

            print(f"Stacking tensors for dict keys {formula_hash}")
            tensor_dict = stack_dict_tensors(data_list)

            network_loader.send(tensor_dict)
    except StopIteration:
        print("Finished processing")


if __name__ == "__main__":
    # aggregate_formulas(
    #     root="data/gnns",
    #     model_hash="f4034364ea-batch",
    #     filename="aggregated.pt",
    #     nobatch=True
    # )

    # stack_gnn_graphs(
    #     root="../../../data/gnns",
    #     model_hash="f4034364ea-batch",
    #     filename="graph_gnns_undirected.pt",
    #     nobatch=True,
    #     as_undirected=True
    # )

    tensor_dict_gnn(
        root="../../../data/gnns",
        model_hash="f4034364ea-batch",
        filename="gnns_tensor_dict.pt",
        nobatch=True,
    )
