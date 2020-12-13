import os
from typing import Dict

import torch
from torch_geometric.data import InMemoryDataset

try:
    from .convert import gnn2data
except ImportError:
    from convert import gnn2data  # type: ignore


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


def aggregate_formulas(
        root: str,
        model_hash: str,
        filename: str,
        nobatch: bool):

    if model_hash not in os.listdir(root):
        raise FileExistsError(
            f"No directory for the current model hash: {root}/{model_hash}")

    model_path = os.path.join(root, model_hash)
    available_formulas = prepare_files(model_path)

    big_dataset = {}

    for formula_hash, formula_file in available_formulas.items():
        file_path = os.path.join(model_path, formula_file)
        print(f"Loading formula {formula_hash}")
        networks = torch.load(file_path)

        dataset = []
        for weights in networks:
            # legacy
            if nobatch:
                weights = clean_state(weights)
            # /legacy

            concat_weights = torch.cat([w.flatten() for w in weights.values()])
            dataset.append(concat_weights)

        print(f"Stacking formula {formula_hash}")
        dataset = torch.stack(dataset)

        big_dataset[formula_hash] = {
            "file": formula_file,
            "data": dataset
        }

    save_file = os.path.join(model_path, "processed", filename)
    print(f"Saving whole dataset")
    torch.save(big_dataset, save_file)


def stack_gnn_graphs(
        root: str,
        model_hash: str,
        filename: str,
        nobatch: bool,
        as_undirected: bool = False):

    if model_hash not in os.listdir(root):
        raise FileExistsError(
            f"No directory for the current model hash: {root}/{model_hash}")

    model_path = os.path.join(root, model_hash)
    available_formulas = prepare_files(model_path)

    big_dataset = {}

    for formula_hash, formula_file in available_formulas.items():
        file_path = os.path.join(model_path, formula_file)
        print(f"Loading formula {formula_hash}")
        networks = torch.load(file_path)

        data_list = []
        for network in networks:
            # legacy
            if nobatch:
                network = clean_state(network)
            # /legacy

            data_list.append(gnn2data(network, undirected=as_undirected))

        print(f"Collating formula/network {formula_hash}")

        dataset, slices = InMemoryDataset.collate(data_list)

        big_dataset[formula_hash] = {
            "file": formula_file,
            "data": (dataset, slices)
        }

    save_file = os.path.join(model_path, "processed", filename)
    print(f"Saving whole dataset")
    torch.save(big_dataset, save_file)


if __name__ == "__main__":
    # aggregate_formulas(
    #     root="data/gnns",
    #     model_hash="f4034364ea-batch",
    #     filename="aggregated.pt",
    #     nobatch=True
    # )

    stack_gnn_graphs(
        root="../../../data/gnns",
        model_hash="f4034364ea-batch",
        filename="graph_gnns_undirected.pt",
        nobatch=True,
        as_undirected=True
    )
