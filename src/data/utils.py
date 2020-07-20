import os
from typing import Any, Dict

from torch.utils.data.dataset import ConcatDataset

from .datasets import NetworkDataset


def clean_state(model_dict):
    return {k: v for k, v in model_dict.items() if "batch" not in k}


def load_gnn_files(root: str, model_hash: str,
                   formula_hashes: Dict[str, Dict[str, Any]]):
    """
    formula hashes has the following format
    hash: {
        limit: number,
        label: any
    }
    """

    def _prepare_files(path):
        files = {}
        for file in os.listdir(path):
            if file.endswith(".pt"):
                _hash = file.split(".")[0].split("-")[-1]
                files[_hash] = file
        return files

    if model_hash not in os.listdir(root):
        raise FileExistsError("No directory for the current model hash")

    model_path = os.path.join(root, model_hash)
    dir_formulas = _prepare_files(model_path)

    assert all(
        f in dir_formulas for f in formula_hashes), "Not all formula hashes are present"

    datasets = []
    for formula_hash, config in formula_hashes.items():
        file_path = os.path.join(model_path, dir_formulas[formula_hash])
        dataset = NetworkDataset(file=file_path, **config)

        datasets.append(dataset)

    return ConcatDataset(datasets)
