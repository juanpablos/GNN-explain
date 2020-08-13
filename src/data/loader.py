from __future__ import annotations

import logging
import os
from typing import Dict, Generic, Iterable, List, Type, Union

from src.typing import S, S_co

from .datasets import LabeledDataset, NetworkDataset

logger = logging.getLogger(__name__)


class FormulaConfig(Generic[S_co]):
    def __init__(
            self,
            formula_hash: str,
            formula_label: S_co,
            limit: int = None):
        self.formula = formula_hash
        self.label = formula_label
        self.limit = limit

    def get_content(self):
        return self.formula, self.label, self.limit

    def __eq__(self, other: Union[str, FormulaConfig]):
        if isinstance(other, str):
            return self.formula == other
        elif isinstance(other, FormulaConfig):
            return self.formula == other.formula
        else:
            return False

    def __hash__(self):
        return hash(self.formula)

    @classmethod
    def from_hashes(cls: Type[FormulaConfig[int]], hashes: Iterable[str]):
        configs: List[FormulaConfig[int]] = []
        for l, h in enumerate(hashes):
            configs.append(cls(h, formula_label=l))

        return configs


def load_gnn_files(root: str, model_hash: str,
                   formulas: Iterable[FormulaConfig[S]],
                   load_all: bool):

    def _prepare_files(path: str):
        files: Dict[str, str] = {}
        # reproducibility, always sorted files
        for file in sorted(os.listdir(path)):
            if file.endswith(".pt"):
                _hash = file.split(".")[0].split("-")[-1]
                files[_hash] = file
        return files

    if model_hash not in os.listdir(root):
        raise FileExistsError(
            f"No directory for the current model hash: {root}")

    model_path = os.path.join(root, model_hash)
    dir_formulas = _prepare_files(model_path)

    if load_all:
        formula_configs = FormulaConfig.from_hashes(dir_formulas.keys())
    else:
        if not all(f in dir_formulas for f in formulas):
            _not = [f for f in formulas if f not in dir_formulas]
            raise ValueError(
                "Not all requested formula hashes are present "
                f"in the directory: {_not}")

        formula_configs = formulas

    mapping: Dict[str, int] = {}
    datasets: List[NetworkDataset[int]] = []
    for configs in formula_configs:
        formula_hash, label, limit = configs.get_content()
        logger.info(f"\tLoading {formula_hash}")

        file_path = os.path.join(model_path, dir_formulas[formula_hash])
        dataset = NetworkDataset(
            file=file_path,
            label=label,
            limit=limit)

        datasets.append(dataset)
        mapping[formula_hash] = label

    return LabeledDataset.from_iterable(datasets), mapping
