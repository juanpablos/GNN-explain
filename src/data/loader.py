import logging
import os
from typing import Dict, List, Union

from src.data.datasets import LabeledDataset, NetworkDataset
from src.data.formula_index import FormulaMapping
from src.data.formulas.filter import FilterApply, SelectFilter
from src.data.formulas.labeler import (
    LabelerApply,
    MultiLabelCategoricalLabeler
)
from src.typing import S, T

logger = logging.getLogger(__name__)


def __prepare_files(path: str):
    files: Dict[str, str] = {}
    # reproducibility, always sorted files
    for file in sorted(os.listdir(path)):
        if file.endswith(".pt"):
            _hash = file.split(".")[0].split("-")[-1]
            files[_hash] = file
    return files


def load_gnn_files(root: str,
                   model_hash: str,
                   selector: Union[SelectFilter, FilterApply],
                   labeler: LabelerApply[T, S],
                   formula_mapping: FormulaMapping,
                   _legacy_load_without_batch: bool = False):

    if model_hash not in os.listdir(root):
        raise FileExistsError(
            f"No directory for the current model hash: {root}")

    if isinstance(labeler, MultiLabelCategoricalLabeler):
        # TODO: implement multi label
        raise NotImplementedError(
            "Multilabel classification is yet to be implemented")

    model_path = os.path.join(root, model_hash)
    # select all formulas available in directory
    # formula_hash -> file_path
    dir_formulas = __prepare_files(model_path)

    logger.debug("Creating formula objects")
    # mapping from formula_hash -> formula object
    dir_mapping = {_hash: formula_mapping[_hash] for _hash in dir_formulas}

    logger.debug(f"Running formula selector {selector}")
    # mapping from the selected formula_hash -> formula object
    selected_formulas = selector(dir_mapping)
    logger.debug(f"Running formula labeler {labeler}")
    # mapping from the selected formula_hash -> label
    # classes is a dictionary class_id -> class_str
    selected_labels, classes = labeler(selected_formulas)

    datasets: List[NetworkDataset[int]] = []

    for formula_hash, label in selected_labels.items():
        file = dir_formulas[formula_hash]
        formula_object = selected_formulas[formula_hash]

        logger.info(f"\tLoading {formula_hash}: {formula_object}: {label}")

        file_path = os.path.join(model_path, file)
        dataset = NetworkDataset(
            file=file_path,
            label=label,
            # limit=limit,
            _legacy_load_without_batch=_legacy_load_without_batch)

        datasets.append(dataset)

    return LabeledDataset.from_iterable(datasets), \
        classes, selected_formulas, selected_labels
