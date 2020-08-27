import logging
import os
from typing import Dict, List, Optional, Union

from src.data.datasets import (
    LabeledDataset,
    NetworkDataset,
    NetworkDatasetCollectionWrapper
)
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
                   testing_selection: Optional[List[str]] = None,
                   _legacy_load_without_batch: bool = False):

    if model_hash not in os.listdir(root):
        raise FileExistsError(
            f"No directory for the current model hash: {root}/{model_hash}")

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
    if testing_selection is not None:
        logger.debug("Adding exclusive testing formulas")
        for _hash in testing_selection:
            selected_formulas[_hash] = formula_mapping[_hash]

    logger.debug(f"Running formula labeler {labeler}")
    # mapping from the selected formula_hash -> label_id
    # classes is a dictionary label_id -> label_name
    selected_labels, classes = labeler(selected_formulas)

    # contains all formulas in use in the experiment
    datasets: List[NetworkDataset[int]] = []

    # contains formulas used for training when test manually selected
    train_dataset: List[NetworkDataset[int]] = []
    # contains formulas used for testing when test manually selected
    test_dataset: List[NetworkDataset[int]] = []
    # do not check existance over a list
    test_formulas = set()
    if testing_selection is not None:
        logger.info("Pre-splitting for testing with selected formulas")
        test_formulas.update(testing_selection)

    logger.info(f"Loading {len(selected_labels)} formulas")
    for formula_hash, label in selected_labels.items():
        file = dir_formulas[formula_hash]
        formula_object = selected_formulas[formula_hash]

        logger.info(f"\tLoading {formula_hash}: {formula_object}: {label}")

        file_path = os.path.join(model_path, file)
        dataset = NetworkDataset(
            file=file_path,
            label=label,
            formula=formula_object,
            _legacy_load_without_batch=_legacy_load_without_batch)

        if testing_selection is not None:
            if formula_hash in test_formulas:
                test_dataset.append(dataset)
            else:
                train_dataset.append(dataset)

        # we append all formulas here
        datasets.append(dataset)

    if testing_selection is None:
        # when the test_set is not manually selected we return a
        # big dataset containing all formulas
        return (LabeledDataset.from_iterable(datasets),
                classes, selected_formulas, selected_labels,
                NetworkDatasetCollectionWrapper(datasets))

    else:
        assert len(test_dataset) > 0, "test_dataset is empty"
        return ((LabeledDataset.from_iterable(train_dataset),
                 LabeledDataset.from_iterable(test_dataset)),
                classes, selected_formulas, selected_labels,
                NetworkDatasetCollectionWrapper(test_dataset))
