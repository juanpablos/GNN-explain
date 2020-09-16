import logging
import os
from typing import Dict, List

from src.data.datasets import (
    AggregatedNetworkDataset,
    LabeledDataset,
    NetworkDataset,
    NetworkDatasetCollectionWrapper
)
from src.data.formula_index import FormulaMapping
from src.data.formulas.filter import Filter
from src.data.formulas.labeler import (
    LabelerApply,
    MultiLabelCategoricalLabeler
)
from src.data.utils import label_idx2tensor
from src.typing import S, Selectable, T

logger = logging.getLogger(__name__)


def __prepare_files(path: str):
    files: Dict[str, str] = {}
    # reproducibility, always sorted files
    for file in sorted(os.listdir(path)):
        if file.endswith(".pt"):
            _hash = file.split(".")[0].split("-")[-1]
            files[_hash] = file
    return files


def load_gnn_files(
        root: str,
        model_hash: str,
        selector: Filter,
        labeler: LabelerApply[T, S],
        formula_mapping: FormulaMapping,
        test_selector: Filter,
        load_aggregated: str = None,
        _legacy_load_without_batch: bool = False):

    if model_hash not in os.listdir(root):
        raise FileExistsError(
            f"No directory for the current model hash: {root}/{model_hash}")

    is_multilabel = False
    if isinstance(labeler.labeler, MultiLabelCategoricalLabeler):
        logger.debug("Using a multilabel labeler")
        is_multilabel = True

    model_path = os.path.join(root, model_hash)

    preloaded_formulas: Selectable = {}
    if load_aggregated is None:
        # select all formulas available in directory
        # formula_hash -> file_path
        available_formulas = __prepare_files(model_path)

    else:
        logging.info("Loading batch formulas")
        _agg = AggregatedNetworkDataset(
            file_path=os.path.join(model_path, load_aggregated))

        available_formulas = _agg.available_formulas()
        preloaded_formulas = _agg

    logger.debug("Creating formula objects")
    # mapping from formula_hash -> formula object
    hash_formula = {_hash: formula_mapping[_hash]
                    for _hash in available_formulas}

    logger.debug(f"Running formula selector {selector}")
    # mapping from the selected formula_hash -> formula object
    selected_formulas = selector(hash_formula)
    logger.debug(f"Running test formula selector {test_selector}")
    testing_selected_formulas = test_selector(hash_formula)
    if testing_selected_formulas:
        logger.debug("Adding exclusive testing formulas")
        selected_formulas.update(testing_selected_formulas)

    logger.debug(f"Running formula labeler {labeler}")
    # mapping from the selected
    #   single label: formula_hash -> label_id
    #   multilabel: formula_hash -> List[label_id]
    # classes is a dictionary label_id -> label_name
    selected_labels, classes = labeler(selected_formulas)

    # contains all formulas in use in the experiment
    datasets: List[NetworkDataset[T]] = []

    # contains formulas used for training when test manually selected
    train_dataset: List[NetworkDataset[T]] = []
    # contains formulas used for testing when test manually selected
    test_dataset: List[NetworkDataset[T]] = []

    n_labels = len(classes)

    logger.info(f"Loading {len(selected_labels)} formulas")
    for formula_hash, label in selected_labels.items():
        formula_object = selected_formulas[formula_hash]

        logger.info(f"\tLoading {formula_hash}: {formula_object}: {label}")

        if is_multilabel:
            label = label_idx2tensor(label=label, n_labels=n_labels)

        if load_aggregated is None:
            file = available_formulas[formula_hash]
            file_path = os.path.join(model_path, file)
            dataset = NetworkDataset(
                file=file_path,
                label=label,
                formula=formula_object,
                _legacy_load_without_batch=_legacy_load_without_batch)
        else:
            dataset = NetworkDataset(
                preloaded=preloaded_formulas[formula_hash],
                label=label,
                formula=formula_object,
                _legacy_load_without_batch=_legacy_load_without_batch)

        if testing_selected_formulas:
            if formula_hash in testing_selected_formulas:
                test_dataset.append(dataset)
            else:
                train_dataset.append(dataset)

        # we append all formulas here
        datasets.append(dataset)

    if testing_selected_formulas:
        assert len(test_dataset) > 0, "test_dataset is empty"
        return ((LabeledDataset.from_iterable(train_dataset,
                                              multilabel=is_multilabel),
                 LabeledDataset.from_iterable(test_dataset,
                                              multilabel=is_multilabel)),
                classes,
                selected_formulas,
                selected_labels,
                NetworkDatasetCollectionWrapper(test_dataset))
    else:
        # when the test_set is not manually selected we return a
        # big dataset containing all formulas
        return (
            LabeledDataset.from_iterable(datasets, multilabel=is_multilabel),
            classes,
            selected_formulas,
            selected_labels,
            NetworkDatasetCollectionWrapper(datasets))
