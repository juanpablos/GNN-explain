import logging
import os
from typing import Any, Dict, List, Optional

import torch

from src.data.auxiliary import (
    AggregatedNetworkDataset,
    FormulaAppliedDatasetWrapper,
    NetworkDatasetCollectionWrapper,
)
from src.data.dataset_splitter import (
    NetworkDatasetCrossFoldSplitter,
    TextNetworkDatasetCrossFoldSplitter,
)
from src.data.datasets import (
    LabeledDataset,
    LabeledSubset,
    NetworkDataset,
    TextSequenceDataset,
)
from src.data.formula_index import FormulaMapping
from src.data.formulas.filter import Filter
from src.data.formulas.labeler import (
    LabelerApply,
    MultiLabelCategoricalLabeler,
    SequenceLabelerApply,
)
from src.data.gnn.utils import prepare_files
from src.data.utils import label_idx2tensor
from src.typing import CrossFoldConfiguration, S, T

logger = logging.getLogger(__name__)


def __load_formulas(
    root: str,
    model_hash: str,
    selector: Filter,
    formula_mapping: FormulaMapping,
    test_selector: Filter,
    load_aggregated: str = None,
    force_preaggregated: bool = False,
):
    if model_hash not in os.listdir(root):
        raise FileExistsError(
            f"No directory for the current model hash: {root}/{model_hash}"
        )

    if load_aggregated is None and force_preaggregated:
        raise ValueError(
            "GNNs as graphs cannot be loaded on demand, they must be preloaded using `stack_gnn_graphs`"
        )

    model_path = os.path.join(root, model_hash)

    preloaded = None
    if load_aggregated is None:
        # select all formulas available in directory
        # formula_hash -> file_path
        available_formulas = prepare_files(model_path)

    else:
        logging.info("Loading batch formulas")
        preloaded = AggregatedNetworkDataset(
            file_path=os.path.join(model_path, "processed", load_aggregated)
        )

        available_formulas = preloaded.available_formulas()

    logger.debug("Creating formula objects")
    # mapping from formula_hash -> formula object
    hash_formula = {_hash: formula_mapping[_hash] for _hash in available_formulas}

    logger.debug(f"Running formula selector {selector}")
    # mapping from the selected formula_hash -> formula object
    selected_formulas = selector(hash_formula)
    logger.debug(f"Running test formula selector {test_selector}")
    testing_selected_formulas = test_selector(hash_formula)
    if testing_selected_formulas:
        logger.debug("Adding exclusive testing formulas")
        selected_formulas.update(testing_selected_formulas)

    return (
        model_path,
        selected_formulas,
        testing_selected_formulas,
        available_formulas,
        preloaded,
    )


def categorical_loader(
    root: str,
    model_hash: str,
    selector: Filter,
    labeler: LabelerApply[T, S],
    formula_mapping: FormulaMapping,
    test_selector: Filter,
    load_aggregated: str = None,
    force_preaggregated: bool = False,
    cross_fold_configuration: Optional[CrossFoldConfiguration] = None,
    labeler_stored_state: Optional[Dict] = None,
    _legacy_load_without_batch: bool = False,
):

    (
        model_path,
        selected_formulas,
        testing_selected_formulas,
        available_formulas,
        preloaded,
    ) = __load_formulas(
        root=root,
        model_hash=model_hash,
        selector=selector,
        formula_mapping=formula_mapping,
        test_selector=test_selector,
        load_aggregated=load_aggregated,
        force_preaggregated=force_preaggregated,
    )

    logger.debug(f"Running formula labeler {labeler}")
    # mapping from the selected
    #   single label: formula_hash -> label_id
    #   multilabel: formula_hash -> List[label_id]
    # classes is a dictionary label_id -> label_name
    if labeler_stored_state is not None:
        labeler.load_labeler_data(data=labeler_stored_state)
    selected_labels, classes = labeler(selected_formulas)
    serialized_labeler = labeler.serialize()
    n_labels = len(classes)

    # * if multilabel, selected_labels is remapped to
    # * formula_hash -> multilabel tensor mask

    is_multilabel = False
    if isinstance(labeler.labeler, MultiLabelCategoricalLabeler):
        logger.debug("Using a multilabel labeler")
        is_multilabel = True
        selected_labels = {
            h: label_idx2tensor(l, n_labels=n_labels)
            for h, l in selected_labels.items()
        }

    # contains all formulas in use in the experiment
    datasets: List[NetworkDataset[T]] = []
    total_samples = 0

    # contains formulas used for training when test manually selected
    train_dataset: List[int] = []
    # contains formulas used for testing when test manually selected
    test_dataset: List[int] = []

    logger.info(f"Labeling {len(selected_labels)} formulas")
    for formula_hash, label in selected_labels.items():
        formula_object = selected_formulas[formula_hash]

        if preloaded is None:
            logger.info(f"\tLoading {formula_hash}: {formula_object}: {label}")
            file = available_formulas[formula_hash]
            file_path = os.path.join(model_path, file)
            dataset = NetworkDataset.categorical(
                label=label,
                formula=formula_object,
                formula_hash=formula_hash,
                file=file_path,
                multilabel=is_multilabel,
                _legacy_load_without_batch=_legacy_load_without_batch,
            )
        else:
            logger.info(f"\tLabeling {formula_hash}: {formula_object}: {label}")
            dataset = NetworkDataset.categorical(
                label=label,
                formula=formula_object,
                formula_hash=formula_hash,
                preloaded=preloaded[formula_hash],
                multilabel=is_multilabel,
            )

        if testing_selected_formulas and cross_fold_configuration is not None:
            current_indices = [i + total_samples for i in range(len(dataset))]
            if formula_hash in testing_selected_formulas:
                test_dataset.extend(current_indices)
            else:
                train_dataset.extend(current_indices)

        # we append all formulas here
        datasets.append(dataset)
        total_samples += len(dataset)

    if cross_fold_configuration is not None:
        return_dataset = NetworkDatasetCrossFoldSplitter(
            datasets=datasets, crossfold_config=cross_fold_configuration
        )
    else:
        dataset_all = LabeledDataset.from_iterable(datasets, multilabel=is_multilabel)

        if testing_selected_formulas:
            assert len(test_dataset) > 0, "test_dataset is empty"
            return_dataset = (
                LabeledSubset(dataset=dataset_all, indices=train_dataset),
                LabeledSubset(dataset=dataset_all, indices=test_dataset),
            )
        else:
            return_dataset = dataset_all

    return (
        return_dataset,
        classes,
        selected_formulas,
        selected_labels,
        NetworkDatasetCollectionWrapper(datasets),
        serialized_labeler,
    )


def text_sequence_loader(
    root: str,
    model_hash: str,
    selector: Filter,
    labeler: SequenceLabelerApply,
    formula_mapping: FormulaMapping,
    test_selector: Filter,
    graph_config: Dict[str, Any],
    load_aggregated: str = None,
    force_preaggregated: bool = False,
    cross_fold_configuration: CrossFoldConfiguration = None,
    labeler_stored_state: Optional[Dict] = None,
    return_list_of_datasets: bool = False,
    _legacy_load_without_batch: bool = False,
):

    (
        model_path,
        selected_formulas,
        testing_selected_formulas,
        available_formulas,
        preloaded,
    ) = __load_formulas(
        root=root,
        model_hash=model_hash,
        selector=selector,
        formula_mapping=formula_mapping,
        test_selector=test_selector,
        load_aggregated=load_aggregated,
        force_preaggregated=force_preaggregated,
    )

    logger.debug(f"Running formula labeler {labeler}")
    # mapping from the selected
    #   formula_hash -> List[token_id]
    # vocabulary is a Vocabulary object holding the tokens and their id mapping
    if labeler_stored_state is not None:
        labeler.load_labeler_data(data=labeler_stored_state)
    selected_labels, vocabulary = labeler(selected_formulas)
    serialized_labeler = labeler.serialize()
    # ! vocabulary is joint for train and test, there is no <unk> token

    # contains all formulas in use in the experiment
    datasets: List[NetworkDataset[torch.Tensor]] = []
    total_samples = 0

    # contains formulas used for training when test manually selected
    train_dataset: List[int] = []
    # contains formulas used for testing when test manually selected
    test_dataset: List[int] = []

    logger.info(f"Labeling {len(selected_labels)} formulas")
    for formula_hash, label in selected_labels.items():
        formula_object = selected_formulas[formula_hash]

        label = torch.tensor(label)

        if preloaded is None:
            logger.info(f"\tLoading {formula_hash}: {formula_object}: {label}")
            file = available_formulas[formula_hash]
            file_path = os.path.join(model_path, file)
            dataset = NetworkDataset.text_sequence(
                label=label,
                formula=formula_object,
                formula_hash=formula_hash,
                file=file_path,
                vocabulary=vocabulary,
                _legacy_load_without_batch=_legacy_load_without_batch,
            )
        else:
            logger.info(f"\tLabeling {formula_hash}: {formula_object}: {label}")
            dataset = NetworkDataset.text_sequence(
                label=label,
                formula=formula_object,
                formula_hash=formula_hash,
                preloaded=preloaded[formula_hash],
                vocabulary=vocabulary,
            )

        if testing_selected_formulas:
            current_indices = [i + total_samples for i in range(len(dataset))]
            if formula_hash in testing_selected_formulas:
                test_dataset.extend(current_indices)
            else:
                train_dataset.extend(current_indices)

        # we append all formulas here
        datasets.append(dataset)
        total_samples += len(dataset)

    if cross_fold_configuration is not None:
        return_dataset = TextNetworkDatasetCrossFoldSplitter(
            datasets=datasets,
            crossfold_config=cross_fold_configuration,
            vocabulary=vocabulary,
        )
    elif return_list_of_datasets:
        return_dataset = datasets
    else:
        dataset_all = TextSequenceDataset.from_iterable(datasets, vocabulary=vocabulary)

        if testing_selected_formulas:
            assert len(test_dataset) > 0, "test_dataset is empty"
            return_dataset = (
                LabeledSubset(dataset=dataset_all, indices=train_dataset),
                LabeledSubset(dataset=dataset_all, indices=test_dataset),
            )
        else:
            return_dataset = dataset_all

    return (
        return_dataset,
        vocabulary,
        selected_formulas,
        selected_labels,
        NetworkDatasetCollectionWrapper(datasets),
        FormulaAppliedDatasetWrapper(datasets, **graph_config),
        serialized_labeler,
    )
