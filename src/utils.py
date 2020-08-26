import csv
import json
import logging
import os
from collections import defaultdict
from collections.abc import Mapping
from typing import Any, Dict

from src.graphs.foc import FOC, Element
from src.typing import GNNModelConfig

logger = logging.getLogger(__name__)


def merge_update(dict1, dict2):
    """
    Only updates if the 2 dictionaries have the same set of keys and types
    Basically they are the same DictType. Other cases are not supported.
    """
    for k, v in dict2.items():
        if isinstance(v, Mapping):
            dict1[k] = merge_update(dict1[k], v)
        else:
            dict1[k] += v

    return dict1


def save_file_exists(root: str, file_name: str):
    """Check if the tuple (model type, model config, formula) already exists (ignores the number of models saved)"""
    def _create_tuple(s):
        left = s.split(".pt")[0]
        splitted = left.split("-")
        # skip the nX
        return splitted[0], splitted[2], splitted[3]

    searching = _create_tuple(file_name)

    logger.debug(f"Searching for existing file: {searching}")
    for file in os.listdir(root):
        if file.endswith(".pt"):
            other = _create_tuple(file)

            if searching == other:
                return True, file

    return False, None


def cleanup(exists, path, file):
    """Does not check if file exists, it should already exist if exists is true. Otherwise a race condition was met, but that is not checked"""
    if exists:
        logger.debug("Deleting old files")
        file_path = os.path.join(path, file)
        os.remove(file_path)
        os.remove(f"{file_path}.stat")


def write_metadata(
        file_path: str,
        model_config: GNNModelConfig,
        model_config_hash: str,
        formula: FOC,
        formula_hash: str,
        data_config: Dict[str, Any],
        seed: int,
        **kwargs: Any):

    logging.debug("Writing metadata")

    """
    * format is:
    * formula hash, formula string, model hash, seed,
    *    model config, data config, others
    """
    with open(file_path, "a", newline='', encoding='utf-8') as f:
        writer = csv.writer(f, quotechar="|")
        writer.writerow([
            formula_hash,
            repr(formula),
            model_config_hash,
            seed,
            json.dumps(model_config),
            json.dumps(data_config),
            json.dumps(kwargs)
        ])


def write_result_info(
        path: str,
        file_name: str,
        hash_formula: Dict[str, Element],
        hash_label: Dict[str, Any],
        classes: Dict[Any, str],
        write_mistakes: bool,
        mistakes: Dict[Element, int],
        formula_count: Dict[Element, int]):

    os.makedirs(f"{path}/info/", exist_ok=True)

    # format:
    # label_id label_name n_formulas
    # \t hash formula

    # label_id -> list[hashes]
    groups = defaultdict(list)
    for _hash, label in hash_label.items():
        groups[label].append(_hash)

    with open(f"{path}/info/{file_name}.txt", "w", encoding="utf-8") as o:
        o.write(f"Labels\n")
        for label_id, label_name in classes.items():
            hashes = groups[label_id]
            o.write(f"{label_id}\t{label_name}\t{len(hashes)}\n")
            for _hash in hashes:
                o.write(f"\t{_hash}\t{hash_formula[_hash]}\n")

        o.write(f"\nNumber of Mistakes\n")
        if write_mistakes:
            if mistakes:
                for formula, n_mistakes in mistakes.items():
                    total_points = formula_count[formula]
                    o.write(f"\t{formula}\t{n_mistakes}/{total_points}\n")
            else:
                o.write("No mistakes\n")
        else:
            o.write("Not available\n")
