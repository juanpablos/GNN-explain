import csv
import json
import logging
import os
from collections.abc import Mapping
from typing import Any, DefaultDict, Dict, List

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


def save_file_exists(root: str, filename: str):
    """Check if the tuple (model type, model config, formula) already exists (ignores the number of models saved)"""
    def _create_tuple(s):
        left = s.split(".pt")[0]
        splitted = left.split("-")
        # skip the nX
        return splitted[0], splitted[2], splitted[3]

    searching = _create_tuple(filename)

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


def get_next_filename(path: str, filename: str, ext: str):
    if os.path.exists(f"{path}/{filename}.{ext}"):
        counter = 1
        while os.path.exists(f"{path}/{filename} ({counter}).{ext}"):
            counter += 1

        count = f" ({counter})"
        filename = filename + count

        return filename, count
    else:
        return filename, ""


def write_result_info(
        path: str,
        filename: str,
        hash_formula: Dict[str, Element],
        hash_label: Dict[str, Any],
        classes: Dict[Any, str],
        multilabel: bool,
        mistakes: Dict[Element, int],
        formula_count: Dict[Element, int]):

    os.makedirs(f"{path}/info/", exist_ok=True)

    # format:
    # label_id label_name n_formulas
    # \t hash formula

    # label_id -> list[hashes]
    groups: DefaultDict[Any, List[str]] = DefaultDict(list)
    for _hash, label in hash_label.items():
        groups[label].append(_hash)

    max_formula_len = max(len(str(formula))
                          for formula in hash_formula.values())

    filename, counter = get_next_filename(path=f"{path}/info",
                                          filename=filename,
                                          ext="txt")

    with open(f"{path}/info/{filename}.txt", "w", encoding="utf-8") as o:
        # format:
        # label_id, label_name, n_formulas with label
        # - formula_hash, formula_repr [, n_mistakes/total formulas in test]
        for label_id, label_name in classes.items():
            hashes = groups[label_id]
            o.write(f"{label_id}\t{label_name}\t{len(hashes)}\n")

            if not multilabel:
                template = "\t{hash}\t{formula:<{pad}}{err}\n"
                key = lambda h: mistakes.get(hash_formula[h], 0)
                for _hash in sorted(hashes, key=key, reverse=True):
                    n_mistakes = mistakes.get(hash_formula[_hash], 0)
                    count = formula_count[hash_formula[_hash]]

                    line = template.format(
                        hash=_hash,
                        formula=str(hash_formula[_hash]),
                        err=f"{n_mistakes}/{count}",
                        pad=max_formula_len + 4)
                    o.write(line)
            else:
                for _hash in hashes:
                    o.write(f"\t{_hash}\t{hash_formula[_hash]}\n")

    return counter
