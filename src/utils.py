import csv
from inspect import getsourcefile
import json
import os
from collections.abc import Mapping
from typing import Callable, Dict
from src.graphs.foc import FOC
from src.typing import GNNModelConfig


def merge_update(dict1, dict2):
    """Only updates if the 2 dictionaries have the same set of keys and types
    Basically they are the same DictType. Other cases are not supported.
    """
    for k, v in dict2.items():
        if isinstance(v, Mapping):
            dict1[k] = merge_update(dict1[k], v)
        else:
            dict1[k] += v

    return dict1


def save_file_exists(root: str, file_name: str):
    """Check if the tuple (model type, model config, formula) already exists (ignores the number of models saved)
    """
    def _create_tuple(s):
        left = s.split(".pt")[0]
        splitted = left.split("-")
        # skip the nX
        return splitted[0], splitted[2], splitted[3]

    searching = _create_tuple(file_name)

    for file in os.listdir(root):
        if file.endswith(".pt"):
            other = _create_tuple(file)

            if searching == other:
                return True, file

    return False, None


def cleanup(exists, path, file):
    """Does not check if file exists, it should already exist if exists is true. Otherwise a race condition was met, but that is not checked
    """
    if exists:
        file_path = os.path.join(path, file)
        os.remove(file_path)
        os.remove(f"{file_path}.stat")


def write_metadata(
        destination: str,
        model_config: GNNModelConfig,
        model_config_hash: str,
        formula: FOC,
        formula_hash: str,
        data_config: Dict,
        seed: int,
        formula_fn: Callable[[], FOC],
        **kwargs):
    formula_source = getsourcefile(formula_fn)

    """
    * format is:
    * formula hash, formula string, model hash, seed,
    *    model config, data config, others, formula source
    """
    with open(destination, "a", newline='', encoding='utf-8') as f:
        writer = csv.writer(f, quotechar="|")
        writer.writerow([
            formula_hash,
            repr(formula),
            model_config_hash,
            seed,
            json.dumps(model_config),
            json.dumps(data_config),
            json.dumps(kwargs),
            formula_source
        ])
