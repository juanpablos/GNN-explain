import os
import torch
from src.data.gnn.utils import prepare_files

# remove gnns that do not reach perfect training

path = "./data/gnns_v2/"
log_file = "formulas_v2.json.4.log"
model_hash = "40e65407aa"

formula_path = os.path.join(path, model_hash)
cleaned_path = os.path.join(formula_path, "cleaned")

formula_hash_to_file = prepare_files(path=formula_path)

current_formula_hash = None
current_formulas = []
current_gnn_index = 0
cleaned_formulas = []

skip_hashes = set()

with open(os.path.join(path, log_file)) as f:
    for line in f:
        _current_formula_hash = line[:10]
        if _current_formula_hash in skip_hashes:
            continue
        if current_formula_hash != _current_formula_hash:
            if current_formula_hash is not None:
                # save what we have and reset
                formula_file_cleaned = formula_hash_to_file[
                    current_formula_hash
                ].replace("5000", str(len(cleaned_formulas)))
                print(f"saving formula {current_formula_hash}")
                torch.save(
                    cleaned_formulas, os.path.join(cleaned_path, formula_file_cleaned)
                )

                cleaned_formulas = []
                current_gnn_index = 0

            if _current_formula_hash not in formula_hash_to_file:
                print(f"Skiping hash {_current_formula_hash}")
                skip_hashes.add(_current_formula_hash)
                current_formula_hash = None
                continue

            current_formula_hash = _current_formula_hash
            formula_file = formula_hash_to_file[current_formula_hash]
            current_formulas = torch.load(os.path.join(formula_path, formula_file))

        if "Training model" in line:
            current_gnn_index = int(line.split("Training model ")[1].split("/")[0]) - 1

        if "src.run_logic INFO" in line and 'INFO " 30' not in line:
            print(f"Adding gnn {current_gnn_index} - on hash {current_formula_hash}")
            cleaned_formulas.append(current_formulas[current_gnn_index])

    if cleaned_formulas and current_formula_hash is not None:
        # when finished, save what is left
        formula_file_cleaned = formula_hash_to_file[current_formula_hash].replace(
            "5000", str(len(cleaned_formulas))
        )
        print(f"saving formula {current_formula_hash}")
        torch.save(cleaned_formulas, os.path.join(cleaned_path, formula_file_cleaned))
