import os


def prepare_files(path: str, model_hash: str = None):
    files = set()
    duplicated = set()
    # reproducibility, always sorted files
    for file in sorted(os.listdir(path)):
        if (
            file.endswith(".pt")
            and "gnn-" in file
            and (model_hash is None or model_hash in file)
        ):
            _hash = file.split(".")[0].split("-")[-1]
            if _hash in files:
                duplicated.add(_hash)
            files.add(_hash)
    return duplicated


path = "./data/gnns_v3/"
model_hash = "40e65407aa"

formula_path = os.path.join(path, model_hash)
cleaned_path = os.path.join(formula_path, "cleaned")
print(prepare_files(cleaned_path))
