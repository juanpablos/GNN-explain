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


data_path = "data"
gnn_path = "gnns_v4"
model_hash = "40e65407aa"

# formula_path = os.path.join(data_path, "delete")
formula_path = os.path.join(data_path, gnn_path, model_hash)
print(prepare_files(formula_path))
