
def clean_state(model_dict):
    # TODO: maybe compress the tensors too
    return {k: v for k, v in model_dict.items() if "batch" not in k}
