def reset(nn):
    def _reset(item):
        if hasattr(item, "reset_parameters"):
            item.reset_parameters()

    if nn is not None:
        if hasattr(nn, "children") and len(list(nn.children())) > 0:
            for item in nn.children():
                _reset(item)
        else:
            _reset(nn)


class Waiter:
    def __init__(self, wait_for: int):
        self.remaining = wait_for

    def ok(self):
        if self.remaining <= 0:
            return True
        else:
            self.remaining -= 1
            return False


def count_parameters(model):
    with_grad_params = 0
    all_params = 0
    for p in model.parameters():
        all_params += p.numel()
        if p.requires_grad:
            with_grad_params += p.numel()
    return all_params, with_grad_params
