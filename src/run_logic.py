import os
import random
from typing import Any, Dict

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    from torch_geometric.data import DataLoader
except ImportError:
    from torch.utils.data import DataLoader


def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def run(
    run_config,
    model_config: Dict[str, Any],
    train_graphs: Dataset,
    test_graphs: Dataset,
    iterations: int,
    gpu_num: int,
    data_workers: int,
    batch_size: int = 64,
    test_batch_size: int = 512,
    lr: float = 0.01
):

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{gpu_num}")
    else:
        device = torch.device("cpu")

    if os.name == "nt":
        data_workers = 0

    train_loader = DataLoader(
        train_graphs,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=True,
        num_workers=data_workers)
    test_loader = DataLoader(
        test_graphs,
        batch_size=test_batch_size,
        pin_memory=True,
        shuffle=True,
        num_workers=data_workers)

    model = run_config.get_model(**model_config)

    model = model.to(device)

    criterion = run_config.get_loss()
    optimizer = run_config.get_optim(model=model, lr=lr)
    scheduler = run_config.get_scheduler(optimizer=optimizer)

    for it in range(1, iterations + 1):

        train_loss = run_config.train(
            model=model,
            training_data=train_loader,
            criterion=criterion,
            device=device,
            optimizer=optimizer,
            scheduler=scheduler,
            binary_prediction=True
        )

        # _, train_micro_acc, train_macro_acc = evaluate(
        #     model=model,
        #     test_data=train_loader,
        #     criterion=criterion,
        #     device=device,
        #     binary_prediction=True)

        # TODO: remove
        if it == iterations:
            test_loss, test_micro_acc, test_macro_acc = run_config.evaluate(
                model=model,
                test_data=test_loader,
                criterion=criterion,
                device=device,
                binary_prediction=True)

            print(
                it,
                f"loss {train_loss: .6f} test_loss {test_loss: .6f} micro {test_micro_acc: .4f} macro {test_macro_acc: .4f}")

        # TODO: better way to handle when the model is not perfect

        # TODO: implement a logger (do not need the logger for the training
        # GNN)

    return model
