import os
import random
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from torch_geometric.data import DataLoader

from .gnn import ACGNN
from .training import evaluate, train


def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def __get_model(name: str,
                input_dim: int,
                hidden_dim: int,
                output_dim: int,
                aggregate_type: str,
                combine_type: str,
                num_layers: int,
                combine_layers: int,
                num_mlp_layers: int,
                task: str,
                truncated_fn: Tuple[int, int]):

    if name == "acgnn":
        return ACGNN(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            aggregate_type=aggregate_type,
            combine_type=combine_type,
            num_layers=num_layers,
            combine_layers=combine_layers,
            num_mlp_layers=num_mlp_layers,
            task=task,
            truncated_fn=truncated_fn
        )
    else:
        raise NotImplementedError


def run(
    model_config: Dict[str, Any],
    train_graphs: Dataset,
    test_graphs: Dataset,
    iterations: int,
    gpu_num: int,
    data_workers: int,
    batch_size: int = 64,
    lr: float = 0.01
):

    if torch.cuda.is_available():
        device = torch.device("cuda:" + gpu_num)
    else:
        device = torch.device("cpu")

    if os.name == "nt":
        data_workers = 0

    train_loader = DataLoader(
        train_graphs,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=data_workers)
    test_loader = DataLoader(
        test_graphs,
        batch_size=512,
        pin_memory=True,
        num_workers=data_workers)

    model = __get_model(**model_config)

    model = model.to(device)

    criterion = nn.BCEWithLogitsLoss(reduction='mean')
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=50, gamma=0.5)

    for _ in range(iterations):

        train_losses = train(
            model=model,
            training_data=train_loader,
            criterion=criterion,
            device=device,
            optimizer=optimizer,
            scheduler=scheduler,
            binary_prediction=True
        )

        _, train_micro_acc, train_macro_acc = evaluate(
            model=model,
            test_data=train_loader,
            criterion=criterion,
            device=device,
            binary_prediction=True)

        _, test_micro_acc, test_macro_acc = evaluate(
            model=model,
            test_data=test_loader,
            criterion=criterion,
            device=device,
            binary_prediction=True)

        # TODO: implement a logger (do not need the logger for the training
        # GNN)

    return model
