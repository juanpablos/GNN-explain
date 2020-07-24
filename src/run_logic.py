import logging
import os
import random
from typing import Any, Dict

import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from torch_geometric.data import DataLoader

from src.training.utils import StopTraining
from src.typing import MinModelConfig, StopFormat, Trainer


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)  # type: ignore
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)  # type: ignore
        torch.cuda.manual_seed_all(seed)  # type: ignore
        torch.backends.cudnn.deterministic = True  # type: ignore
        torch.backends.cudnn.benchmark = False  # type: ignore


def run(
    run_config: Trainer,
    model_config: MinModelConfig,
    train_data: Dataset,
    test_data: Dataset,
    iterations: int,
    gpu_num: int,
    data_workers: int,
    test_data_workers: int = 1,
    batch_size: int = 64,
    test_batch_size: int = 512,
    lr: float = 0.01,
    stop_when: StopFormat = None
):

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{gpu_num}")
    else:
        device = torch.device("cpu")

    if os.name == "nt":
        data_workers = 0
        test_data_workers = 0

    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=True,
        num_workers=data_workers)
    test_loader = DataLoader(
        test_data,
        batch_size=test_batch_size,
        pin_memory=True,
        shuffle=True,
        num_workers=test_data_workers)

    model = run_config.get_model(**model_config)
    model = model.to(device)

    criterion = run_config.get_loss()
    optimizer = run_config.get_optim(model=model, lr=lr)
    scheduler = run_config.get_scheduler(optimizer=optimizer)

    stop = StopTraining(stop_when)
    info: Dict[str, Any] = {}

    it = 1
    for it in range(1, iterations + 1):

        train_loss = run_config.train(
            model=model,
            training_data=train_loader,
            criterion=criterion,
            device=device,
            optimizer=optimizer,
            scheduler=scheduler,
            binary_prediction=True,
            collector=info
        )

        # _, train_micro_acc, train_macro_acc = evaluate(
        #     model=model,
        #     test_data=train_loader,
        #     criterion=criterion,
        #     device=device,
        #     binary_prediction=True)

        test_metrics = run_config.evaluate(
            model=model,
            test_data=test_loader,
            criterion=criterion,
            device=device,
            binary_prediction=True,
            collector=info)

        if stop(**info):
            break
        logging.debug(f"{it: <4} {run_config.log(info)}")

    logging.info(f"{it} {run_config.log(info)}")

    return model, info
