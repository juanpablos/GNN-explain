import logging
import os
import random

import numpy as np
import torch
from torch.utils.data.dataset import Dataset

from src.training import TrainerBuilder
from src.training.utils import StopTraining
from src.typing import MinModelConfig, StopFormat

logger = logging.getLogger(__name__)


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
    train_builder: TrainerBuilder,
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
    stop_when: StopFormat = None,
    run_train_test: bool = False
):

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{gpu_num}")
    else:
        device = torch.device("cpu")

    train_builder.init_device(device=device)

    if os.name == "nt":
        data_workers = 0
        test_data_workers = 0

    train_loader = train_builder.init_dataloader(
        train_data,
        mode="train",
        batch_size=batch_size,
        pin_memory=False,
        shuffle=True,
        num_workers=data_workers)
    test_loader = train_builder.init_dataloader(
        test_data,
        mode="test",
        batch_size=test_batch_size,
        pin_memory=False,
        shuffle=True,
        num_workers=test_data_workers)

    model = train_builder.init_model(**model_config)

    criterion = train_builder.init_loss()
    optimizer = train_builder.init_optim(lr=lr)

    trainer = train_builder.validate_trainer()
    stop = StopTraining(stop_when)

    it = 1
    for it in range(1, iterations + 1):

        train_loss = trainer.train(binary_prediction=True)

        if run_train_test:
            train_test_metrics = trainer.evaluate(
                use_train_data=True,
                binary_prediction=True,
            )

        test_metrics = trainer.evaluate(
            use_train_data=False,
            binary_prediction=True,
        )

        if stop(**trainer.metric_logger):
            break
        logger.debug(f"{it: 03d} {trainer.log()}")

    logger.info(f"{it: 03d} {trainer.log()}")

    return model
