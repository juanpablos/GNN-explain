import logging
import os
import random

import numpy as np
import torch

from src.training import TrainerBuilder
from src.training.utils import StopTraining
from src.typing import MinModelConfig, StopFormat, T

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
    train_builder: TrainerBuilder[T],
    model_config: MinModelConfig,
    iterations: int,
    gpu_num: int,
    lr: float = 0.01,
    stop_when: StopFormat = None,
    run_train_test: bool = False
):

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{gpu_num}")
    else:
        device = torch.device("cpu")

    train_builder.init_device(device=device)

    model = train_builder.init_model(**model_config)

    train_builder.init_loss()
    train_builder.init_optim(lr=lr)

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
