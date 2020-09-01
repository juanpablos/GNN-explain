import logging
import os

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

from src.typing import MetricHistory

logger = logging.getLogger(__name__)


def plot_training(
        metric_history: MetricHistory,
        save_path: str,
        use_selected: bool = False,
        filename: str = None,
        title: str = None):
    fig, ax1 = plt.subplots(figsize=(20, 10))
    ax2 = ax1.twinx()  # type: ignore

    plt.title(title if title is not None else "", wrap=True)

    logger.debug("Plotting metrics")
    x_axis = None
    for name, historic in metric_history.items(select=use_selected):
        if x_axis is None:
            x_axis = np.arange(len(historic)) + 1
        if any(m in name for m in ["precision", "recall", "f1", "acc"]):
            ax2.plot(x_axis, historic, label=name)
        else:
            ax1.plot(x_axis, historic, label=name)  # type: ignore

    ax1.set_xlabel("Epochs")  # type: ignore

    ax1.set_ylabel("Losses")  # type: ignore
    ax2.set_ylabel("Metrics")

    ax1.set_ylim(bottom=0)  # type: ignore
    ax2.set_ylim((0, 1))

    ax1.xaxis.set_major_locator(ticker.MaxNLocator())  # type: ignore

    ax2.yaxis.set_major_locator(ticker.MultipleLocator(0.1))

    ax1.margins(0)  # type: ignore
    ax2.margins(0)

    mat1, lab1 = ax1.get_legend_handles_labels()  # type: ignore
    mat2, lab2 = ax2.get_legend_handles_labels()

    ax2.legend(mat1 + mat2, lab1 + lab2)

    for line, label in zip(mat1, lab1):
        y = line.get_ydata()[-1]
        ax1.annotate(f"{label} ({y:.2f})",  # type: ignore
                     xy=(1, y),
                     xytext=(6, 0),
                     color=line.get_color(),
                     xycoords=ax1.get_yaxis_transform(),  # type: ignore
                     textcoords="offset points",
                     size=14,
                     va="center")
    for line, label in zip(mat2, lab2):
        y = line.get_ydata()[-1]
        ax2.annotate(f"{label} ({y:.2f})",  # type: ignore
                     xy=(1, y),
                     xytext=(6, 0),
                     color=line.get_color(),
                     xycoords=ax2.get_yaxis_transform(),  # type: ignore
                     textcoords="offset points",
                     size=14,
                     va="center")

    plt.tight_layout()
    os.makedirs(f"{save_path}/train/", exist_ok=True)
    plt.savefig(f"{save_path}/train/{filename}.png")
    plt.close()
