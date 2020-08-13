import logging

import matplotlib.pyplot as plt
from src.typing import MetricHistory

logger = logging.getLogger(__name__)


def plot_training(
        metric_history: MetricHistory,
        save_path: str,
        use_selected: bool = False,
        file_name: str = None,
        title: str = None):
    fig, ax1 = plt.subplots(figsize=(20, 10))
    ax2 = ax1.twinx()  # type: ignore

    plt.title(title if title is not None else "")

    logger.debug("Plotting metrics")
    length = 0
    for name, historic in metric_history.items(select=use_selected):
        if length == 0:
            length = len(historic)
        if any(m in name for m in ["precision", "recall", "f1", "acc"]):
            ax2.plot(historic, label=name)
        else:
            ax1.plot(historic, label=name)  # type: ignore

    ax1.set_xlabel("Epochs")  # type: ignore

    ax1.set_ylabel("Losses")  # type: ignore
    ax2.set_ylabel("Metrics")

    ax1.set_ylim(bottom=0)  # type: ignore
    ax2.set_ylim((0, 1))

    ax1.set_xticks(list(range(length)))  # type: ignore
    ax1.set_xticklabels(list(range(1, length + 1)))  # type: ignore

    ax1.margins(0)  # type: ignore
    ax2.margins(0)

    mat1, lab1 = ax1.get_legend_handles_labels()  # type: ignore
    mat2, lab2 = ax2.get_legend_handles_labels()

    ax2.legend(mat1 + mat2, lab1 + lab2)

    for line, label in zip(mat1, lab1):
        y = line.get_ydata()[-1]
        ax1.annotate(label,  # type: ignore
                     xy=(1, y),
                     xytext=(6, 0),
                     color=line.get_color(),
                     xycoords=ax1.get_yaxis_transform(),  # type: ignore
                     textcoords="offset points",
                     size=14,
                     va="center")
    for line, label in zip(mat2, lab2):
        y = line.get_ydata()[-1]
        ax2.annotate(label,  # type: ignore
                     xy=(1, y),
                     xytext=(6, 0),
                     color=line.get_color(),
                     xycoords=ax2.get_yaxis_transform(),  # type: ignore
                     textcoords="offset points",
                     size=14,
                     va="center")

    plt.tight_layout()
    plt.savefig(f"{save_path}/train/{file_name}.png")
    plt.close()
