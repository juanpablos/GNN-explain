import logging
import math
import os
from typing import List

import matplotlib.pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    confusion_matrix,
    multilabel_confusion_matrix,
)

logger = logging.getLogger(__name__)


def plot_confusion_matrix(
    y,
    y_pred,
    save_path: str,
    labels: List[str],
    filename: str = None,
    title: str = None,
    *,
    normalize_cm: bool = True,
):
    logger.info(f"Calculating confusion matrix")

    if normalize_cm:
        normalize = "true"
    else:
        normalize = None

    matrix_og = confusion_matrix(y, y_pred, normalize=normalize)

    size = 10 if len(labels) < 10 else len(labels) * 1.25
    fig, ax = plt.subplots(figsize=(size, size))

    disp = ConfusionMatrixDisplay(matrix_og, display_labels=labels)

    logger.debug("Plotting confusion matrix")
    disp.plot(cmap="Blues", ax=ax, xticks_rotation=30)

    plt.setp(
        ax.get_xticklabels(),
        rotation=30,  # type:ignore
        horizontalalignment="right",
    )
    if normalize_cm:
        disp.im_.set_clim(0, 1)
    plt.tight_layout()

    figure_name = filename if filename is not None else "confusion_matrix"
    figure_title = title if title is not None else ""

    plt.title(figure_title, wrap=True)

    os.makedirs(f"{save_path}/cm/", exist_ok=True)
    plt.savefig(f"{save_path}/cm/{figure_name}.png")
    plt.close()


def plot_multilabel_confusion_matrix(
    y,
    y_pred,
    save_path: str,
    labels: List[str],
    label_totals: List[int],
    filename: str = None,
    title: str = None,
):
    logger.info(f"Calculating multilabel confusion matrix")

    cm_array = multilabel_confusion_matrix(y, y_pred)
    size = len(labels) * 3

    n_rows = math.ceil(math.sqrt(len(labels)))
    n_cols = math.ceil(len(labels) / n_rows)
    fig, axs = plt.subplots(ncols=int(n_cols), nrows=int(n_rows), figsize=(size, size))

    axs = axs.flatten()  # type: ignore

    logger.debug("Plotting confusion matrices")
    for i, (cm, ax) in enumerate(zip(cm_array, axs)):

        norm_cm = cm / cm.sum(axis=1, keepdims=True)

        rest_label = f"rest ({sum(label_totals) - label_totals[i]})"
        current_label = f"{labels[i]} ({label_totals[i]})"

        label_names = [rest_label, current_label]
        disp = ConfusionMatrixDisplay(
            confusion_matrix=norm_cm, display_labels=label_names
        )

        logger.debug(f"Plotting for {labels[i]}")
        disp.plot(cmap="Blues", ax=ax, xticks_rotation=30)

        plt.setp(
            ax.get_xticklabels(),
            rotation=30,  # type:ignore
            horizontalalignment="right",
        )
        disp.im_.set_clim(0, 1)

        ax.set_title(f"CM for all vs {labels[i]}")

    figure_name = filename if filename is not None else "confusion_matrix"
    figure_title = title if title is not None else ""

    plt.suptitle(figure_title, wrap=True)
    plt.tight_layout()

    os.makedirs(f"{save_path}/cm/", exist_ok=True)
    plt.savefig(f"{save_path}/cm/{figure_name}.png")
    plt.close()
