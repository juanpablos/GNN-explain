import logging
import math
import os
from typing import List

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid
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
    plot_precision_and_recall: bool = True,
):
    logger.info(f"Calculating confusion matrix")
    size = 10 if len(labels) < 10 else len(labels) * 1.25

    if plot_precision_and_recall:
        fig = plt.figure(figsize=(2 * size, size))
        axs = AxesGrid(
            fig,
            111,
            nrows_ncols=(1, 2),
            axes_pad=0.5,
            cbar_mode="single",
            cbar_location="right",
            cbar_pad=0.5,
        )
        normalize_list = ["true", "pred"]
        titles = ["Recall", "Precision"]
    else:
        fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(size, size))
        axs = [ax]
        normalize_list = ["true" if normalize_cm else None]
        titles = [None]

    for ax, normalize, ax_title in zip(axs, normalize_list, titles):
        matrix_og = confusion_matrix(y, y_pred, normalize=normalize)

        disp = ConfusionMatrixDisplay(matrix_og, display_labels=labels)

        logger.debug("Plotting confusion matrix")
        disp.plot(cmap="Blues", ax=ax, xticks_rotation=30, colorbar=False)

        plt.setp(
            ax.get_xticklabels(),
            rotation=30,  # type:ignore
            horizontalalignment="right",
        )

        ax.set_title(ax_title)

    if normalize_cm or plot_precision_and_recall:
        disp.im_.set_clim(0, 1)
    ax.cax.colorbar(disp.im_)
    ax.cax.toggle_label(True)

    if not plot_precision_and_recall:
        plt.tight_layout()

    figure_name = filename if filename is not None else "confusion_matrix"
    figure_title = title if title is not None else ""

    plt.suptitle(figure_title, wrap=True)

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

    confusion_matrices = multilabel_confusion_matrix(y, y_pred)
    size = len(labels) * 3

    n_rows = math.ceil(math.sqrt(len(labels)))
    n_cols = math.ceil(len(labels) / n_rows)

    fig = plt.figure(constrained_layout=True, figsize=(size * 2, size))
    subfigs = fig.subfigures(n_rows, n_cols, wspace=0.07).flatten()

    logger.debug("Plotting confusion matrices")
    for i, (subfig, cm) in enumerate(zip(subfigs, confusion_matrices)):
        ax1, ax2 = subfig.subplots(1, 2, sharey=True)

        precision_cm = cm / cm.sum(axis=0, keepdims=True)
        recall_cm = cm / cm.sum(axis=1, keepdims=True)

        current_label = f"{labels[i]} ({label_totals[i]})"
        rest_label = f"rest ({sum(label_totals) - label_totals[i]})"

        label_names = [rest_label, current_label]
        precision_disp = ConfusionMatrixDisplay(
            confusion_matrix=precision_cm, display_labels=label_names
        )
        recall_disp = ConfusionMatrixDisplay(
            confusion_matrix=recall_cm, display_labels=label_names
        )

        logger.debug(f"Plotting for {labels[i]}")
        recall_disp.plot(cmap="Blues", ax=ax1, xticks_rotation=30, colorbar=False)
        precision_disp.plot(cmap="Blues", ax=ax2, xticks_rotation=30, colorbar=False)

        plt.setp(
            ax1.get_xticklabels(),
            rotation=30,  # type:ignore
            horizontalalignment="right",
        )
        plt.setp(
            ax2.get_xticklabels(),
            rotation=30,  # type:ignore
            horizontalalignment="right",
        )

        ax1.set_title(f"Recall")
        ax2.set_title(f"Precision")

        subfig.suptitle(f"CM for all vs {labels[i]}")

    figure_name = filename if filename is not None else "confusion_matrix"
    figure_title = title if title is not None else ""

    plt.suptitle(figure_title, wrap=True)

    os.makedirs(f"{save_path}/cm/", exist_ok=True)
    plt.savefig(f"{save_path}/cm/{figure_name}.png")
    plt.close()
