import logging
import os

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

logger = logging.getLogger(__name__)


def plot_confusion_matrix(
        y,
        y_pred,
        save_path: str,
        filename: str = None,
        title: str = None,
        *,
        labels=None,
        normalize_cm: bool = True):
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

    plt.setp(ax.get_xticklabels(), rotation=30,  # type:ignore
             horizontalalignment='right')
    if normalize_cm:
        disp.im_.set_clim(0, 1)
    plt.tight_layout()

    figure_name = filename if filename is not None else "confusion_matrix"
    figure_title = title if title is not None else ""

    plt.title(figure_title, wrap=True)

    os.makedirs(f"{save_path}/cm/", exist_ok=True)
    plt.savefig(f"{save_path}/cm/{figure_name}.png")
    plt.close()
