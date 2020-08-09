import logging

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

logger = logging.getLogger(__name__)


def plot_confusion_matrix(
        y,
        y_pred,
        save_path,
        *,
        labels=None,
        each_label: int = None):
    logger.info(f"Calculating confusion matrix")

    normalize = None
    if each_label is not None:
        normalize = "true"

    matrix_og = confusion_matrix(y, y_pred, normalize=normalize)

    size = 10 if len(labels) < 10 else len(labels) * 1.25
    fig, ax = plt.subplots(figsize=(size, size))

    disp = ConfusionMatrixDisplay(matrix_og, display_labels=labels)

    logger.debug("Plotting confusion matrix")
    disp.plot(cmap="Blues", ax=ax, xticks_rotation=30)

    plt.setp(ax.get_xticklabels(), rotation=30,  # type:ignore
             horizontalalignment='right')
    if each_label is None:
        plt.title("Confusion matrix")
    else:
        plt.title(f"Confusion matrix: {each_label} elements each class")
        disp.im_.set_clim(0, 1)
    plt.tight_layout()
    plt.savefig(f"{save_path}/confusion_matrix.png")
    plt.close()
