import logging

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

logger = logging.getLogger(__name__)


def plot_confusion_matrix(y, y_pred, save_path, *, labels=None):
    logger.info("Calculating confusion matrix")
    matrix_og = confusion_matrix(y, y_pred, normalize=None)
    # matrix_norm = confusion_matrix(y, y_pred, normalize="true")

    fig, ax = plt.subplots()

    disp_og = ConfusionMatrixDisplay(matrix_og, display_labels=labels)
    # disp_norm = ConfusionMatrixDisplay(matrix_norm, display_labels=labels)

    logger.debug("Plotting confusin matrix")
    disp_og.plot(cmap="Blues", ax=ax, xticks_rotation=45)
    # disp_norm.plot(cmap="Blues", ax=ax2, xticks_rotation=45)

    plt.title("Confusion matrix")
    plt.tight_layout()
    plt.savefig(f"{save_path}/confusion_matrix.png")
    plt.close()
