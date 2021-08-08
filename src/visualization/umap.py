import os
from typing import Dict

import matplotlib.pyplot as plt
import umap
import umap.plot


def plot_embedding_2d(
    embedding,
    labels,
    labels_categorical_mapping: Dict[int, str],
    save_path: str,
    filename: str,
    seed=None,
):
    fig, ax = plt.subplots(figsize=(20, 10))

    mapper = umap.UMAP(random_state=seed, low_memory=False).fit(embedding)

    string_labels = [labels_categorical_mapping[label] for label in labels]

    umap.plot.points(mapper, labels=string_labels, ax=ax, width=1600, height=1000)

    embedding_path = os.path.join(save_path, "embedding")
    os.makedirs(embedding_path, exist_ok=True)
    plt.savefig(os.path.join(embedding_path, f"{filename}.png"))
    plt.close()
