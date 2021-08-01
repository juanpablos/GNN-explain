import umap
import umap.plot
import matplotlib.pyplot as plt
import os


def plot_embedding_2d(
    embedding,
    labels,
    save_path: str,
    filename: str,
    seed=None,
):
    fig, ax = plt.subplots(figsize=(20, 10))

    mapper = umap.UMAP(random_state=seed).fit(embedding)
    umap.plot.points(mapper, labels=labels, ax=ax)

    plt.tight_layout()
    os.makedirs(f"{save_path}/embedding/", exist_ok=True)
    plt.savefig(f"{save_path}/embedding/{filename}.png")
    plt.close()
