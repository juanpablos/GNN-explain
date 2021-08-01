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
    umap.plot.points(mapper, labels=labels, ax=ax, width=1600, height=1000)

    embedding_path = os.path.join(save_path, "embedding")
    os.makedirs(embedding_path, exist_ok=True)
    plt.savefig(os.path.join(embedding_path, f"{filename}.png"))
    plt.close()
