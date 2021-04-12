import os
from typing import List, Optional

import numpy as np
import torch
import torch.nn
import torch.nn.functional as F
import torch.optim
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import TruncatedSVD
from torch.utils.data.dataloader import DataLoader
from torch_geometric import datasets
from torch_geometric.data.data import Data

from src.models.mlp import MLP
from src.run_logic import seed_everything

seed_everything(42)

model_path = os.path.join("data", "cora_data")
os.makedirs(model_path, exist_ok=True)


class Encoder(MLP):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_layers: Optional[List[int]],
        use_batch_norm: bool,
        **kwargs,
    ):
        super().__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_layers=hidden_layers,
            use_batch_norm=use_batch_norm,
            num_layers=-1,
            hidden_dim=-1,
            **kwargs,
        )

    def forward(self, x, **kwargs):
        return super().forward(x)


class Decoder(MLP):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_layers: Optional[List[int]],
        use_batch_norm: bool,
        **kwargs,
    ):
        super().__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_layers=hidden_layers,
            use_batch_norm=use_batch_norm,
            num_layers=-1,
            hidden_dim=-1,
            **kwargs,
        )

    def forward(self, x, **kwargs):
        return super().forward(x)


class AutoEncoder(torch.nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x, **kwargs):
        x = self.encoder(x=x)
        x = self.decoder(x=x)

        return x

    def predict(self, x):
        return self.encoder(x=x)


class WeightedLoss(torch.nn.Module):
    def __init__(self, p):
        super().__init__()
        self.p = p

    def forward(self, x, y):
        mse_loss = torch.mean((y - x) ** 2)

        non_zero_inds = (y == 1).nonzero(as_tuple=True)
        x_ = x[non_zero_inds]
        y_ = y[non_zero_inds]

        extra = torch.mean((y_ - x_) ** 2) * self.p

        return mse_loss + extra


def train_autoencoder(autoencoder, data, epoch, batch_size, loss_p=1):
    model = autoencoder.cuda()
    criterion = WeightedLoss(p=loss_p)
    # criterion = torch.nn.MSELoss(reduction="sum")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    dataloader = DataLoader(data, batch_size=batch_size, shuffle=True)

    for epoch in range(epoch):
        epoch_loss = []
        for epoch_data in dataloader:
            x = epoch_data.cuda()

            output = model(x=x)
            loss = criterion(output, x)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss.append(loss.detach().cpu().numpy())

        average_loss = np.mean(epoch_loss)

        print(f"epoch {epoch} with loss {average_loss}")

    return model


def dimension_reduction(original_data, dim):
    _x = original_data.x

    feats = TruncatedSVD(n_components=dim).fit_transform(_x)
    features = torch.from_numpy(feats)

    new_dataset = Data(
        x=features.float(),
        edge_index=original_data.edge_index,
        test_mask=original_data.test_mask,
        train_mask=original_data.train_mask,
        val_mask=original_data.val_mask,
        y=original_data.y,
        edge_attr=original_data.edge_attr,
    )

    return new_dataset


def clusterize(original_data, new_features, cluster_name="agglomerative"):
    if cluster_name == "agglomerative":
        clusters = AgglomerativeClustering(n_clusters=4, linkage="ward").fit_predict(
            new_features
        )
    elif cluster_name == "kmeans":
        clusters = np.array(
            KMeans(n_clusters=4).fit_predict(new_features), dtype=np.int64
        )
    else:
        raise ValueError()
    features = torch.from_numpy(clusters)
    feats = F.one_hot(features)

    new_dataset = Data(
        x=feats.float(),
        edge_index=original_data.edge_index,
        test_mask=original_data.test_mask,
        train_mask=original_data.train_mask,
        val_mask=original_data.val_mask,
        y=original_data.y,
        edge_attr=original_data.edge_attr,
    )

    return new_dataset


dataset = datasets.Planetoid("data/Cora", "Cora")
num_classes = dataset.num_classes
num_features = dataset.num_features
dataset = dataset[0]

reductor = "original"
method_name = "kmeans"

if reductor == "autoencoder":
    _embedded_dim = 32
    _encoder = Encoder(
        input_dim=num_features,
        output_dim=_embedded_dim,
        hidden_layers=[512],
        use_batch_norm=True,
    )
    _decoder = Decoder(
        input_dim=_embedded_dim,
        output_dim=num_features,
        hidden_layers=[512],
        use_batch_norm=True,
    )
    autoencoder_model = AutoEncoder(encoder=_encoder, decoder=_decoder)

    train_autoencoder(
        autoencoder=autoencoder_model,
        data=dataset.x,
        epoch=500,
        batch_size=128,
        loss_p=0.7,
    )

    with torch.no_grad():
        reduced_data = autoencoder_model.predict(x=dataset.x.cuda()).detach().cpu()

    reconstructed_data = autoencoder_model(x=dataset.x.cuda()).cpu()
    nonzero = dataset.x == 1

    bad_1s = (reconstructed_data[nonzero] < 0.5).sum()
    bad_0s = (reconstructed_data[~nonzero] >= 0.5).sum()

    print(bad_1s / nonzero.sum())
    print(bad_0s / (~nonzero).sum())

    new_cora = clusterize(
        original_data=dataset, new_features=reduced_data, cluster_name=method_name
    )

    torch.save(
        new_cora,
        os.path.join(
            model_path,
            f"processed_cora_ae_h{_embedded_dim}-mid512-p07_{method_name}.pt",
        ),
    )
elif reductor == "dimension":
    reduced_dim = 500
    reduced_data = dimension_reduction(original_data=dataset, dim=reduced_dim)

    new_cora = clusterize(
        original_data=dataset, new_features=reduced_data.x, cluster_name=method_name
    )

    print(new_cora)

    torch.save(
        new_cora,
        os.path.join(model_path, f"processed_cora_svd_{method_name}_d{reduced_dim}.pt"),
    )

else:
    new_cora = clusterize(
        original_data=dataset, new_features=dataset.x, cluster_name=method_name
    )

    print(new_cora)

    torch.save(
        new_cora,
        os.path.join(model_path, f"original_reduced_cora.pt"),
    )
