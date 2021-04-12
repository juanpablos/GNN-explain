import os
import warnings
from copy import deepcopy

import torch
import torch.nn
import torch.nn.functional as F
import torch.optim
import torch_geometric.transforms as T
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from torch_geometric.data.data import Data
from torch_geometric.nn import SplineConv

from src.models.ac_gnn import ACGNN, ACGNNNoInput
from src.models.mlp import MLP
from src.run_logic import seed_everything

seed_everything(42)
cora_data_path = os.path.join("data", "cora_data")
model_path = os.path.join("data", "gnns_v2", "40e65407aa")


class SplineNet(torch.nn.Module):
    def __init__(self, n_features, n_classes, hidden):
        super(SplineNet, self).__init__()
        self.conv1 = SplineConv(n_features, hidden, dim=1, kernel_size=2)
        self.conv2 = SplineConv(hidden, n_classes, dim=1, kernel_size=2)

    def forward(self, x, edge_index, edge_attr, **kwargs):
        x = F.dropout(x, training=self.training)
        x = F.elu(self.conv1(x, edge_index, edge_attr))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index, edge_attr)
        return x


class MyMLP(MLP):
    def __init__(
        self,
        num_layers: int,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        use_batch_norm: bool,
        **kwargs,
    ):
        super().__init__(
            num_layers,
            input_dim,
            hidden_dim,
            output_dim,
            use_batch_norm=use_batch_norm,
            **kwargs,
        )

    def forward(self, x, **kwargs):
        return super().forward(x)


def binarize_target(all_data, to_label):
    _y = all_data.y

    new_y = (_y == to_label).long()

    new_dataset = Data(
        x=all_data.x,
        edge_index=all_data.edge_index,
        test_mask=all_data.test_mask,
        train_mask=all_data.train_mask,
        val_mask=all_data.val_mask,
        y=new_y,
        edge_attr=all_data.edge_attr,
    )

    return new_dataset


def train(use_data, use_model, optimizer, binary):
    use_model.train()
    optimizer.zero_grad()
    output = use_model(
        x=use_data.x,
        edge_index=use_data.edge_index,
        edge_attr=use_data.edge_attr,
        batch=None,
    )

    if binary:
        new_y = F.one_hot(use_data.y).float()
        F.binary_cross_entropy_with_logits(
            output[use_data.train_mask], new_y[use_data.train_mask]
        ).backward()
    else:
        F.cross_entropy(
            output[use_data.train_mask], use_data.y[use_data.train_mask]
        ).backward()
    optimizer.step()


def test(use_data, use_model):
    use_model.eval()
    logits, accs, pre, rec, f1s = (
        use_model(
            x=use_data.x,
            edge_index=use_data.edge_index,
            edge_attr=use_data.edge_attr,
            batch=None,
        ),
        [],
        [],
        [],
        [],
    )
    for _, mask in use_data("train_mask", "val_mask", "test_mask"):
        pred = logits[mask].max(1)[1]
        target = use_data.y[mask]
        acc = pred.eq(target).sum().item() / mask.sum().item()

        target_ = target.detach().cpu().numpy()
        pred_ = pred.detach().cpu().numpy()

        precision = precision_score(target_, pred_)
        recall = recall_score(target_, pred_)
        f1 = f1_score(target_, pred_)

        accs.append(acc)
        pre.append(precision)
        rec.append(recall)
        f1s.append(f1)
    return accs, pre, rec, f1s


def run(model, use_data, binary, log_epoch):
    device = torch.device("cuda")
    model, use_data = model.to(device), use_data.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.005)

    good_models = []

    best_val_acc = (
        test_acc
    ) = (
        val_precision
    ) = test_precision = val_recall = test_recall = val_f1_score = test_f1_score = 0
    log = "Epoch: {:03d}, Train: ({:.4f}, {:.4f}, {:.4f}, {:.4f}), Val: ({:.4f}, {:.4f}, {:.4f}, {:.4f}), Test: ({:.4f}, {:.4f}, {:.4f}, {:.4f})"
    for epoch in range(1, 300):
        train(use_data, model, optimizer, binary)
        _accs, _pres, _recs, _f1s = test(use_data, model)

        (train_acc, val_acc, tmp_test_acc) = _accs
        (train_pre, val_pre, tmp_test_pre) = _pres
        (train_rec, val_rec, tmp_test_rec) = _recs
        (train_f1, val_f1, tmp_test_f1) = _f1s
        if val_f1 > val_f1_score:
            best_val_acc = val_acc
            test_acc = tmp_test_acc

            val_precision = val_pre
            test_precision = tmp_test_pre

            val_recall = val_rec
            test_recall = tmp_test_rec

            val_f1_score = val_f1
            test_f1_score = tmp_test_f1

            good_models.append((epoch, deepcopy(model.state_dict())))

        if log_epoch:
            print(
                log.format(
                    epoch,
                    train_acc,
                    train_pre,
                    train_rec,
                    train_f1,
                    best_val_acc,
                    val_precision,
                    val_recall,
                    val_f1_score,
                    test_acc,
                    test_precision,
                    test_recall,
                    test_f1_score,
                )
            )
    print(
        log.format(
            epoch,
            train_acc,
            train_pre,
            train_rec,
            train_f1,
            best_val_acc,
            val_precision,
            val_recall,
            val_f1_score,
            test_acc,
            test_precision,
            test_recall,
            test_f1_score,
        )
    )

    if good_models:
        return good_models[-1]
    else:
        return -1, deepcopy(model.state_dict())


all_datasets = [
    "original_reduced_cora",
    "processed_cora_svd_kmeans_d500",
    "processed_cora_ae_h32-mid512-p01_agglomerative",
    "processed_cora_ae_h32-mid512-p03_agglomerative",
    "processed_cora_ae_h32-mid512-p07_agglomerative",
    "processed_cora_ae_h32-mid512-p1_agglomerative",
    "processed_cora_ae_h128-mid512-p01_agglomerative",
    "processed_cora_ae_h128-mid512-p03_agglomerative",
    "processed_cora_ae_h128-mid512-p07_agglomerative",
    "processed_cora_ae_h128-mid512-p1_agglomerative",
]

warnings.catch_warnings().__enter__()
warnings.filterwarnings("ignore")

for reduced_dataset in all_datasets:
    print(reduced_dataset)
    n_models = 10
    should_log = False
    log_results = True

    data = torch.load(os.path.join(cora_data_path, f"{reduced_dataset}.pt"))

    # from torch_geometric import datasets
    # dataset = datasets.Planetoid("data/Cora", "Cora", transform=T.TargetIndegree())[0]

    print("num features", data.num_features)

    all_models = []
    reports = []
    for m in range(n_models):
        print(f"{m+1}/{n_models}")
        _model = ACGNN(
            input_dim=data.num_features,
            hidden_dim=8,
            output_dim=2,
            aggregate_type="add",
            combine_type="identity",
            num_layers=2,
            combine_layers=1,
            mlp_layers=1,
            task="node",
            use_batch_norm=False,
        )

        # _model = MyMLP(
        #     num_layers=2,
        #     input_dim=data.num_features,
        #     hidden_dim=8,
        #     output_dim=2,
        #     use_batch_norm=True,
        # )

        # _model = SplineNet(n_features=data.num_features, n_classes=2, hidden=8)

        epoch, best_model = run(
            model=_model, use_data=data, binary=True, log_epoch=should_log
        )

        if should_log or log_results:
            test_mask = data.test_mask
            target_data = data.y[test_mask]

            _model.load_state_dict(best_model)
            _model.eval()
            pred = _model(
                x=data.x,
                edge_index=data.edge_index,
                edge_attr=data.edge_attr,
                batch=None,
            )[test_mask].max(1)[1]

            target_ = target_data.detach().cpu().numpy()
            pred_ = pred.detach().cpu().numpy()

            report = classification_report(target_, pred_)

            print(reduced_dataset)
            print("data from epoch", epoch)
            print(report, "\n")

            reports.append(f"data from epoch, {epoch}\n\n{str(report)}")

        all_models.append(best_model)

    torch.save(all_models, os.path.join(model_path, f"{reduced_dataset}.pt"))

    if should_log or log_results:
        with open(os.path.join("results", "cora", f"{reduced_dataset}.txt"), "w") as f:
            for report in reports:
                f.write(report)
