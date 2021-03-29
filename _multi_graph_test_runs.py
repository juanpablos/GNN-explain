from copy import deepcopy

import torch
import torch.nn.functional as F
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from torch_geometric import datasets
from torch_geometric.data.data import Data
import torch_geometric.transforms as T
from torch_geometric.nn import SplineConv
from src.models.ac_gnn import ACGNNNoInput, ACGNN
from src.models.mlp import MLP
from src.run_logic import seed_everything
from .temp_chem import MoleculeNet

seed_everything(42)


class MyMLP(MLP):
    def __init__(
        self,
        num_layers: int,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        use_batch_norm: bool,
        **kwargs
    ):
        super().__init__(
            num_layers,
            input_dim,
            hidden_dim,
            output_dim,
            use_batch_norm=use_batch_norm,
            **kwargs
        )

    def forward(self, x, **kwargs):
        return super().forward(x)

def train(use_data, use_model):
    use_model.train()
    optimizer.zero_grad()
    output = use_model(
        x=use_data.x,
        edge_index=use_data.edge_index,
        batch=None,
    )

    new_y = F.one_hot(use_data.y).float()
    F.binary_cross_entropy_with_logits(
        output[use_data.train_mask], new_y[use_data.train_mask]
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


write_model = False
dataset = datasets.Planetoid("delete/hey/Cora", "Cora", transform=T.TargetIndegree())
single_data = dataset[0]
num_classes = dataset.num_classes
binary = False

single_data = reduce_features(single_data, 4)

print("num features", single_data.num_features, "num classes", num_classes)

seed_everything(42)
data = binarize_target(single_data, 2)
num_classes = 2
binary = True

# model = ACGNNNoInput(
#     input_dim=data.num_features,
#     hidden_dim=8,
#     output_dim=num_classes,
#     aggregate_type="add",
#     combine_type="identity",
#     num_layers=2,
#     combine_layers=1,
#     mlp_layers=1,
#     task="node",
#     use_batch_norm=False,
# )

# model = MyMLP(
#     num_layers=2,
#     input_dim=data.num_features,
#     hidden_dim=8,
#     output_dim=num_classes,
#     use_batch_norm=True,
# )

model = SplineNet(n_features=data.num_features, n_classes=num_classes, hidden=8)

device = torch.device("cuda")
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model, data = model.to(device), data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.005)

good_models = []

best_val_acc = (
    test_acc
) = (
    val_precision
) = test_precision = val_recall = test_recall = val_f1_score = test_f1_score = 0
log = "Epoch: {:03d}, Train: ({:.4f}, {:.4f}, {:.4f}, {:.4f}), Val: ({:.4f}, {:.4f}, {:.4f}, {:.4f}), Test: ({:.4f}, {:.4f}, {:.4f}, {:.4f})"
for epoch in range(1, 300):
    train(data, model)
    _accs, _pres, _recs, _f1s = test(data, model)

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

    # print(log.format(epoch,
    #                  train_acc, train_pre, train_rec, train_f1,
    #                  best_val_acc, val_precision, val_recall, val_f1_score,
    # test_acc, test_precision, test_recall, test_f1_score))
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

test_mask = data.test_mask
target_data = data.y[test_mask]
for e, goods in good_models[-1:]:
    model.load_state_dict(goods)
    model.eval()
    pred = model(
        x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr, batch=None
    )[test_mask].max(1)[1]

    target_ = target_data.detach().cpu().numpy()
    pred_ = pred.detach().cpu().numpy()

    report = classification_report(target_, pred_)

    print("data from epoch", e)
    print(report, "\n")

    if write_model:
        data = data.to("cpu")
        torch.save([goods], "trained_cora.pt")
        torch.save(data, "reduced_cora.pt")
