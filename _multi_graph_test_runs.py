from copy import deepcopy

import torch
import torch.nn.functional as F
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader as TorchLoader
from torch.utils.data import Dataset
from torch_geometric.data import DataLoader as GeometricLoader

from src.models.ac_gnn import ACGNN, ACGNNNoInput
from src.models.mlp import MLP
from src.run_logic import seed_everything
from temp_chem import MoleculeNet

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


def flatten_graphs(data):
    new_dataset = []
    new_targets = []
    for _data in data:
        new_dataset.extend(_data.x)
        new_targets.extend(_data.y)

    return torch.stack(new_dataset), torch.stack(new_targets)


class TensorDataset(Dataset):
    def __init__(self, _x, _y):
        self._x = _x
        self._y = _y

    def __getitem__(self, i):
        return self._x[i], self._y[i]

    def __len__(self):
        return len(self._x)


def train_mlp(use_data, use_model):
    use_model.train()
    optimizer.zero_grad()
    for x, y in use_data:
        x = x.to("cuda")
        y = y.to("cuda")
        output = use_model(x)

        new_y = F.one_hot(y).float()
        F.binary_cross_entropy_with_logits(output, new_y).backward()
        optimizer.step()


def train_gnn(use_data, use_model):
    use_model.train()
    optimizer.zero_grad()
    for data in use_data:
        data = data.to("cuda")
        output = use_model(
            x=data.x,
            edge_index=data.edge_index,
            batch=None,
        )

        new_y = F.one_hot(data.y).float()
        F.binary_cross_entropy_with_logits(output, new_y).backward()
        optimizer.step()


def test_mlp(use_data, use_model):
    use_model.eval()

    logits = []
    targets = []
    for x, y in use_data:
        x = x.to("cuda")
        output = use_model(x)

        logits.append(output.detach().cpu())
        targets.append(y.cpu())

    logits = torch.cat(logits)
    pred = logits.max(1)[1]
    target = torch.cat(targets)

    acc = pred.eq(target).float().mean().item()

    target_ = target.detach().cpu().numpy()
    pred_ = pred.detach().cpu().numpy()

    precision = precision_score(target_, pred_)
    recall = recall_score(target_, pred_)
    f1 = f1_score(target_, pred_)

    return pred, target, acc, precision, recall, f1


def test_gnn(use_data, use_model):
    use_model.eval()

    logits = []
    targets = []
    for data in use_data:
        data = data.to("cuda")
        output = use_model(
            x=data.x,
            edge_index=data.edge_index,
            edge_attr=data.edge_attr,
            batch=None,
        )

        logits.append(output.detach().cpu())
        targets.append(data.y.cpu())

    logits = torch.cat(logits)
    pred = logits.max(1)[1]
    target = torch.cat(targets)

    acc = pred.eq(target).float().mean().item()

    target_ = target.detach().cpu().numpy()
    pred_ = pred.detach().cpu().numpy()

    precision = precision_score(target_, pred_)
    recall = recall_score(target_, pred_)
    f1 = f1_score(target_, pred_)

    return pred, target, acc, precision, recall, f1


write_model = True
use_mlp = False
dataset = MoleculeNet("./data/chem", "esol")

usable_set = [*dataset]
train_set, test_set = train_test_split(
    usable_set, test_size=0.2, random_state=42, shuffle=True
)
train_set, val_set = train_test_split(
    train_set, test_size=0.25, random_state=42, shuffle=True
)

seed_everything(42)

# ---- MLP -----
# model = MyMLP(
#     num_layers=2,
#     input_dim=dataset.num_features,
#     hidden_dim=8,
#     output_dim=dataset.num_classes,
#     use_batch_norm=True,
# )
# train_set = TorchLoader(
#     TensorDataset(*flatten_graphs(train_set)), batch_size=128, shuffle=True
# )
# val_set = TorchLoader(
#     TensorDataset(*flatten_graphs(val_set)), batch_size=512, shuffle=False
# )
# test_set = TorchLoader(
#     TensorDataset(*flatten_graphs(test_set)), batch_size=512, shuffle=False
# )
# use_mlp = True


# ---- GNN -----
model = ACGNN(
    input_dim=dataset.num_features,
    hidden_dim=8,
    output_dim=dataset.num_classes,
    aggregate_type="add",
    combine_type="identity",
    num_layers=2,
    combine_layers=1,
    mlp_layers=1,
    task="node",
    use_batch_norm=False,
)
train_set = GeometricLoader(train_set, batch_size=128, shuffle=True)
val_set = GeometricLoader(val_set, batch_size=512, shuffle=False)
test_set = GeometricLoader(test_set, batch_size=512, shuffle=False)


device = torch.device("cuda")
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.005)

train = train_mlp if use_mlp else train_gnn
test = test_mlp if use_mlp else test_gnn

good_models = []

best_val_acc = (
    test_acc
) = (
    val_precision
) = test_precision = val_recall = test_recall = val_f1_score = test_f1_score = 0
log = "Epoch: {:03d}, Train: ({:.4f}, {:.4f}, {:.4f}, {:.4f}), Val: ({:.4f}, {:.4f}, {:.4f}, {:.4f}), Test: ({:.4f}, {:.4f}, {:.4f}, {:.4f})"
for epoch in range(1, 300):
    _accs, _pres, _recs, _f1s = [], [], [], []

    train(train_set, model)

    for data_set in [train_set, val_set, test_set]:
        *_, _acc, _pre, _rec, _f1 = test(data_set, model)
        _accs.append(_acc)
        _pres.append(_pre)
        _recs.append(_rec)
        _f1s.append(_f1)

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

for e, goods in good_models[-1:]:  # only the best
    model.load_state_dict(goods)
    model.eval()
    pred, target, *_ = test(test_set, model)

    pred_ = pred.detach().cpu().numpy()

    report = classification_report(target.cpu().numpy(), pred_)

    print("data from epoch", e)
    print(report, "\n")

    if write_model:
        torch.save([goods], "trained_molecules.pt")
