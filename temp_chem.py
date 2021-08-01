import os
import os.path as osp
import re

import torch
import torch.nn.functional as F
from torch_geometric.data import Data, InMemoryDataset, download_url, extract_zip

try:
    from rdkit import Chem
except ImportError:
    Chem = None

x_map = {
    "atomic_num": list(range(0, 119)),
    "chirality": [
        "CHI_UNSPECIFIED",
        "CHI_TETRAHEDRAL_CW",
        "CHI_TETRAHEDRAL_CCW",
        "CHI_OTHER",
    ],
    "degree": list(range(0, 11)),
    "formal_charge": list(range(-5, 7)),
    "num_hs": list(range(0, 9)),
    "num_radical_electrons": list(range(0, 5)),
    "hybridization": [
        "UNSPECIFIED",
        "S",
        "SP",
        "SP2",
        "SP3",
        "SP3D",
        "SP3D2",
        "OTHER",
    ],
    "is_aromatic": [False, True],
    "is_in_ring": [False, True],
}

e_map = {
    "bond_type": [
        "misc",
        "SINGLE",
        "DOUBLE",
        "TRIPLE",
        "AROMATIC",
    ],
    "stereo": [
        "STEREONONE",
        "STEREOZ",
        "STEREOE",
        "STEREOCIS",
        "STEREOTRANS",
        "STEREOANY",
    ],
    "is_conjugated": [False, True],
}


class MoleculeNet(InMemoryDataset):
    r"""The `MoleculeNet <http://moleculenet.ai/datasets-1>`_ benchmark
    collection  from the `"MoleculeNet: A Benchmark for Molecular Machine
    Learning" <https://arxiv.org/abs/1703.00564>`_ paper, containing datasets
    from physical chemistry, biophysics and physiology.
    All datasets come with the additional node and edge features introduced by
    the `Open Graph Benchmark <https://ogb.stanford.edu/docs/graphprop/>`_.
    Args:
        root (string): Root directory where the dataset should be saved.
        name (string): The name of the dataset (:obj:`"ESOL"`,
            :obj:`"FreeSolv"`, :obj:`"Lipo"`, :obj:`"PCBA"`, :obj:`"MUV"`,
            :obj:`"HIV"`, :obj:`"BACE"`, :obj:`"BBPB"`, :obj:`"Tox21"`,
            :obj:`"ToxCast"`, :obj:`"SIDER"`, :obj:`"ClinTox"`).
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        pre_filter (callable, optional): A function that takes in an
            :obj:`torch_geometric.data.Data` object and returns a boolean
            value, indicating whether the data object should be included in the
            final dataset. (default: :obj:`None`)
    """

    url = (
        "https://s3-us-west-1.amazonaws.com/deepchem.io/datasets/"
        "molnet_publish/{}.zip"
    )

    # Format: name: [display_name, url_name, csv_name, smiles_idx, y_idx]
    names = {
        "esol": ["ESOL", "ESOL", "delaney-processed", -1, -2],
        "freesolv": ["FreeSolv", "FreeSolv", "SAMPL", 1, 2],
        "lipo": ["Lipophilicity", "lipophilicity", "Lipophilicity", 2, 1],
        "pcba": ["PCBA", "pcba", "pcba", -1, slice(0, 128)],
        "muv": ["MUV", "muv", "muv", -1, slice(0, 17)],
        "hiv": ["HIV", "hiv", "HIV", 0, -1],
        "bace": ["BACE", "bace", "bace", 0, 2],
        "bbbp": ["BBPB", "bbbp", "BBBP", -1, -2],
        "tox21": ["Tox21", "tox21", "tox21", -1, slice(0, 12)],
        "toxcast": ["ToxCast", "toxcast", "toxcast_data", 0, slice(1, 618)],
        "sider": ["SIDER", "sider", "sider", 0, slice(1, 28)],
        "clintox": ["ClinTox", "clintox", "clintox", 0, slice(1, 3)],
    }

    def __init__(self, root, name, transform=None, pre_transform=None, pre_filter=None):

        if Chem is None:
            raise ImportError("`MoleculeNet` requires `rdkit`.")

        self.name = name.lower()
        assert self.name in self.names.keys()
        super(MoleculeNet, self).__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        return osp.join(self.root, self.name, "raw")

    @property
    def processed_dir(self):
        return osp.join(self.root, self.name, "processed")

    @property
    def raw_file_names(self):
        r"""The name of the files to find in the :obj:`self.raw_dir` folder in
        order to skip the download."""
        return [f"{self.names[self.name][2]}.csv"]

    @property
    def processed_file_names(self):
        r"""The name of the files to find in the :obj:`self.processed_dir`
        folder in order to skip the processing."""
        return [f"{self.name}.pt"]

    def download(self):
        url = self.url.format(self.names[self.name][1])
        path = download_url(url, self.raw_dir)
        extract_zip(path, self.raw_dir)
        os.unlink(path)

    def process(self):
        with open(self.raw_paths[0], "r") as f:
            dataset = f.read().split("\n")[1:-1]
            dataset = [x for x in dataset if len(x) > 0]  # Filter empty lines.

        data_list = []
        for line in dataset:
            line = re.sub(r"\".*\"", "", line)  # Replace ".*" strings.
            line = line.split(",")

            smiles = line[self.names[self.name][3]]

            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue

            feature_list = []
            target_list = []
            for atom in mol.GetAtoms():
                target_list.append(x_map["atomic_num"].index(atom.GetAtomicNum()))
                # x.append(x_map['chirality'].index(str(atom.GetChiralTag())))
                # x.append(x_map['degree'].index(atom.GetTotalDegree()))
                # x.append(atom.GetFormalCharge())
                # x.append(atom.GetExplicitValence())
                # feature_list.append(x_map["num_hs"].index(atom.GetTotalNumHs()))
                feature_list.append(
                    x_map["num_radical_electrons"].index(atom.GetNumRadicalElectrons())
                )
                # x.append(x_map['hybridization'].index(
                #     str(atom.GetHybridization())))
                # y.append(x_map["is_aromatic"].index(atom.GetIsAromatic()))
                # target_list.append(x_map["is_in_ring"].index(atom.IsInRing()))

            features = torch.tensor(feature_list, dtype=torch.long).clip(0, 3)
            features = F.one_hot(features, num_classes=4).float()

            # features = torch.tensor(feature_list, dtype=torch.long).view(-1, 1)
            # target = torch.tensor(target_list, dtype=torch.long)
            target = (torch.tensor(target_list, dtype=torch.long) == 6).long()

            edge_indices = []
            for bond in mol.GetBonds():
                i = bond.GetBeginAtomIdx()
                j = bond.GetEndAtomIdx()
                edge_indices += [[i, j], [j, i]]

            edge_index = torch.tensor(edge_indices)
            edge_index = edge_index.t().to(torch.long).view(2, -1)

            # Sort indices.
            if edge_index.numel() > 0:
                perm = (edge_index[0] * features.size(0) + edge_index[1]).argsort()
                edge_index = edge_index[:, perm]

            data = Data(x=features, edge_index=edge_index, y=target)

            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)

        torch.save(self.collate(data_list), self.processed_paths[0])

    def __repr__(self):
        return "{}({})".format(self.names[self.name][0], len(self))


import numpy as np

if __name__ == "__main__":
    # esol, freesolv, lipo, pcba, muv, hiv, bace, bbbp, tox21, toxcast, sider, clintox

    d = MoleculeNet("./data/chem3", "esol")
    print(d)
    print(d[0])
    print(d[100].x.squeeze())
    print(d[100].y.squeeze())

    all_x = torch.cat([d[i].x for i in range(len(d))])
    all_y = torch.cat([d[i].y.view(-1) for i in range(len(d)) if d[i].y.dim() > 0])

    y_unique, y_counts = np.unique(all_y.numpy(), return_counts=True)
    print("feature count", all_x.sum(dim=0))
    print(f"[{all_y.min().item()}, {all_y.max().item()}]", y_unique, y_counts)
