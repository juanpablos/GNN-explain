import os

import torch
from torch_geometric.data import InMemoryDataset

try:
    from gnn.convert import gnn2data
except BaseException:
    import sys
    sys.path.insert(1, os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    from gnn.convert import gnn2data


class GNNGraphDataset(InMemoryDataset):
    def __init__(self, root: str):
        super(GNNGraphDataset, self).__init__(root, None, None)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return ['graph_gnns.pt']

    def process(self):
        data_list = []

        counter = 0
        for file in os.listdir(self.root):
            if 'acgnn' in file and 'stat' not in file:
                file_path = os.path.join(self.root, file)

                print(file_path)

                networks = torch.load(file_path)
                for network in networks:
                    data_list.append(gnn2data(network))

                counter += 1

                if counter >= 10:
                    break

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        # torch.save(data_list, self.processed_paths[0])


if __name__ == "__main__":
    datas = GNNGraphDataset('./data/gnns/f4034364ea-batch - Copy')
