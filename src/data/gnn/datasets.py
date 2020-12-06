import os
from typing import List

import torch
from torch_geometric.data import InMemoryDataset

try:
    from .convert import gnn2data
except BaseException:
    import sys
    sys.path.insert(1,
                    os.path.dirname(
                        os.path.dirname(
                            os.path.abspath(__file__))))
    from gnn.convert import gnn2data  # type: ignore


class GNNGraphDataset(InMemoryDataset):
    def __init__(self, root: str):
        super(GNNGraphDataset, self).__init__(root, None, None)
        self.data, self.slices, self.processed_filenames = torch.load(
            self.processed_paths[0])

    @property
    def processed_file_names(self):
        return ['graph_gnns.pt']

    def process(self):
        data_list = []
        processed_filenames = []

        for file in os.listdir(self.root):
            if 'acgnn' in file and 'stat' not in file:
                file_path = os.path.join(self.root, file)

                print(file_path)
                networks = torch.load(file_path)
                for network in networks:
                    data_list.append(gnn2data(network))

                processed_filenames.append(file)

        data, slices = self.collate(data_list)
        torch.save((data, slices, set(processed_filenames)),
                   self.processed_paths[0])

    def update_processed(self, new_files: List[str]):
        data_list = [self.data]

        repeated_files = [
            file for file in new_files if file in self.processed_filenames]

        if repeated_files:
            raise ValueError(
                f'There are files already loaded: {repeated_files}')

        for file in new_files:
            file_path = os.path.join(self.root, file)
            print(file_path)

            networks = torch.load(file_path)
            for network in networks:
                data_list.append(gnn2data(network))

            self.processed_filenames.append(file)

        self.data, slices = self.collate(data_list)

        for key, values in self.slices.items():
            self.slices[key] = torch.cat(
                (values, slices[key][2:]), dim=0)  # type: ignore

        torch.save(
            (self.data,
             self.slices,
             self.processed_filenames),
            self.processed_paths[0])


if __name__ == "__main__":
    datas = GNNGraphDataset('../../../data/gnns/f4034364ea-batch')
