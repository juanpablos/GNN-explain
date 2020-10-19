
import bisect
import logging
from typing import Dict, List, Sequence

import torch

from src.data.datasets import NetworkDataset
from src.graphs.foc import Element

logger = logging.getLogger(__name__)


class NetworkDatasetCollectionWrapper:
    def __init__(self, datasets: Sequence[NetworkDataset]):
        if len(datasets) < 1:
            raise ValueError("datasets cannot be an empty sequence")

        self.formulas: List[Element] = []
        for d in datasets:
            assert isinstance(
                d, NetworkDataset), "elements should be NetworkDatasets"
            self.formulas.append(d.formula)

        self.cumulative_sizes = self.cumsum(datasets)

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, index: int) -> Element:
        dataset_index = bisect.bisect_right(self.cumulative_sizes, index)
        return self.formulas[dataset_index]

    @staticmethod
    def cumsum(sequence: Sequence):
        seq: List[int] = []
        curr = 0
        for s in sequence:
            l = len(s)
            seq.append(curr + l)
            curr += l
        return seq


class AggregatedNetworkDataset:
    def __init__(self, file_path: str):
        """
        format:
        {
            formula_hash: {
                file: file_path,
                data: tensor_data
            }
        }
        """
        logger.debug("Loading formulas")
        self.formulas = torch.load(file_path)

    def __getitem__(self, formula: str) -> torch.Tensor:
        return self.formulas[formula]["data"]

    def available_formulas(self) -> Dict[str, str]:
        return {k: v["file"] for k, v in self.formulas.items()}
