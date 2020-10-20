import bisect
import logging
import random
from typing import Dict, List, Optional, Sequence, Tuple, overload

import torch
from torch import Tensor

from src.data.datasets import NetworkDataset
from src.generate_graphs import graph_object_stream
from src.graphs.foc import FOC, Element

logger = logging.getLogger(__name__)


def cumsum(sequence: Sequence):
    seq: List[int] = []
    curr = 0
    for s in sequence:
        l = len(s)
        seq.append(curr + l)
        curr += l
    return seq


class NetworkDatasetCollectionWrapper:
    def __init__(self, datasets: Sequence[NetworkDataset]):
        if len(datasets) < 1:
            raise ValueError("datasets cannot be an empty sequence")

        self.formulas: List[Element] = []
        for d in datasets:
            assert isinstance(
                d, NetworkDataset), "elements should be NetworkDatasets"
            self.formulas.append(d.formula)

        self.cumulative_sizes = cumsum(datasets)

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, index: int) -> Element:
        dataset_index = bisect.bisect_right(self.cumulative_sizes, index)
        return self.formulas[dataset_index]


class FormulaAppliedDatasetWrapper:
    def __init__(
            self,
            datasets: Sequence[NetworkDataset],
            configs: List[Dict[str, int]],
            n_properties: int = 4,
            seed: int = 0):
        if len(datasets) < 1:
            raise ValueError("datasets cannot be an empty sequence")

        self.formulas: List[FOC] = []
        self.applied: List[Tensor] = []

        for d in datasets:
            assert isinstance(
                d, NetworkDataset), "elements should be NetworkDatasets"
            self.formulas.append(FOC(d.formula))

        self.cumulative_sizes = cumsum(datasets)

        self.graphs = []
        self._create_graphs(
            configs=configs,
            n_properties=n_properties,
            seed=seed)

        self.n_graphs = len(self.graphs)
        self.n_nodes = [len(g) for g in self.graphs]

        self._run_formulas()

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, index: int) -> Tuple[Tensor, int]:
        dataset_index = bisect.bisect_right(self.cumulative_sizes, index)
        return self.applied[dataset_index], self.n_nodes[dataset_index]

    def _create_graphs(
            self,
            configs: List[Dict[str, int]],
            n_properties: int,
            seed: int = 0):

        logger.debug("Creating graphs")

        rand = random.Random(seed)

        for config in configs:
            # configs is a list of config with each config:
            # m: number of edges in the graph
            # min_nodes
            # max_nodes
            # n_graphs
            m = config["m"]
            min_nodes = config["min_nodes"]
            max_nodes = config["max_nodes"]
            n_graphs = config["n_graphs"]

            curr_seed = rand.randint(1, 1 << 30)

            # need kwargs to have "seed" and "n_properties"
            stream = graph_object_stream(
                seed=curr_seed,
                generator_fn="random",
                property_distribution="uniform",
                distribution=None,
                verbose=0,
                n_properties=n_properties,
                name="erdos",
                min_nodes=min_nodes,
                max_nodes=max_nodes,
                m=m)

            for _ in range(n_graphs):
                self.graphs.append(next(stream))

    def _run_formulas(self):
        logger.debug("Evaluating graphs with formulas")
        for formula in self.formulas:
            result = self.run_formula(formula)

            if result is None:
                raise ValueError("One of the target formulas is not valid")

            self.applied.append(result)

    @overload
    def run_formula(self, formula: None) -> None: ...
    @overload
    def run_formula(self, formula: FOC) -> Tensor: ...

    def run_formula(self, formula: Optional[FOC]):
        if formula is None:
            return None

        results = []
        for graph in self.graphs:
            res = formula(graph)

            results.append(torch.from_numpy(res))

        return torch.cat(results, dim=0)


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
