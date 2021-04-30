import csv
import logging
import os
import random
from typing import List

import numpy as np
import torch
import torch.nn.functional
from torch_geometric.data import DataLoader
from torch_scatter import scatter_mean
from torch_scatter.scatter import scatter_add

from src.data.formula_index import FormulaMapping
from src.data.gnn.utils import prepare_files
from src.generate_graphs import graph_data_stream
from src.graphs.foc import *
from src.models.ac_gnn import ACGNN
from src.run_logic import seed_everything

logger = logging.getLogger(__name__)
# graph_logger = logging.getLogger("graph_metrics")


def get_dataset(formula, run_uniform, rand_gen):
    data_config = {
        "formula": FOC(formula),
        "generator_fn": "random",
        "min_nodes": 20,  # 20, 80
        "max_nodes": 120,  # 80, 120
        # "seed": 0,
        "n_properties": 4,
        "verbose": 0,
        # --- generator config
        "name": "erdos",
        # ! because the generated graph is undirected,
        # ! the number of average neighbors will be double `m`
        "m": 4,
    }
    dataset = []

    logger.info("Generating graph dataset")

    if run_uniform:
        logger.info("Generating Uniform dataset")
        graph_generator = graph_data_stream(
            **data_config,
            distribution=None,
            property_distribution="uniform",
            number_of_graphs=1000,
            seed=rand_gen.randint(1, 1 << 30),
        )
        dataset.extend(graph_generator)

    else:
        logger.info("Generating unbalanced dataset")
        color_tuples = []
        for a in range(0, 101, 5):
            for b in range(0, 101, 5):
                for c in range(0, 101, 5):
                    for d in range(0, 101, 5):
                        if a + b + c + d != 100:
                            continue
                        color_tuples.append(
                            (a / 100.0, b / 100.0, c / 100.0, d / 100.0)
                        )

        for color_tuple in color_tuples:
            graph_generator = graph_data_stream(
                **data_config,
                distribution=color_tuple,
                property_distribution="manual",
                number_of_graphs=1,
                seed=rand_gen.randint(1, 1 << 30),
            )
            dataset.extend(graph_generator)

    return dataset


def evaluate_gnn(model, dataset, is_uniform: bool, formula_hash: str, gnn_id: int):
    model = model.to("cuda:0")
    model = model.eval()

    logger.debug("Evaluating GNN")

    loader = DataLoader(dataset, batch_size=2048)

    graph_acc = []
    bad_nodes_graph_list = []
    node_count_list = []

    n_nodes_badly_classified = 0
    n_graphs_with_nodes_badly_classified = 0

    avg_graph_missed_nodes = 0

    for data in loader:
        data = data.to("cuda:0")
        with torch.no_grad():
            output = model(x=data.x, edge_index=data.edge_index, batch=data.batch)

        output = torch.sigmoid(output)
        _, predicted_labels = output.max(dim=1)

        node_matches = torch.eq(predicted_labels, data.y).float().to("cuda:0")

        acc = scatter_mean(node_matches, data.batch)
        bad_nodes_graph = scatter_add((node_matches == 0).float(), data.batch)
        _, node_count_per_graph = torch.unique_consecutive(
            data.batch, return_counts=True
        )

        graph_acc.extend(acc.tolist())
        bad_nodes_graph_list.extend(bad_nodes_graph.tolist())
        node_count_list.extend(node_count_per_graph.tolist())

        n_nodes_badly_classified += (node_matches == 0).sum().item()
        n_graphs_with_nodes_badly_classified += (bad_nodes_graph > 0).sum().item()
        avg_graph_missed_nodes += bad_nodes_graph.sum().item()

    # for i, (g_acc, bad_nodes_g, node_count_g) in enumerate(
    #     zip(graph_acc, bad_nodes_graph_list, node_count_list)
    # ):
    #     graph_logger.debug(
    #         f"{formula_hash}:::{is_uniform}:::{gnn_id}:::{i},{g_acc},{bad_nodes_g},{node_count_g}"
    #     )

    if avg_graph_missed_nodes > 0:
        assert n_graphs_with_nodes_badly_classified > 0

    try:
        avg_graph_missed_nodes = (
            avg_graph_missed_nodes / n_graphs_with_nodes_badly_classified
        )
    except ZeroDivisionError:
        avg_graph_missed_nodes = 0
    return (
        np.min(graph_acc),
        np.mean(graph_acc),
        n_nodes_badly_classified,
        n_graphs_with_nodes_badly_classified,
        avg_graph_missed_nodes,
    )


def run_checks(gnns, model_configs, dataset, is_uniform: bool, formula_hash: str):
    _gnn_min = []
    _gnn_mean = []
    _gnn_node_misses = []
    _gnn_any_graph_misses = []
    _gnn_avg_misses_nodes_in_graph = []
    _gnn_not_perfect = 0

    msg = []
    gnn_model = ACGNN(**model_configs)

    logger.info("Running checks for formula")

    total_gnns = len(gnns)
    for i, gnn in enumerate(gnns):
        logger.debug(f"{i+1}/{total_gnns}")
        if i % 100 == 0:
            logger.info(f"{i+1}/{total_gnns}")
        gnn_model.load_state_dict(gnn)
        _min, _mean, _node_miss, _graph_miss, _avg_missed_nodes_in_graph = evaluate_gnn(
            model=gnn_model,
            dataset=dataset,
            is_uniform=is_uniform,
            formula_hash=formula_hash,
            gnn_id=i,
        )

        _gnn_min.append(_min)
        _gnn_mean.append(_mean)
        _gnn_node_misses.append(_node_miss)
        _gnn_any_graph_misses.append(_graph_miss)
        _gnn_avg_misses_nodes_in_graph.append(_avg_missed_nodes_in_graph)
        _gnn_not_perfect += _graph_miss > 0

        if _graph_miss > 0:
            assert _node_miss > 0
        if _node_miss > 0:
            assert _graph_miss > 0

    logger.info("Finished running gnn checks")

    msg.append(
        f"Min/Max/Avg min "
        f"{np.min(_gnn_min)} "
        f"{np.max(_gnn_min)} "
        f"{np.mean(_gnn_min)}"
    )
    msg.append(
        f"Min/Max/Avg mean "
        f"{np.min(_gnn_mean)} "
        f"{np.max(_gnn_mean)} "
        f"{np.mean(_gnn_mean)}"
    )

    msg.append(
        "Min/Max/Avg node misses "
        f"{np.min(_gnn_node_misses)} "
        f"{np.max(_gnn_node_misses)} "
        f"{np.mean(_gnn_node_misses)}"
    )
    msg.append(
        "Min/Max/Avg graph misses "
        f"{np.min(_gnn_any_graph_misses)} "
        f"{np.max(_gnn_any_graph_misses)} "
        f"{np.mean(_gnn_any_graph_misses)}"
    )
    msg.append(
        "Min/Max/Avg node misses in graph "
        f"{np.min(_gnn_avg_misses_nodes_in_graph)} "
        f"{np.max(_gnn_avg_misses_nodes_in_graph)} "
        f"{np.mean(_gnn_avg_misses_nodes_in_graph)}"
    )
    msg.append(f"Num with min <98 {(np.array(_gnn_min) < 0.98).sum()}")
    msg.append(f"Num with min <99 {(np.array(_gnn_min) < 0.99).sum()}")
    msg.append(f"Num with min <100 {(np.array(_gnn_min) < 1).sum()}")
    msg.append(f"GNNs not perfect (in some degree) {_gnn_not_perfect}")

    return (
        msg,
        total_gnns,
        _gnn_not_perfect,
        _gnn_node_misses,
        _gnn_any_graph_misses,
        _gnn_avg_misses_nodes_in_graph,
        _gnn_min,
        _gnn_mean,
        len(dataset),
    )


def run(
    results_path: str,
    tagging_folder: str,
    summary_file: str,
    hashes_to_use: List[str] = None,
    _debug_limit_dataset: int = None,
):
    seed = 42

    tagging_path = os.path.join(results_path, tagging_folder)
    os.makedirs(tagging_path, exist_ok=True)

    model_configs = {
        "input_dim": 4,
        "hidden_dim": 8,
        "hidden_layers": None,
        "output_dim": 2,
        "aggregate_type": "add",
        "combine_type": "identity",
        "num_layers": 2,
        "mlp_layers": 1,  # the number of layers in A and V
        "combine_layers": 2,  # layers in the combine MLP if combine_type=mlp
        "task": "node",
        "use_batch_norm": False,
    }

    gnn_path = os.path.join("data", "full_gnn", "40e65407aa")
    formula_files = prepare_files(path=gnn_path, model_hash="40e65407aa")

    if hashes_to_use is not None:
        formula_files = {
            _hash: file
            for _hash, file in formula_files.items()
            if _hash in hashes_to_use
        }

    logger.info(f"Running {len(formula_files)} formulas")

    formula_mapping = FormulaMapping(os.path.join("data", "formulas.json"))

    n_formulas = len(formula_files)

    with open(
        os.path.join(results_path, summary_file), "a", encoding="utf-8"
    ) as results_output:
        for i, (formula_hash, formula_file) in enumerate(formula_files.items()):
            formula = formula_mapping[formula_hash]

            logger.debug(f"Loading GNNs for formula {repr(formula)}")
            gnns = torch.load(os.path.join(gnn_path, formula_file))

            formula_msg_list = []

            for uniform in [True, False]:
                logger.info(
                    f"{i+1}/{n_formulas} "
                    f"Running formula {repr(formula)} for uniform={uniform}"
                )
                seed_everything(seed)
                rand = random.Random(seed)

                dataset = get_dataset(
                    formula=formula, run_uniform=uniform, rand_gen=rand
                )
                if _debug_limit_dataset is not None:
                    dataset = dataset[:_debug_limit_dataset]

                (
                    msg_list,
                    total,
                    failed,
                    failed_nodes,
                    failed_graphs,
                    avg_failed_nodes_per_graph,
                    gnn_min,
                    gnn_mean,
                    n_graphs,
                ) = run_checks(
                    gnns=gnns,
                    model_configs=model_configs,
                    dataset=dataset,
                    is_uniform=uniform,
                    formula_hash=formula_hash,
                )

                msg = "\n\t\t".join(msg_list)
                msg = (
                    f"\n\tuniform={uniform} "
                    f"{failed}/{total} ({float(failed)/total:.1%}) "
                    f"did not learn the formula\n\t\t{msg}"
                )
                formula_msg_list.append(msg)

                if not uniform:
                    with open(
                        os.path.join(tagging_path, f"{formula_file}.csv"),
                        "w",
                        encoding="utf-8",
                        newline="",
                    ) as formula_tags_output:
                        writer = csv.writer(
                            formula_tags_output,
                            delimiter=",",
                        )
                        for i, tags in enumerate(
                            zip(
                                failed_nodes,
                                failed_graphs,
                                avg_failed_nodes_per_graph,
                                gnn_min,
                                gnn_mean,
                            )
                        ):
                            writer.writerow((i,) + tags + (n_graphs,))

            formula_msg = "\n".join(formula_msg_list)
            formula_msg = f"{formula}\n{repr(formula)}\n{formula_msg}\n\n"

            results_output.write(formula_msg)
            results_output.flush()


if __name__ == "__main__":
    __results_path = os.path.join("results", "validity_checks", "full_gnn")
    os.makedirs(__results_path, exist_ok=True)

    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    # graph_logger.setLevel(logging.DEBUG)
    # graph_logger.propagate = False

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    _console_f = logging.Formatter("%(message)s")
    ch.setFormatter(_console_f)

    # fh = logging.FileHandler(os.path.join(__results_path, "checks.log"))
    # fh.setLevel(logging.DEBUG)
    # _file_f = logging.Formatter("%(asctime)s:::%(levelname)s:::%(message)s")
    # fh.setFormatter(_file_f)

    logger.addHandler(ch)
    # logger.addHandler(fh)

    # graph_fh = logging.FileHandler(
    #     os.path.join(__results_path, "per_graph_gnn_results.log")
    # )
    # graph_fh.setLevel(logging.DEBUG)
    # _file_f = logging.Formatter("%(asctime)s:::%(message)s")
    # graph_fh.setFormatter(_file_f)

    # graph_logger.addHandler(graph_fh)

    run(
        hashes_to_use=["dc670b1bec", "c439b78825", "d8a8c1299e", "8a4d116623"],
        results_path=__results_path,
        tagging_folder="tagging",
        summary_file="check_results.txt",
        _debug_limit_dataset=None,
    )
