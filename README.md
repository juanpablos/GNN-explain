# GNN logic formula extractor

## Scripts

### Graphs

* `generate_train_graphs`: pregenerate graphs for training and testing GNNs. Both graphs datasets are build with different edge densities and sizes. Both datasets consider all posible 4-tuples of numbers from 0 to 100 in increments of 5, representing the probability of each color (4 colors) of being randomly selected when selecting the value of each node.
* `generate_formula_hashes`: pregenerate multiple formulas and combinations for easy access later. Also calculate a unique hash for each formula so it can be used in a more compact format.
* `prelabel_training_graphs`: using the pregenerated graphs for training and testing, and a subset of selected formulas, run the formulas over the graphs and label them. This process can take time, so doing it before training time speeds things up.
* `graph_label_distributions`: calculates positive rates for each formula labels. Allows to get an idea how many positive information the GNN is going to have at training time.

### GNNs

GNNs are trained over the random graphs generated with the graph scripts. We call a graph labeled using a formula a formula-labeled graph, meaning that each node label for that graph was generated based on rules using a logical formula.

* `generate_formulas`: trains `N` GNNs over a set of formulas using the pregenerated formula-labeled graphs and the pregenerated labels (if available). It will use several split strategies to automatically balance the training graphs depending on their labels to have a reasonable amount of possitive samples. The same pregenerated test graph dataset is used for all GNNs. We do early stopping once the GNN achieves 100% micro and macro accuracy on the test set. When a GNN achieves 100% accuracy on a formula-labeled graph, we say that the GNN have learned the formula.
* `_clean_networks`: given the generated training logs on `generate_formulas`, filter out from the `N` GNNs per formula the ones that did not achieve perfect accuracy on the test set. With this distinction the dataset of all trained GNNs is called `raw` and the cleaned subset is called `cleaned`.
* `find_duplicated_cleaned`: becuase logs can be splited, `_clean_networks` may generate duplicated cleaned files, this script lists those duplicated files.
* `check_incorrect_gnns`: does a double check for all selected formulas to see if they have 100% accuracy on a set of graphs.

### Metamodel

At this point we assume the GNNs have "learned" the formula it was trained on, as it was trained on formula-labeled graphs. We call these GNNs formula-GNNs and assume the weights of the GNN represents the formula it was trained on.

The Metamodel trains over formula-GNNs. Takes trained GNNs as input and tries to automatically extract information based on the input weights.
All training protocols implement 5-fold crossvalidation. Folds split by formula, so all GNNs of that formula are moved to the train or test set.

#### Flatten weights

All these metamodels just flatten all GNN weights and use the flattened vector as input.

* `main_2`: trains an encoder used for classification. Supports binary, multiclass and multilabel classification. This allows to check what is identifiable using the encoder.
* `main_3`: trains an encoder and LSTM-based text decoder to extract a tokenized formula from the GNN. We expect to extract the same formula the GNN was trained on. This script also implements an inference function.

### Metamodel Evaluation

* `main_3_evaluate`: performs evaluation over the metamodel. Takes the test split and runs the metamodel on each of the GNNs, extracting a formula. Then run those formulas over a formula-labeled graph dataset (the same used when testing the formula-GNNs). Using there graphs perform a semantic evaluation of the extracted formula and see how much it matches with the expected extracted formulas. It also implements an heuristic approach as a baseline instead of using the metamodel.

#### Synthetic data analysis

* `multilabel_analysis`: performs analysis for multilabel classification for the metamodel. The resulting file indicates: wrongly labeled labels, and when wrongly labeled which other label was selected, and that label was also wrongly selected. Basically the idea is to get a correlation that could indicate which other wrong label was selected when the expected label was not selected.
* `_group_text_evaluation`: takes the evaluation files generated by `main_3_evaluate` and compacts and summarizes them.
* `_text_evaluation_tree_edit_distance`: takes the evaluation files generated by `main_3_evaluate` and calculates the tree edit distance of the extracted formula and the expected formula.

#### Cora data analysis

* `build_cora_datasets`
* `_single_graph_test_runs`
* `_multi_graph_test_runs`
* `_explore_formula_match_cora`
* `evaluate_random_classifier`
* `evaluate_cora_formulas`

### Deprecated

* `main_1`: same as `generate_formulas`, for single formulas.
* `main_1.2`: same as `generate_formulas`, for bulk formulas.
* `main_4`: metamodel where encodes takes a graph as input instead of a flattened vector. The graph corresponds to the computational graph of the GNN, taking neurons as nodes and weights as edges.
* `main_5`: metamodel where encoder takes as input a list of flattened vectors instead of a single flattened vector. There list of flattened vectors corresponds to a flattened vector per GNN layer.

## Data

* `Cora`
* `cora_data`
* `experiment_sampler`
* `full_gnn`
* `gnns`
* `gnns_v2`
* `gnns_v3`
* `gnns_v4`
* `graphs`
