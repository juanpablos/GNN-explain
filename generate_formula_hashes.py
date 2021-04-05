import hashlib
import json
import itertools
from src.graphs.foc import *
import random

color_names = ["RED", "BLUE", "GREEN", "BLACK"]
formulas = []


def get_hash(_formula):
    return hashlib.md5(repr(_formula).encode()).hexdigest()[:10]


def generate_colors():
    return [Property(c) for c in color_names]


def generate_pair_formulas(pairs, operator):
    combinations = []
    for pair in pairs:
        first, second = pair
        combinations.append(operator(first, second))
    return combinations


def generate_gt_exist_random_base(bases, nested, lower, upper):
    exists = []
    for inner in nested:
        for i in range(lower, upper + 1):
            base = random.choice(bases)

            f = AND(base, Exist(AND(Role("EDGE"), inner), lower=i))

            exists.append(f)

    return exists


def generate_lt_exist_random_base(bases, nested, lower, upper):
    exists = []
    for inner in nested:
        for i in range(lower, upper + 1):
            base = random.choice(bases)

            f = AND(base, Exist(AND(Role("EDGE"), inner), upper=i))

            exists.append(f)

    return exists


def generate_gtlt_exist_random_base(bases, nested, restriction_pairs):
    exists = []
    for lower, upper in restriction_pairs:
        base = random.choice(bases)
        inner = random.choice(nested)

        f = AND(base, Exist(AND(Role("EDGE"), inner), lower=lower, upper=upper))

        exists.append(f)

    return exists


def write(formula_file, n_splits):
    mapping = {get_hash(_formula): repr(_formula) for _formula in formulas}

    with open(formula_file, "w") as f:
        json.dump(mapping, f, indent=2)
    if n_splits > 1:
        tuples = list(mapping.items())
        splits = [tuples[i::n_splits] for i in range(n_splits)]

        for i, split in enumerate(splits, start=1):
            mapping = dict(split)
            with open(f"{formula_file}.{i}", "w") as f:
                json.dump(mapping, f, indent=2)


color_bases = generate_colors()
formulas.extend(color_bases)


pair_combinations = list(itertools.combinations(color_bases, 2))
or_pairs = generate_pair_formulas(pair_combinations, OR)

# this doesnt make any sense
# and_pairs = generate_pair_formulas(pair_combinations, AND)
formulas.extend(or_pairs)


# at lest N
color_gt_exist = generate_gt_exist_random_base(
    bases=color_bases, nested=color_bases, lower=1, upper=5
)
# at most N
color_lt_exist = generate_lt_exist_random_base(
    bases=color_bases, nested=color_bases, lower=1, upper=5
)
formulas.extend(color_gt_exist + color_lt_exist)


and_or_pairs = or_pairs
# at lest N, + AND/OR
operation_gt_exist = generate_gt_exist_random_base(
    bases=and_or_pairs, nested=color_bases, lower=1, upper=5
)
# at most N, + AND/OR
operation_lt_exist = generate_lt_exist_random_base(
    bases=and_or_pairs, nested=color_bases, lower=1, upper=5
)
formulas.extend(operation_gt_exist + operation_lt_exist)

# between N and M, + AND/OR/*
restrictions = list(itertools.combinations_with_replacement([1, 2, 3, 4, 5], r=2))
operation_gtlt_exist_1 = generate_gtlt_exist_random_base(
    bases=and_or_pairs + color_bases, nested=color_bases, restriction_pairs=restrictions
)
operation_gtlt_exist_2 = generate_gtlt_exist_random_base(
    bases=and_or_pairs + color_bases, nested=color_bases, restriction_pairs=restrictions
)
formulas.extend(operation_gtlt_exist_1 + operation_gtlt_exist_2)

write(formula_file="data/formulas_v3.json", n_splits=6)
