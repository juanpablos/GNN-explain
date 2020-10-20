import re
import warnings
from typing import Dict, List, Optional, Set, Tuple

from src.data.vocabulary import Vocabulary
from src.graphs.foc import *


class FormulaReconstruction:
    def __init__(self, vocabulary: Vocabulary):
        self.vocabulary = vocabulary

        self.properties = self.get_ids(["RED", "BLUE", "GREEN", "BLACK"])
        self.relations = self.get_ids(["EDGE"])
        self.operators = self.get_ids(["AND", "OR"])

        self.exist_indices = set()
        for token, token_id in self.vocabulary.token2id.items():
            if "Exist" in token:
                self.exist_indices.add(token_id)

        self.exist_values: Dict[int, Tuple[str, str]] = {}
        for exist_ind in self.exist_indices:
            token = self.vocabulary.get_token(exist_ind)
            _, lower, upper, *_ = re.split(r",|\(|\)", token)
            self.exist_values[exist_ind] = (lower, upper)

    def get_ids(self, tokens):
        token_ids: Set[int] = set()
        for token in tokens:
            try:
                token_id = self.vocabulary.get_id(token)
                token_ids.add(token_id)
            except KeyError as e:
                warnings.warn(f"{e.args[0]} is not in the vocabulary")

        return token_ids

    def id2expression(self, token_id: int):
        # token_id is an index, we have to remap it to the token

        token = self.vocabulary.get_token(token_id)

        if token_id in self.properties:
            expression = f"Property('{token}')"
            operands = 0
        elif token_id in self.relations:
            expression = f"Role('{token}')"
            operands = 0
        elif token_id in self.operators:
            expression = f"{token}({{0}},{{1}})"
            operands = 2
        elif token_id in self.exist_indices:
            lower, upper = self.exist_values[token_id]
            expression = f"Exist({{0}},{lower},{upper})"
            operands = 1
        else:
            raise ValueError(f"{token} is not a valid token")

        return expression, operands

    def tokens2expression(self, token_ids: List[int]):
        def rec(index):
            if index >= len(token_ids):
                raise ValueError("Invalid input: not enough tokens to parse")

            token_id = token_ids[index]
            expr, n = self.id2expression(token_id)

            if n == 0:
                return index, expr
            else:
                inner = []
                for _ in range(n):
                    index, s = rec(index + 1)
                    inner.append(s)

                expr = expr.format(*inner)

                return index, expr

        # * the token_ids should start with a valid token and end with a <eos>
        # * followed by padding. So the last index should be <eos>. If not,
        # * then the formula is not correct

        last, string = rec(0)

        # the next index should be accesable and should be <eos>
        if last + 1 > len(token_ids) - 1:
            raise ValueError(
                "Invalid input: the formula does not have end token")

        if token_ids[last + 1] != self.vocabulary.end_token_id:
            raise ValueError(
                "Invalid input: there are tokens left in the input "
                "that are not padding")

        return string

    def batch2expression(self, batch_data: List[List[int]]):
        expressions: List[Optional[FOC]] = []
        n_compiled = 0
        for sample in batch_data:
            try:
                formula = self.tokens2expression(sample)
                expressions.append(FOC(eval(formula)))
                n_compiled += 1
            except ValueError:
                expressions.append(None)

        return expressions, n_compiled
