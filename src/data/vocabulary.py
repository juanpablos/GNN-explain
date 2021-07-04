from typing import Dict


class Vocabulary:
    def __init__(self):
        self.id2token: Dict[int, str] = {0: "<pad>", 1: "<sos>", 2: "<eos>"}
        self.token2id: Dict[str, int] = {v: k for k, v in self.id2token.items()}

    def __len__(self):
        return len(self.id2token)

    def add_or_get(self, token: str) -> int:
        if token not in self.token2id:
            _id = len(self.token2id)
            self.token2id[token] = _id
            self.id2token[_id] = token

        return self.token2id[token]

    def add(self, *tokens: str) -> None:
        for token in tokens:
            self.add_or_get(token)

    def get_id(self, token: str) -> int:
        return self.token2id[token]

    def get_token(self, token_id: int) -> str:
        return self.id2token[token_id]

    def load_vocab(self, vocab: Dict[str, int]):
        self.token2id = vocab.copy()
        self.id2token = {v: k for k, v in vocab.items()}

    @property
    def pad_token_id(self):
        return self.token2id["<pad>"]

    @property
    def start_token_id(self):
        return self.token2id["<sos>"]

    @property
    def end_token_id(self):
        return self.token2id["<eos>"]
