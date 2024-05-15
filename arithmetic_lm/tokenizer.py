import string
from abc import ABC, abstractmethod

from torch import Tensor


class Tokenizer(ABC):
    """Abstract tokenizer class"""

    @abstractmethod
    def encode(self, text: str) -> list[int]:
        raise NotImplementedError

    @abstractmethod
    def decode(self, tokens: int | list[int] | Tensor) -> str:
        raise NotImplementedError


class CharTokenizer(Tokenizer):
    """Character tokenizer, using printable chars by default"""

    CHAR_VOCAB = list(string.printable)
    PLAIN_ADD_VOCAB = list("0123456789+=\n")

    def __init__(self, vocab: list | str = CHAR_VOCAB):
        self.vocab = vocab if isinstance(vocab, list) else list(vocab)
        self.vocab_size = len(self.vocab)
        self.stoi = {char: i for i, char in enumerate(self.vocab)}
        self.itos = {i: char for i, char in enumerate(self.vocab)}
        self.pad_token_id = 99

    def encode(self, text: str) -> list[int]:
        return [self.stoi[char] for char in text]

    def decode(self, tokens: int | list[int] | Tensor) -> str:
        if isinstance(tokens, int):
            tokens = [tokens]
        if isinstance(tokens, Tensor):
            tokens = tokens.tolist()
        return "".join([self.itos[token] for token in tokens])


TOKENIZERS = {
    "CharTokenizer": CharTokenizer,
}
