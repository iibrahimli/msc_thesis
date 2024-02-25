import string
from abc import ABC, abstractmethod


class Tokenizer(ABC):
    """Abstract tokenizer class"""

    @abstractmethod
    def encode(self, text: str) -> list[int]:
        raise NotImplementedError

    @abstractmethod
    def decode(self, tokens: list[int]) -> str:
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

    def encode(self, text: str) -> list[int]:
        return [self.stoi[char] for char in text]

    def decode(self, tokens: list[int]) -> str:
        if isinstance(tokens, int):
            tokens = [tokens]
        return "".join([self.itos[token] for token in tokens])
