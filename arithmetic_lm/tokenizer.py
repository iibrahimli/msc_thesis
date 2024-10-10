import string
from abc import ABC, abstractmethod

import numpy as np
import torch
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
    """Character tokenizer optimized with NumPy arrays"""

    CHAR_VOCAB = list(string.printable)
    PLAIN_ADD_VOCAB = list("0123456789+=\n")

    def __init__(self, vocab: list | str = CHAR_VOCAB):
        self.vocab = vocab if isinstance(vocab, list) else list(vocab)
        self.vocab_size = len(self.vocab)
        self.stoi_array = np.full(
            128, -1, dtype=np.int32
        )  # Initialize with -1 (invalid)
        for i, char in enumerate(self.vocab):
            self.stoi_array[ord(char)] = i  # Fill only valid character positions
        self.itos = np.array(self.vocab)  # NumPy array for faster index lookups
        self.pad_token_id = 99

    def encode(
        self, text: str, return_tensors: bool = False
    ) -> list[int] | torch.Tensor:
        tokens = [self.stoi_array[ord(char)] for char in text]
        if return_tensors:
            return torch.tensor(tokens, dtype=torch.int32)
        return tokens

    def decode(self, tokens: int | list[int] | torch.Tensor) -> str:
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.tolist()
        elif isinstance(tokens, int):
            tokens = [tokens]
        if (
            isinstance(tokens, list)
            and isinstance(tokens[0], list)
            and len(tokens) == 1
        ):
            tokens = tokens[0]
        return "".join(self.itos[token] for token in tokens)
