import random

import torch
from torch import Tensor, nn


class LearnedPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int, dropout: float, max_shift: int = 0):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.max_shift = max_shift
        self.max_len = max_len
        self.pe = nn.Embedding(max_len, d_model)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
        """
        shift = (
            random.randint(0, min(self.max_shift, self.max_len - x.size(1)))
            if self.training
            else 0
        )
        x = x + self.pe.weight[shift : x.size(1) + shift]
        return self.dropout(x)
