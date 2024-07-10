import math
import random

import torch
import torch.nn.functional as F
from torch import Tensor, nn


# from https://pytorch.org/tutorials/beginner/transformer_tutorial.html
class AbsolutePositionalEncoding(nn.Module):

    def __init__(
        self,
        d_model: int,
        max_len: int,
        dropout: float,
        concat: bool = False,
        max_shift: int = 0,
    ):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.max_shift = max_shift
        self.concat = concat

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
        """
        shift = (
            random.randint(0, min(self.max_shift, self.pe.shape[1] - x.size(1)))
            if self.training
            else 0
        )
        pes = self.pe[:, shift : x.size(1) + shift]
        if self.concat:
            # replace half of x with pos embeddings, effectively halving the embedding dim
            half_emb_dim = x.size(-1) // 2
            x[:, :, half_emb_dim:] = pes[:, :, :half_emb_dim]
        else:
            x = x + pes
        return self.dropout(x)
