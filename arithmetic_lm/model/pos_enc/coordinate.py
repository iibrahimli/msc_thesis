import math

import torch
from torch import Tensor, nn


class CoordinateEncoding(nn.Module):
    """Like position encoding, but with added timestep information."""

    def __init__(self, d_model: int, max_len: int, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("div_term", div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor, timestep: int) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
        """
        x = x + self.pe[:, : x.size(1)]
        # add timestep to the positional encoding
        x[:, :, 0::2] += torch.sin(timestep * self.div_term)
        x[:, :, 1::2] += torch.cos(timestep * self.div_term)
        return self.dropout(x)
