import math

import torch
from torch import Tensor, nn


class RandomizedPositionalEncoding(nn.Module):
    """
    Paper: https://arxiv.org/abs/2305.16843
    """

    def __init__(
        self,
        d_model: int,
        max_len: int,
        dropout: float,
    ):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

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
        if self.training:
            max_rand_idx = min(int(x.size(1) * 1.5), self.pe.size(1))
            # sample [x.size(0), x.size(1)] random indices from range(0, max_rand_idx)
            # without replacement
            rand_idx = torch.cat(
                [
                    torch.randperm(max_rand_idx)[None, : x.size(1)]
                    for _ in range(x.size(0))
                ],
                dim=0,
            )
            # sort along dim=1
            rand_idx, _ = rand_idx.sort(dim=1)
            rand_idx = rand_idx.to(x.device)
            # get positional encoding for random indices
            pes = self.pe[:, rand_idx, :]
        else:
            pes = self.pe[:, : x.size(1), :]
        return self.dropout(x + pes)
