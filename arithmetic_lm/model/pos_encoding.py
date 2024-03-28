import math
from typing import Optional

import torch
from torch import Tensor
from torch import functional as F
from torch import nn
from torch.nn.modules.activation import (
    _arg_requires_grad,
    _check_arg_device,
    _is_make_fx_tracing,
)


# from https://pytorch.org/tutorials/beginner/transformer_tutorial.html
class AbsolutePositionalEncoding(nn.Module):

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
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
        """
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


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


class RelativePositionalEncoding(nn.Module):
    pass


# Since relative positional encoding implementation changes the multi-head attention layer,
# we need to child class the original MHA and override the forward method to include it.
class RelativeMultiheadAttention(nn.MultiheadAttention):
    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        **kwargs,
    ) -> tuple[Tensor, Optional[Tensor]]:
        """
        query: Query embeddings of shape :math:`(L, E_q)` for unbatched input, :math:`(L, N, E_q)` when ``batch_first=False``
            or :math:`(N, L, E_q)` when ``batch_first=True``, where :math:`L` is the target sequence length,
            :math:`N` is the batch size, and :math:`E_q` is the query embedding dimension ``embed_dim``.
            Queries are compared against key-value pairs to produce the output.
            See "Attention Is All You Need" for more details.
        key: Key embeddings of shape :math:`(S, E_k)` for unbatched input, :math:`(S, N, E_k)` when ``batch_first=False``
            or :math:`(N, S, E_k)` when ``batch_first=True``, where :math:`S` is the source sequence length,
            :math:`N` is the batch size, and :math:`E_k` is the key embedding dimension ``kdim``.
            See "Attention Is All You Need" for more details.
        value: Value embeddings of shape :math:`(S, E_v)` for unbatched input, :math:`(S, N, E_v)` when
            ``batch_first=False`` or :math:`(N, S, E_v)` when ``batch_first=True``, where :math:`S` is the source
            sequence length, :math:`N` is the batch size, and :math:`E_v` is the value embedding dimension ``vdim``.
            See "Attention Is All You Need" for more details.
        """
        # add relative positions to keys and values
        # TODO

        # call the parent class forward method
        return super().forward(query, key, value, **kwargs)
