import math
from typing import Optional

import torch
from torch import Tensor, nn


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


class RandomPositionalEncoding(AbsolutePositionalEncoding):
    """
    Like absolute, but sample ordered random indices
    Paper: https://arxiv.org/abs/2305.16843
    """

    def __init__(
        self,
        d_model: int,
        max_len: int,
        dropout: float,
        max_rand_len: int = 32,
    ):
        super().__init__(d_model, max_len, dropout)
        self.max_rand_len = max_rand_len

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
        """
        # uniformly sample indices
        random_idxs = torch.randperm(self.max_rand_len)[: x.shape[1]]
        # sort
        random_idxs, _ = random_idxs.sort()
        x = x + self.pe[:, random_idxs]
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


class RandomCoordinateEncoding(CoordinateEncoding):
    """Random positional encoding for UT"""

    def __init__(
        self,
        d_model: int,
        max_len: int,
        dropout: float,
        max_rand_len: int = 32,
    ):
        super().__init__(d_model, max_len, dropout)
        self.max_rand_len = max_rand_len

    def forward(self, x: Tensor, timestep: int) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
        """

        # uniformly sample indices
        random_idxs = torch.randperm(self.max_rand_len)[: x.shape[0]]
        # sort
        random_idxs, _ = random_idxs.sort()
        print("random_idxs:", random_idxs)
        print("self.pe[:, random_idxs].shape:", self.pe[:, random_idxs].shape)
        x = x + self.pe[:, random_idxs]

        # add timestep to the positional encoding
        x[:, :, 0::2] += torch.sin(timestep * self.div_term)
        x[:, :, 1::2] += torch.cos(timestep * self.div_term)

        return self.dropout(x)


def rel_pos_indices(n: int, k: int) -> Tensor:
    """
    Returns a tensor of shape [n, n] where the value at position [i, j] is the
    index of embedding for relative position of j to i (clipped to window size k).
    """
    return torch.clamp(torch.arange(n).unsqueeze(1) - torch.arange(n), -k, k) + k


# Since relative positional encoding implementation changes the multi-head attention layer,
# we need to child class the original MHA and override the forward method to include it.
class RelativeMultiheadAttention(nn.MultiheadAttention):
    """
    A multihead attention layer with relative positional encoding.
    """

    def __init__(self, *args, rel_pos_k: int = 16, **kwargs):
        """
        rel_pos_k: the window size for relative positional encoding.
        """

        super().__init__(*args, **kwargs)
        self.rel_pos_k = rel_pos_k
        self.rel_pos_embedding = nn.Embedding(2 * self.rel_pos_k + 1, self.embed_dim)

        # TODO: test init to 0 to see if loss NaNs or not
        nn.init.zeros_(self.rel_pos_embedding.weight)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attn_mask: Optional[Tensor] = None,
        **kwargs,
    ) -> tuple[Tensor, Optional[Tensor]]:
        """
        THIS IMPLEMENTATION SHARES REL EMBEDDING ACROSS HEADS
        For rel pos enc, see Shaw et al. 2018: https://arxiv.org/abs/1803.02155
        A modified version is implemented here, see Huang et al. 2018: https://arxiv.org/pdf/1809.04281.pdf
        (no relative position for the value, only for the key)
        This is an inefficient implementation, could be optimized for less memory usign skewing as shown in Huang et al. 2018.,
        see: https://jaketae.github.io/study/relative-positional-encoding/
        Another reference: https://ychai.uk/notes/2019/10/17/NN/Transformer-variants-a-peek/

        Args, from torch docs:
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

        # query: [B, L, D] (D is query embed_dim)
        # R: [L, L, D] (relative positions embeddings)
        # 1. reshape query to [L, B, D]
        # 2. S_rel = Q_reshaped * R^T (batch matrix-matrix product)
        # 3. add S_rel / sqrt(D) to attn_mask (since attn_mask is added to softmax input term QK^T)
        # 4. call super().forward with the modified attn_mask
        R = self.rel_pos_embedding(
            rel_pos_indices(query.size(1), self.rel_pos_k).to(query.device)
        )
        S_rel = torch.einsum("bld,lkd->blk", query, R)
        rel_pos_mask = S_rel / (self.embed_dim**0.5)

        # TODO: might be wrong, not sure if it's flattened batch first or heads first
        # rel_pos_mask has shape [B, L, L], we need [B*n_heads, L, L]
        rel_pos_mask = (
            rel_pos_mask.unsqueeze(1)
            .expand(-1, self.num_heads, -1, -1)
            .reshape(-1, query.size(1), query.size(1))
        )

        if attn_mask is None:
            attn_mask = rel_pos_mask
        else:
            if attn_mask.ndim == 2:
                attn_mask = attn_mask.unsqueeze(0)
            attn_mask = attn_mask + rel_pos_mask

        return super().forward(query, key, value, attn_mask=attn_mask, **kwargs)
