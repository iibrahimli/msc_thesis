import math
import random
from typing import Optional

import torch
from torch import Tensor, nn


# from https://pytorch.org/tutorials/beginner/transformer_tutorial.html
class AbsolutePositionalEncoding(nn.Module):

    def __init__(
        self,
        d_model: int,
        max_len: int,
        dropout: float,
        concat: bool = True,  # TODO: set to False
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


class LearnedPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int, dropout: float, max_shift: int = 0):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.max_shift = max_shift
        self.pe = nn.Parameter(torch.randn(max_len, d_model) * 0.2)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
        """
        shift = (
            random.randint(0, min(self.max_shift, self.pe.shape[0] - x.size(1)))
            if self.training
            else 0
        )
        x = x + self.pe[shift : x.size(1) + shift]
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


class AbacusEncoding(nn.Module):
    """
    Abacus Embeddings, learned emebddings resued for each digit.
    Integers must be reversed for this to work correctly.
    Transformers Can Do Arithmetic with the Right Embeddings, McLeish et al. (2024)
    Taken from: https://github.com/mcleish7/arithmetic/blob/86022a57d38c0fde46444d62e8dcbebcc0af614c/abacus.py
    """

    def __init__(
        self,
        # digit_tokens: list[int],
        embedding_dim: int,
        max_seq_length: int = 256,
        max_k: int = 30,
    ):
        """
        digit_tokens (list): list of the tokens for each of the 10 digits, in pseudocode:
            `digit_tokens = tokenizer.convert_tokens_to_ids(['0','1','2','3','4','5','6','7','8','9'])`
        embedding_dim (int): dimension to embed into
        max_seq_length (int): maximum number of embeddings that can be trained
        max_k (int): maximum k value which we randomly shift by during training
        """
        super().__init__()
        self.embedding = torch.nn.Embedding(max_seq_length, embedding_dim)

        # TODO: hardcoded, CharTokenizer maps 0..9 -> 0..9
        digit_tokens = list(range(10))
        self.register_buffer("digits", torch.tensor(digit_tokens), persistent=False)

        self.max_k = max_k

    def helper(self, mask: Tensor, device: torch.device):
        """
        Converts a binary mask of digit locations into spans of consecutive digits
        """
        mask_shape = mask.shape

        # Create a shifted version of the mask to detect changes from 0 to 1
        shifted_mask = torch.cat(
            [
                torch.zeros((mask_shape[0], 1), device=device, dtype=mask.dtype),
                mask[:, :-1],
            ],
            dim=1,
        )
        starts = (shifted_mask != mask) & mask

        # Generate IDs for each segment of 1s, processing row-wise
        segment_ids = torch.cumsum(starts, dim=1)

        # Generate an index array row-wise
        index = torch.arange(mask.size(1)).repeat(mask.size(0), 1).to(device)

        # Reset index at the start of each segment
        reset_index = torch.zeros_like(mask).long()
        second_term = index * starts.long()
        reset_index = reset_index.scatter_add(1, segment_ids, second_term)

        # Calculate positions in segment
        positions = index - reset_index.gather(1, segment_ids) + 1

        # Ensure only values within 1-segments are non-zero
        result = positions * mask

        return result

    def forward(self, input_ids: Tensor):
        """
        input_ids (tensor): a batch of inputs, each row is a sample
        """
        mask = torch.isin(input_ids, self.digits)
        output = self.helper(mask, input_ids.device)

        k = 0
        if self.training:
            k = random.randint(0, self.max_k)
            # as we already have ones in the tensor, the tensor values will be k+1
            output[output > 0] += k

        return self.embedding(output)
