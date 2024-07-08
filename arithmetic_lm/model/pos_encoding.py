import math
import random
from typing import Optional

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
        # shape [n_heads, 2 * rel_pos_k + 1, head_dim]
        self.rel_pos_embedding = nn.Parameter(
            torch.randn(self.num_heads, 2 * self.rel_pos_k + 1, self.head_dim)
        )

    def rel_pos_indices(self, n: int, k: int) -> Tensor:
        """
        Returns a tensor of shape [n, n] where the value at position [i, j] is the
        index of embedding for relative position of j to i (clipped to window size k).
        """
        return torch.clamp(torch.arange(n).unsqueeze(1) - torch.arange(n), -k, k) + k

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        need_weights: bool = True,
        average_attn_weights: bool = True,
        **kwargs,
    ) -> tuple[Tensor, Tensor | None]:
        """
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

        # NOTE: only batch first supported

        # Get batch size and sequence length
        batch_size, seq_len, _ = query.size()

        # Perform input projection
        if self._qkv_same_embed_dim:
            q, k, v = F.linear(query, self.in_proj_weight, self.in_proj_bias).chunk(
                3, dim=-1
            )
        else:
            w_q, w_k, w_v = self.in_proj_weight.split(self.embed_dim)
            b_q, b_k, b_v = (
                self.in_proj_bias.split(self.embed_dim)
                if self.in_proj_bias is not None
                else (None, None, None)
            )
            q = F.linear(query, w_q, b_q)
            k = F.linear(key, w_k, b_k)
            v = F.linear(value, w_v, b_v)

        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # at this point, q and k have shape [B, n_heads, L, head_dim]

        # compute S_rel
        # For each head, Shaw et al. (2018) instantiate an intermediate tensor R of shape (L, L, Dh ),
        # containing the embeddings that correspond to the relative distances between all keys and queries.
        # Q is then reshaped to an (L, 1, Dh) tensor, and Srel = QR^T

        # R shape [n_heads, L, L, head_dim]
        R = self.rel_pos_embedding[
            :, self.rel_pos_indices(query.size(-2), self.rel_pos_k), :
        ]

        # S_rel = QR^T
        S_rel = torch.einsum("bhld,hsld->bhsl", q, R)

        # Compute attention scores (adding S_rel)
        scores = (torch.matmul(q, k.transpose(-2, -1)) + S_rel) / math.sqrt(
            self.head_dim
        )

        # Apply softmax to get attention weights
        attn_weights = F.softmax(scores, dim=-1)

        # Apply attention weights to values
        attn_output = torch.matmul(attn_weights, v)

        # Reshape and project the output
        attn_output = (
            attn_output.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, self.embed_dim)
        )
        output = self.out_proj(attn_output)

        if need_weights:
            if average_attn_weights:
                # Average attention weights over heads
                attn_weights = attn_weights.mean(dim=1)
            else:
                # Reshape attention weights to include head dimension
                attn_weights = attn_weights.view(
                    batch_size, self.num_heads, seq_len, seq_len
                )
            return output, attn_weights
        else:
            return output, None


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
