import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class RelativeMultiheadAttention(nn.MultiheadAttention):
    """
    A multihead attention layer with relative positional encoding.
    """

    def __init__(self, *args, rel_pos_k: int = 128, **kwargs):
        """
        rel_pos_k: the window size for relative positional encoding.
        """

        super().__init__(*args, **kwargs)
        self.rel_pos_k = rel_pos_k
        # shape [n_heads, 2 * rel_pos_k - 1, head_dim]
        self.rel_pos_emb = nn.Parameter(
            torch.empty(self.num_heads, 2 * self.rel_pos_k - 1, self.head_dim)
        )
        torch.nn.init.xavier_uniform_(self.rel_pos_emb)

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
        attn_mask: Optional[Tensor] = None,
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
        attn_mask: If specified, a 2D or 3D mask preventing attention to certain positions. Must be of shape
            :math:`(L, S)` or :math:`(N\cdot\text{num\_heads}, L, S)`, where :math:`N` is the batch size,
            :math:`L` is the target sequence length, and :math:`S` is the source sequence length. A 2D mask will be
            broadcasted across the batch while a 3D mask allows for a different mask for each entry in the batch.
            Binary and float masks are supported. For a binary mask, a ``True`` value indicates that the
            corresponding position is not allowed to attend. For a float mask, the mask values will be added to
            the attention weight.
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

        R = self.rel_pos_emb[:, self.rel_pos_indices(query.size(-2), self.rel_pos_k), :]
        S_rel = torch.matmul(q.permute(1, 2, 0, 3), R.transpose(-1, -2)).permute(
            2, 0, 1, 3
        )

        # Compute attention scores (adding S_rel)
        scores = (torch.matmul(q, k.transpose(-2, -1)) + S_rel) / math.sqrt(
            self.head_dim
        )

        # Apply attention mask if provided
        if attn_mask is not None:
            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(0)
            scores = scores + attn_mask

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
