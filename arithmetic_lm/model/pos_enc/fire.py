import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class FIRE(nn.Module):
    def __init__(
        self,
        num_heads: int,
        mlp_width: int = 32,
        init_c: float = 0.1,
        init_L: int = 512.0,
        eps: float = 1e-6,
    ):
        """
        FIRE attention bias module.
        Args:
          num_heads: number of attention heads.
          mlp_width: Width of MLP.
          init_c: initial value of log transformation parameter
          init_L: initial value of thresholding parameter
          eps: small constant for numerical stability
        """
        super().__init__()
        # Define the MLP layers
        self.mlp = nn.Sequential(
            nn.Linear(1, mlp_width), nn.ReLU(), nn.Linear(mlp_width, num_heads)
        )
        # Initialize c (log transformation parameter)
        self.c = nn.Parameter(torch.tensor(init_c))

        # Initialize L (threshold)
        self.init_L = nn.Parameter(torch.tensor(init_L), requires_grad=False)
        # Learn a multiplier to L
        self.L_multiplier = nn.Parameter(torch.tensor(1.0))
        self.eps = eps

    def forward(self, seq_len: int, device: torch.device) -> Tensor:
        """
        Compute FIRE attention bias.
        Args:
        Returns:
            attention bias,
            shape [1, num_heads, seq_len, seq_len]
        """
        positions = torch.arange(seq_len, dtype=torch.float, device=device)
        rel_distance = positions[:, None] - positions[None, :]
        # Thresholding the normalizer
        threshold = torch.abs(self.L_multiplier * self.init_L)
        pos_normalizer = torch.max(positions, threshold)
        pos_normalizer = pos_normalizer[:, None]
        # Amplifying differences among local positions with log transform
        rel_distance = torch.log(torch.abs(self.c * rel_distance) + 1)
        pos_normalizer = torch.log(torch.abs(self.c * pos_normalizer) + 1) + self.eps
        # Progressive interpolation
        normalized_distance = rel_distance / pos_normalizer
        fire_bias = self.mlp(normalized_distance.unsqueeze(-1))
        fire_bias = fire_bias.unsqueeze(0).permute(0, 3, 1, 2)
        return fire_bias


class FireMultiheadAttention(nn.MultiheadAttention):
    """
    A multihead attention layer with FIRE.
    Paper: https://arxiv.org/pdf/2310.04418
    """

    def __init__(self, *args, **kwargs):
        """
        npos_max: the window size for relative positional encoding.
        """

        super().__init__(*args, **kwargs)
        self.fire = FIRE(num_heads=self.num_heads)

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
            :math:`(L, S)` or :math:`(N . \text{num_heads}, L, S)`, where :math:`N` is the batch size,
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

        scores = torch.matmul(q, k.transpose(-2, -1))
        # scores shape: [B, n_heads, L, L]

        # === fire ===
        bias = self.fire(seq_len, device=query.device)
        # bias shape: [1, n_heads, L, L]

        # add bias to scores
        scores += bias

        # ============

        # normalize scores by sqrt(d_k)
        scores = scores / math.sqrt(self.head_dim)

        # Apply attention mask if provided
        if attn_mask is not None:
            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(0)
            scores = scores + attn_mask

        # Apply softmax to get attention weights
        attn_weights = F.softmax(scores, dim=-1)

        # Apply dropout
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)

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
