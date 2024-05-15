import torch
from torch import Tensor, nn

from .pos_encoding import (
    CoordinateEncoding,
    RandomCoordinateEncoding,
    RelativeMultiheadAttention,
)
from .utils import init_weights


class UniversalTransformerDecoder(nn.Module):
    """A decoder-only universal transformer model"""

    def __init__(
        self,
        context_len: int,
        n_embd: int,
        n_head: int,
        max_steps: int,
        vocab_size: int,
        ff_factor: int = 4,
        dropout: float = 0.1,
        pos_enc: str = "abs",
        emb_weight_sharing: bool = True,
    ):
        """
        Arguments:
            context_len: context length, i.e. the number of expected features in the input
            n_embd: dimensionality of model embeddings
            n_head: number of heads in the multi-head attention
            max_steps: number of maximum recurrent steps
            vocab_size: size of the vocabulary
            ff_factor: factor by which to scale the hidden layer dimensionality in the feedforward layer
            dropout: dropout probability
            pos_enc: type of positional encoding to use, either "abs" for absolute or "rel" for relative
        """

        super().__init__()

        self.context_len = context_len
        self.n_embd = n_embd
        self.n_head = n_head
        self.vocab_size = vocab_size
        self.ff_factor = ff_factor
        self.dropout = dropout
        self.max_steps = max_steps
        self.enc_dec = False
        self.pos_enc = pos_enc

        # embedding
        self.embedding = nn.Embedding(vocab_size, n_embd)
        if pos_enc == "abs":
            self.coord_encoder = CoordinateEncoding(
                n_embd, max_len=context_len, dropout=dropout
            )
        elif pos_enc == "random":
            self.coord_encoder = RandomCoordinateEncoding(
                n_embd, max_len=context_len, dropout=dropout
            )

        # same as decoder layer essentially, but without cross attention
        self.layer = nn.TransformerEncoderLayer(
            d_model=n_embd,
            nhead=n_head,
            dim_feedforward=n_embd * ff_factor,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.norm = nn.LayerNorm(n_embd)

        # change all self-attention layers to relative multi-head attention
        if self.pos_enc == "rel":
            self.layer.self_attn = RelativeMultiheadAttention(
                n_embd,
                n_head,
                dropout=dropout,
                bias=True,  # is true by default
                batch_first=True,
                rel_pos_k=16,
            )

        # output to vocab dim
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

        # weight tying
        if emb_weight_sharing:
            self.lm_head.weight = self.embedding.weight

        self.apply(init_weights)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len]``

        Returns:
            logits: Tensor, shape ``[batch_size, seq_len, vocab_size]``
        """
        x = self.embedding(x)

        for t in range(self.max_steps):
            # TODO: this is needed for timestep info, keep even with relative pos enc
            x = self.coord_encoder(x, timestep=t)
            x = self.layer(
                x,
                is_causal=True,
                src_mask=nn.Transformer.generate_square_subsequent_mask(x.size(1)).to(
                    x.device
                ),
            )

        x = self.norm(x)
        x = self.lm_head(x)
        return x

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
