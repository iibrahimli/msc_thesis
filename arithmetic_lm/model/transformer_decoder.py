from typing import Literal

from torch import Tensor, nn

from arithmetic_lm.model.pos_enc import *
from arithmetic_lm.model.utils import SinusoidalEmbedding, init_weights


class TransformerDecoder(nn.Module):
    """Simple small decoder-only transformer model"""

    def __init__(
        self,
        context_len: int,
        n_embd: int,
        n_head: int,
        n_layers: int,
        vocab_size: int,
        ff_factor: int = 4,
        dropout: float = 0.1,
        pos_enc: Literal[
            "abs",
            "learned",
            "abacus",
            "nope",
            "rel",
            "rotary",
            "cope",
            "alibi",
            "random",
        ] = "abs",
        pos_enc_max_shift: int = 0,
        emb_type: Literal["learned", "sinusoidal"] = "learned",
    ):
        """
        Arguments:
            context_len: context length, i.e. the number of expected features in the input
            n_embd: dimensionality of model embeddings
            n_head: number of heads in the multi-head attention
            n_layers: number of layers
            vocab_size: size of the vocabulary
            ff_factor: factor by which to scale the hidden layer dimensionality in the feedforward layer
            dropout: dropout probability
            pos_enc: type of positional encoding to use
        """

        super().__init__()
        self.context_len = context_len
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_layers = n_layers
        self.vocab_size = vocab_size
        self.ff_factor = ff_factor
        self.dropout = dropout
        self.enc_dec = False
        self.pos_enc = pos_enc

        # embedding
        # default is legacy name
        if emb_type == "learned" or emb_type == "default":
            self.embedding = nn.Embedding(vocab_size, n_embd)
        elif emb_type == "sinusoidal":
            self.embedding = SinusoidalEmbedding(vocab_size, n_embd)

        # positional encoding
        if self.pos_enc == "abs":
            self.pos_encoder = AbsolutePositionalEncoding(
                n_embd,
                max_len=context_len,
                dropout=dropout,
                max_shift=pos_enc_max_shift,
            )
        elif self.pos_enc == "learned":
            self.pos_encoder = LearnedPositionalEncoding(
                n_embd,
                max_len=context_len,
                dropout=dropout,
                max_shift=pos_enc_max_shift,
            )
        elif self.pos_enc == "abacus":
            self.pos_encoder = AbacusEncoding(
                embedding_dim=n_embd,
                max_seq_length=context_len,
                max_k=30,
            )
        elif self.pos_enc == "random":
            self.pos_encoder = RandomizedPositionalEncoding(
                d_model=n_embd,
                max_len=context_len,
                dropout=dropout,
            )
        elif self.pos_enc == "nope":
            self.pos_encoder = nn.Identity()

        # same as decoder layer essentially, but without cross attention
        layer = nn.TransformerEncoderLayer(
            d_model=n_embd,
            nhead=n_head,
            dim_feedforward=n_embd * ff_factor,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer_encoder = nn.TransformerEncoder(
            layer,
            num_layers=n_layers,
            norm=nn.LayerNorm(n_embd),
        )

        # change all self-attention layers to relative multi-head attention
        if self.pos_enc == "rel":
            for layer in self.transformer_encoder.layers:
                layer.self_attn = RelativeMultiheadAttention(
                    n_embd,
                    n_head,
                    dropout=dropout,
                    bias=True,  # is true by default
                    batch_first=True,
                    rel_pos_k=128,
                )
        elif self.pos_enc == "rotary":
            self.rotary_emb = RotaryEmbedding(dim=self.n_embd / self.n_head)
            for layer in self.transformer_encoder.layers:
                layer.self_attn = RotaryMultiheadAttention(
                    n_embd,
                    n_head,
                    dropout=dropout,
                    bias=True,  # is true by default
                    batch_first=True,
                    rotary_emb=self.rotary_emb,
                )
        elif self.pos_enc == "cope":
            for layer in self.transformer_encoder.layers:
                layer.self_attn = ContextualMultiheadAttention(
                    n_embd,
                    n_head,
                    dropout=dropout,
                    bias=True,  # is true by default
                    batch_first=True,
                    npos_max=128,
                )
        elif self.pos_enc == "alibi":
            for layer in self.transformer_encoder.layers:
                layer.self_attn = AlibiMultiheadAttention(
                    n_embd,
                    n_head,
                    dropout=dropout,
                    bias=True,  # is true by default
                    batch_first=True,
                )

        # output to vocab dim
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

        # weight tying
        if self.embedding is not None:
            self.lm_head.weight = self.embedding.weight

        self.apply(init_weights)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len]``

        Returns:
            logits: Tensor, shape ``[batch_size, seq_len, vocab_size]``
        """
        ids = x.detach().clone()
        x = self.embedding(x)
        if self.pos_enc in ["abs", "learned"]:
            x = self.pos_encoder(x)
        if self.pos_enc == "abacus":
            x += self.pos_encoder(ids)

        x = self.transformer_encoder(
            x,
            mask=nn.Transformer.generate_square_subsequent_mask(x.size(1)).to(
                x.device
            ),  # use to instead of passing device= due to bug that returns NaNs on MPS
        )
        x = self.lm_head(x)
        return x

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
