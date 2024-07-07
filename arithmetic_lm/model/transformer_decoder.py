from torch import Tensor, nn

from arithmetic_lm.model.pos_encoding import (
    AbacusEncoding,
    AbsolutePositionalEncoding,
    LearnedPositionalEncoding,
    RelativeMultiheadAttention,
)
from arithmetic_lm.model.rotary_pos_encoding import RotaryMultiheadAttention
from arithmetic_lm.model.utils import init_weights


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
        pos_enc: str = "abs",
        pos_enc_max_shift: int = 0,
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
        self.embedding = nn.Embedding(vocab_size, n_embd // 2)  # TODO: revert to n_embd
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
                embedding_dim=n_embd, max_seq_length=context_len
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
                    rel_pos_k=16,
                )
        elif self.pos_enc == "rotary":
            for layer in self.transformer_encoder.layers:
                layer.self_attn = RotaryMultiheadAttention(
                    n_embd,
                    n_head,
                    dropout=dropout,
                    bias=True,  # is true by default
                    batch_first=True,
                )

        # output to vocab dim
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

        # weight tying
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
        print(x.shape)
        if self.pos_enc == "abs":
            x = self.pos_encoder(x)

        print("before enc", x.shape)
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
