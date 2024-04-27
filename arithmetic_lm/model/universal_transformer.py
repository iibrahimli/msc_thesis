import torch
from torch import Tensor, nn

from .pos_encoding import CoordinateEncoding, RelativeMultiheadAttention
from .utils import init_weights


class UniversalTransformer(nn.Module):
    """Encoder-decoder universal transformer model"""

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
    ):
        """
        Arguments:
            context_len: context length, i.e. the number of expected features in the input
            n_embd: dimensionality of model embeddings
            n_head: number of heads in the multi-head attention
            max_steps: number of maximum recurrent steps (both enc and dec)
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
        self.enc_dec = True
        self.pos_enc = pos_enc

        # embedding (TODO: hardcoded pad index for char tokenizer)
        self.embedding = nn.Embedding(vocab_size, n_embd, padding_idx=99)
        self.coord_encoder = CoordinateEncoding(
            n_embd, max_len=context_len, dropout=dropout
        )

        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=n_embd,
            nhead=n_head,
            dim_feedforward=n_embd * ff_factor,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=n_embd,
            nhead=n_head,
            dim_feedforward=n_embd * ff_factor,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )

        # change all self-attention layers to relative multi-head attention
        if self.pos_enc == "rel":
            for layer in self.encoder.layers:
                layer.self_attn = RelativeMultiheadAttention(
                    n_embd,
                    n_head,
                    dropout=dropout,
                    bias=True,  # is true by default
                    batch_first=True,
                    rel_pos_k=16,
                )
            for layer in self.decoder.layers:
                layer.self_attn = RelativeMultiheadAttention(
                    n_embd,
                    n_head,
                    dropout=dropout,
                    bias=True,  # is true by default
                    batch_first=True,
                    rel_pos_k=16,
                )
                # TODO: also change cross attention to relative maybe?

        # output to vocab dim
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

        # weight tying
        self.lm_head.weight = self.embedding.weight

        # TODO init all weights
        # self.apply(init_weights)

    def encode(
        self,
        source: Tensor,
        src_padding_mask: Tensor = None,
        max_steps: int = None,
    ) -> tuple[Tensor, Tensor]:
        max_steps = max_steps if max_steps else self.max_steps
        source = self.embedding(source)
        for i in range(max_steps):
            # TODO: this is needed for timestep info, keep even with relative pos enc
            source = self.coord_encoder(source, timestep=i)
            source = self.encoder_layer(source, src_key_padding_mask=src_padding_mask)
        return source  # return memory

    def decode(
        self,
        target: Tensor,
        memory: Tensor,
        tgt_padding_mask: Tensor = None,
        memory_padding_mask: Tensor = None,
        max_steps: int = None,
    ) -> Tensor:
        max_steps = max_steps if max_steps else self.max_steps
        target = self.embedding(target)
        for i in range(max_steps):
            # TODO: this is needed for timestep info, keep even with relative pos enc
            target = self.coord_encoder(target, timestep=i)
            target = self.decoder_layer(
                target,
                memory,
                tgt_mask=nn.Transformer.generate_square_subsequent_mask(
                    target.size(1)
                ).to(target.device),
                tgt_is_causal=True,
                tgt_key_padding_mask=tgt_padding_mask,
                memory_key_padding_mask=memory_padding_mask,
            )

        logits = self.lm_head(target)
        return logits

    def forward(self, source: Tensor, target: Tensor, max_steps: int = None) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len]``
            target: Tensor, shape ``[batch_size, seq_len]`` with the target sequence

        Returns:
            logits: Tensor, shape ``[batch_size, seq_len, vocab_size]``
        """

        # TODO: hardcoded pad token for char tokenizer
        src_padding_mask = source == 99
        tgt_padding_mask = target == 99

        max_steps = max_steps if max_steps else self.max_steps

        # encoder
        memory = self.encode(
            source=source, src_padding_mask=src_padding_mask, max_steps=max_steps
        )

        # decoder
        logits = self.decode(
            target=target,
            memory=memory,
            tgt_padding_mask=tgt_padding_mask,
            memory_padding_mask=src_padding_mask,
            max_steps=max_steps,
        )

        return logits

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
