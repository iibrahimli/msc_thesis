import torch
from torch import Tensor, nn

from .utils import CoordinateEncoding


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

        # embedding (TODO: hardcoded pad index for char tokenizer)
        self.embedding = nn.Embedding(vocab_size, n_embd, padding_idx=99)
        self.pos_encoder = CoordinateEncoding(
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

        # output to vocab dim
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

        # weight tying
        self.lm_head.weight = self.embedding.weight

        # init all weights
        # TODO: for some reason with init both train and val loss get stuck
        # and do not progress, model outputs the same character
        # for Experiment 1 both losses plateau at 2.60
        # self.apply(self._init_weights)
        # # apply special scaled init to the residual projections, per GPT-2 paper
        # for pn, p in self.named_parameters():
        #     if pn.endswith("self_attn.out_proj.weight") or pn.endswith(
        #         "linear2.weight"
        #     ):
        #         torch.nn.init.normal_(
        #             p, mean=0.0, std=0.02 / math.sqrt(2 * self.n_layers)
        #         )

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x: Tensor, target: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len]``
            target: Tensor, shape ``[batch_size, seq_len]`` with the target sequence

        Returns:
            logits: Tensor, shape ``[batch_size, seq_len, vocab_size]``
        """
        # TODO: hardcoded pad token for char tokenizer
        src_padding_mask = x == 99
        tgt_padding_mask = target == 99

        x = self.embedding(x)
        target = self.embedding(target)

        # encoder
        for t in range(self.max_steps):
            x = self.pos_encoder(x, timestep=t)
            x = self.encoder_layer(x, src_key_padding_mask=src_padding_mask)

        # decoder
        for t in range(self.max_steps):
            x = self.pos_encoder(x, timestep=t)
            x = self.decoder_layer(
                target,
                x,
                # tgt_mask=nn.Transformer.generate_square_subsequent_mask(
                #     x.size(1), device=x.device
                # ),
                tgt_is_causal=True,
                tgt_key_padding_mask=tgt_padding_mask,
                memory_key_padding_mask=src_padding_mask,
            )

        x = self.lm_head(x)
        return x

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
