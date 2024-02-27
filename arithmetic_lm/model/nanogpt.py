import math

import lightning as L
import torch
from torch import Tensor, nn

from arithmetic_lm.eval_utils import eval_on_batch
from arithmetic_lm.tokenizer import Tokenizer
from arithmetic_lm.train_utils import lr_cosine_annealing_with_warmup


# from https://pytorch.org/tutorials/beginner/transformer_tutorial.html
class PositionalEncoding(nn.Module):

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


class NanoGPT(nn.Module):
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
        """

        super().__init__()
        self.context_len = context_len
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_layers = n_layers
        self.vocab_size = vocab_size
        self.ff_factor = ff_factor
        self.dropout = dropout

        # embedding
        self.embedding = nn.Embedding(vocab_size, n_embd)
        self.pos_encoder = PositionalEncoding(
            n_embd, max_len=context_len, dropout=dropout
        )

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

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len]``

        Returns:
            logits: Tensor, shape ``[batch_size, seq_len, vocab_size]``
        """
        x = self.embedding(x)
        x = self.pos_encoder(x)

        x = self.transformer_encoder(
            x,
            is_causal=True,
            mask=nn.Transformer.generate_square_subsequent_mask(
                x.size(1), device=x.device
            ),
        )
        x = self.lm_head(x)
        return x

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @torch.inference_mode()
    def generate(
        self,
        idx: Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int = 1,
        stop_token: int = None,
        seed: int = 42,
    ) -> Tensor:
        """
        Take a conditioning sequence of indices idx (tensor of shape [batch, seq_len]) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """

        # TODO implement seed w/ device support

        # unsqueeze
        if idx.ndim == 1:
            idx = idx.unsqueeze(0)

        assert isinstance(idx, torch.Tensor), "idx must be a torch.Tensor"
        assert idx.dim() == 2, "idx must be a 2D tensor of shape [batch, seq_len]"
        assert idx.size(1) <= self.context_len, "sequence length must be <= context_len"
        assert idx.size(0) == 1, "only batch size = 1 supported"

        # keep track of where generated part starts to only return it
        gen_start_idx = idx.size(-1)

        for _ in range(max_new_tokens):
            # crop to context_len if necessary
            if idx.size(1) > self.context_len:
                idx_cond = idx[:, -self.context_len :]
                # can only move by 1, since 1 token is generated
                gen_start_idx = max(0, gen_start_idx - 1)
            else:
                idx_cond = idx

            # logits shape: [batch, seq_len, vocab_size]
            logits = self.forward(idx_cond)

            # get logits at final step and apply temperature
            logits = logits[:, -1, :] / temperature

            # optionally apply top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")

            # apply softmax
            probs = nn.functional.softmax(logits, dim=-1)

            # sample from the distribution
            next_token = torch.multinomial(
                probs,
                num_samples=1,
            )

            # append to the sequence
            idx = torch.cat([idx, next_token], dim=1)

            # stop if stop_token is generated
            if stop_token is not None and next_token.item() == stop_token:
                break

        return idx[:, gen_start_idx:]
