import math

import torch
from torch import Tensor, nn


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


@torch.inference_mode()
def generate(
    model: nn.Module,
    idx: Tensor,
    max_new_tokens: int,
    encoder_source: Tensor = None,
    temperature: float = 1.0,
    top_k: int = 1,
    stop_token: int = None,
    seed: int = 42,
) -> Tensor:
    """
    Take a conditioning sequence of indices idx (tensor of shape [batch, seq_len]) and complete
    the sequence max_new_tokens times, feeding the predictions back into the model each time.
    Most likely you'll want to make sure to be in model.eval() mode of operation for this.

    encoder_source hints that the model is an encoder-decoder model and the encoder_source
    will be encoded and used as memory for the decoder.
    """

    # TODO implement seed w/ device support

    # unsqueeze
    if idx.ndim == 1:
        idx = idx.unsqueeze(0)

    if isinstance(stop_token, list) and len(stop_token) == 1:
        stop_token = stop_token[0]

    assert isinstance(idx, torch.Tensor), "idx must be a torch.Tensor"
    assert idx.dim() == 2, "idx must be a 2D tensor of shape [batch, seq_len]"
    assert idx.size(1) <= model.context_len, "sequence length must be <= context_len"
    assert idx.size(0) == 1, "only batch size = 1 supported"

    # keep track of where generated part starts to only return it
    gen_start_idx = idx.size(-1)

    # get hidden state from encoder
    if encoder_source is not None:
        # TODO: hardcoded pad token for char tokenizer
        # don't care about masks for now since only batch size = 1
        memory = model.encode(encoder_source)

    for _ in range(max_new_tokens):
        # crop to context_len if necessary
        if idx.size(1) > model.context_len:
            idx_cond = idx[:, -model.context_len :]
            # can only move by 1, since 1 token is generated
            gen_start_idx = max(0, gen_start_idx - 1)
        else:
            idx_cond = idx

        # logits shape: [batch, seq_len, vocab_size]
        if encoder_source is not None:
            # enc-dec model
            logits = model.decode(idx_cond, memory)
        else:
            logits = model(idx_cond)

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
