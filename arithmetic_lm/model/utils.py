import math
from pathlib import Path

import torch
from torch import nn


def init_weights(module: nn.Module):
    # default pytorch initialization
    # for linear layers: uniform(-1/sqrt(in_features), 1/sqrt(in_features))
    pass

    # from NanoGPT:
    # if isinstance(module, nn.Linear):
    #     torch.nn.init.xavier_uniform_(module.weight)
    #     if hasattr(module, "bias") and module.bias is not None:
    #         torch.nn.init.zeros_(module.bias)
    # elif isinstance(module, nn.Embedding):
    #     torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    # from torch Transformer:
    # for p in module.parameters():
    #     if p.dim() > 1:
    #         nn.init.xavier_uniform_(p)

    # from lucidrains x-transformers:
    # for p in module.parameters():
    #     if p.dim() > 1:
    #         nn.init.kaiming_normal_(p)


def load_model(
    ckpt_path: str, model_class: str = "TransformerDecoder", map_location: str = "mps"
) -> tuple[torch.nn.Module, dict]:
    from arithmetic_lm.model import MODELS

    # load model
    ckpt = torch.load(ckpt_path, map_location=map_location)
    model = MODELS[model_class](
        **ckpt["hyper_parameters"]["model_hparams"],
        # vocab_size=tokenizer.vocab_size,
    )
    # state dict has a prefix "model." in the key names
    model.load_state_dict(
        {k[6:]: v for k, v in ckpt["state_dict"].items() if k.startswith("model.")}
    )
    model.eval()
    return model, ckpt["hyper_parameters"]


def find_latest_ckpt(dir_path: str | Path) -> str:
    dir_path = Path(dir_path)
    ckpts = list(dir_path.glob("**/*.ckpt"))
    if not ckpts:
        raise ValueError(f"No checkpoints found in {dir_path}")
    return max(ckpts, key=lambda x: x.stat().st_mtime)


class SinusoidalEmbedding(nn.Embedding):
    """Like regular emb, but use sinusoidal embeddings for tokens 0-9"""

    def __init__(
        self, num_embeddings: int, embedding_dim: int, padding_idx: int = None
    ):
        super().__init__(num_embeddings, embedding_dim, padding_idx=padding_idx)

        # compute sinusoidal embeddings for tokens 0-9
        n_digits = 10
        position = torch.arange(n_digits).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embedding_dim, 2) * (-math.log(10.0) / embedding_dim)
        )

        with torch.no_grad():
            self.weight[:n_digits, 0::2] = torch.sin(position * div_term)
            self.weight[:n_digits, 1::2] = torch.cos(position * div_term)


def answer_mask(
    target: torch.Tensor, pad_token_id: int, equal_token_id: int
) -> torch.Tensor:
    """
    Mask out all tokens except answer tokens, i.e. set everything before the
    equal sign token to pad_token_id since pad token is ignored in loss.

    Args:
        target: target tensor of shape (batch_size, seq_len)
        pad_token_id: padding token id
        equal_token_id: equal sign token id, after which the answer starts
    """

    # find the first equal sign token
    equal_mask = target == equal_token_id
    before_equal_mask = equal_mask.cumsum(dim=1) == 0

    # mask out everything before the equal sign token
    target[before_equal_mask] = pad_token_id

    return target


def get_linear_lsd_weight(coeff: float, tgt: torch.Tensor, pad_token_id: int):
    """
    Get a linear weight to the sequence positions in the loss tensor to
    encourage predicting first digits first. Ignore padding tokens.
    """

    # get mask for non-padding tokens
    mask = tgt != pad_token_id

    # weight = 1 + coeff * (1 - position / max_position)
    # position = 0 for first token
    # position = max_position for last token
    max_position = mask.sum(1, keepdim=True)
    position = torch.arange(tgt.size(1), device=tgt.device).unsqueeze(0)
    weight = 1 + coeff * (1 - position.float() / max_position.float())

    return weight
