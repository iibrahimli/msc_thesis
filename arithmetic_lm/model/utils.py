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
    ckpt_path: str, model_class: str = "TransformerDecoder"
) -> tuple[torch.nn.Module, dict]:
    from arithmetic_lm.model import MODELS

    # load model
    ckpt = torch.load(ckpt_path, map_location="mps")
    model = MODELS[model_class](
        **ckpt["hyper_parameters"]["model_hparams"],
        # vocab_size=tokenizer.vocab_size,
    )
    # state dict has a prefix "model." in the key names
    model.load_state_dict({k[6:]: v for k, v in ckpt["state_dict"].items()})
    model.eval()
    return model, ckpt["hyper_parameters"]


def find_latest_ckpt(dir_path: str | Path) -> str:
    dir_path = Path(dir_path)
    ckpts = list(dir_path.glob("**/*.ckpt"))
    if not ckpts:
        raise ValueError(f"No checkpoints found in {dir_path}")
    return max(ckpts, key=lambda x: x.stat().st_mtime)
