import torch
from torch import nn


def init_weights(module: nn.Module):
    # pass

    # from NanoGPT:
    # if isinstance(module, nn.Linear):
    #     torch.nn.init.xavier_uniform_(module.weight)
    #     if hasattr(module, "bias") and module.bias is not None:
    #         torch.nn.init.zeros_(module.bias)
    # elif isinstance(module, nn.Embedding):
    #     torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    # from torch Transformer:
    for p in module.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
