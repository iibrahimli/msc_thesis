import torch
from torch import nn


def init_weights(module: nn.Module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
        torch.nn.init.xavier_uniform_(module.weight)
        if hasattr(module, "bias") and module.bias is not None:
            torch.nn.init.zeros_(module.bias)
    # elif isinstance(module, nn.Embedding):
    #     torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
