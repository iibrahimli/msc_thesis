"""Training utility functions"""

import math

import numpy as np
import torch


def lr_cosine_annealing_with_warmup(
    it: int,
    learning_rate: float,
    warmup_iters: int,
    lr_decay_iters: int,
    min_lr: float = None,
):
    if min_lr is None:
        min_lr = learning_rate / 10

    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters

    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr

    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1

    return min_lr + coeff * (learning_rate - min_lr)


@torch.no_grad
def insert_pause_tokens(
    x: torch.Tensor,
    pause_token_id: int,
    n_pause_tokens: int,
    pause_type: str = "pretrain",
) -> torch.Tensor:
    """
    Insert pause tokens, slow stupid implementation, NOTE: also inserts
    pause tokens in padded regions

    Args:
        x: input tensor of shape [B, L]
        n_pause_tokens: number of pause tokens per sequence
        pause_type: "pretrain" or "finetune"

    Returns:
        Tensor of shape [B, L + n_pause_tokens]
    """

    if n_pause_tokens == 0:
        return x

    if pause_type != "pretrain":
        raise NotImplementedError(f"pause-finetuning not supported yet")

    l = x.size(1)

    # select random positions from range [0, L]
    pos = set(np.random.choice(l, n_pause_tokens, replace=False))

    # indices
    idx = []
    for i in range(l):
        if i in pos:
            # insert dummy known value
            idx.append(-1)
        idx.append(i)
    idx = torch.tensor(idx, dtype=int)

    res = x[:, idx]
    res[:, idx == -1] = pause_token_id

    return res
