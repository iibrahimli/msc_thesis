import random

import torch


def get_torch_device() -> torch.device:
    """Get torch device, in order of MPS, CUDA, CPU"""

    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def is_jupyter_notebook() -> bool:
    """Check if the current environment is a Jupyter notebook"""

    try:
        from IPython import get_ipython

        if "IPKernelApp" not in get_ipython().config:
            return False
    except (ImportError, AttributeError):
        return False
    return True


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility"""

    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def transformer_decoder_param_count(
    context_len: int,
    vocab_size: int,
    hidden_dim: int,
    n_layers: int,
    mlp_mult: int = 4,
):

    n_params = 0

    # embedding
    n_params += vocab_size * hidden_dim

    # assuming abs pos embeddings
    n_params += context_len * hidden_dim

    #### decoder layer
    n_params_layer = 0

    # attention q, k, v + bias
    n_params_layer += 3 * hidden_dim * hidden_dim + 3 * hidden_dim

    # attention out + bias
    n_params_layer += hidden_dim * hidden_dim + hidden_dim

    # mlp: hidden -> (hidden * mlp_mult) and back + biases
    n_params_layer += hidden_dim * hidden_dim * mlp_mult * 2 + hidden_dim * (
        1 + mlp_mult
    )

    # layer norm (x2, each has 2 * hidden_dim params)
    n_params_layer += 2 * hidden_dim * 2

    n_params += n_layers * n_params_layer

    # final layer norm
    n_params += 2 * hidden_dim

    # logit layer + bias
    n_params += hidden_dim * vocab_size + vocab_size

    return n_params


def get_carry_str(a: str, b: str, reverse: bool = False) -> str:
    """
    given a and b (non-reversed), return the carry string
    which contains:
        '.' if there is no carry at that position,
        'c' if there is current generated carry, but no carry from previous position
        'p' if there is carry from previous position, but no current generated carry
        'C' if there is both current generated carry and carry from previous position
    """

    carries = []
    carry = 0

    if reverse:
        a = a[::-1]
        b = b[::-1]

    for i in range(max(len(a), len(b))):
        aa = int(a[i]) if i < len(a) else 0
        bb = int(b[i]) if i < len(b) else 0
        s = aa + bb + carry
        if s >= 10:
            if carry == 1:
                carries.append("C")
            else:
                carries.append("c")
            carry = 1
        else:
            if carry == 1:
                carries.append("p")
            else:
                carries.append(".")
            carry = 0
        print(aa, bb, s, carry, carries)

    if carry == 1:
        carries.append("p")

    res = "".join(carries)

    if reverse:
        res = res[::-1]

    return res
