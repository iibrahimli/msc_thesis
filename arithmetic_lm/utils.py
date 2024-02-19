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
