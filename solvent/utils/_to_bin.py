"""
STATUS: FINISHED

"""

import torch


def to_bin(x: torch.Tensor) -> torch.Tensor:
    """Converts a sigmoid output to a binary classification"""
    return x.reshape(-1).detach().round()
