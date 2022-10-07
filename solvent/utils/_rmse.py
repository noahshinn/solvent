"""
STATUS: PRODUCTION

"""

import torch


def rmse(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    """
    Root mean squared error.

    Args:
        x1 (torch.Tensor): first input
        x2 (torch.Tensor): second input

    Returns:
        (torch.Tensor)

    """
    return (x1 - x2).pow(2).mean().sqrt()
