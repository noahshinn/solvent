"""
STATUS: PRODUCTION

"""

import torch


def mae(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    """
    Mean absolute error.

    Args:
        x1 (torch.Tensor): first input
        x2 (torch.Tensor): second input

    Returns:
        (torch.Tensor)

    """
    return (x1 - x2).abs().mean()
