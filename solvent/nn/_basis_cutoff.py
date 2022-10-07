"""
STATUS: NOT TESTED

"""

import math
import torch


def basis_cutoff(edge_lengths: torch.Tensor, max_radius: float) -> torch.Tensor:
    """
    A smooth function that returns zero at the max radius.

    Args:
        edge_lengths (torch.Tensor): A tensor of edge lengths.
        max_radius (float): Maximum interaction radius.

    Returns:
        y (torch.Tensor): out

    """
    x = edge_lengths / max_radius
    u = 2 * (x - 1)
    y = (math.pi * u).cos().neg().add(1).div(2)
    y[u > 0] = 0
    y[u < -1] = 1

    return y
