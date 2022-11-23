"""
STATUS: NOT TESTED

"""

import torch

from typing import Union


def normalize(x: torch.Tensor, mu: Union[float, torch.Tensor], std: Union[float, torch.Tensor]) -> torch.Tensor:
    """Normalizes an input"""
    return (x - mu) / std
