"""
STATUS: NOT TESTED

"""

import torch

from typing import Union


def renormalize(
        x: torch.Tensor,
        mu: Union[float, torch.Tensor],
        std: Union[float, torch.Tensor],
    ) -> torch.Tensor:
    """Rescales and reshifts the given value"""
    return x * std + mu
