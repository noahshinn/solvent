"""
STATUS: NOT TESTED

"""

import torch

from typing import Union


def to_mc(label: Union[int, torch.Tensor], nclasses: int) -> torch.Tensor:
    """Converts a label to a one-hot label tensor"""
    if isinstance(label, torch.Tensor):
        assert label.size(dim=0) == 1
    return torch.nn.functional.one_hot(label, num_classes=nclasses).flatten().to(torch.float32)
