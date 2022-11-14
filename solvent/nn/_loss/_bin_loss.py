"""
STATUS: NOT TESTED

"""

import torch

from solvent.nn import LossMixin


class BinLoss(LossMixin):
    def __init__(self) -> None:
        self._loss = torch.nn.BCELoss()

    def __call__(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self._loss(input, target)

    def reset(self) -> None:
        return
