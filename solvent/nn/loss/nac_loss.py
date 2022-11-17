"""
STATUS: NOT TESTED

"""

import torch

from solvent.utils import mse
from ._loss import LossMixin


class NACLoss(LossMixin):
    def __init__(self, device: str = 'cuda') -> None:
        self._device = device

        self._c_loss = torch.zeros(1).to(device)
        self._n = 0

    def __call__(self, nac_pred: torch.Tensor, nac_target: torch.Tensor) -> None:
        self._c_loss += mse(nac_pred, nac_target)
        self._n += 1

    def compute_metrics(self) -> torch.Tensor:
        return self._c_loss / self._n

    def compute_loss(self) -> torch.Tensor:
        return self._c_loss
    
    def reset(self) -> None:
        self._c_loss = torch.zeros(1).to(self._device)
        self._n = 0
