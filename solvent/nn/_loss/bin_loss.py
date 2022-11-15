"""
STATUS: NOT TESTED

"""

import torch

from solvent.nn import LossMixin


# FIXME: acc, prec, rec
class BinLoss(LossMixin):
    def __init__(self, device: str = 'cuda') -> None:
        self._loss = torch.nn.BCELoss()
        self._device = device
        self._c_loss = torch.zeros(1).to(device)
        self._n = 0

    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        self._c_loss += self._loss(pred, target)
        self._n += 1

    def compute_metrics(self) -> torch.Tensor:
        return self._c_loss.detach() / self._n

    def compute_loss(self) -> torch.Tensor:
        return self._c_loss

    def reset(self) -> None:
        self._c_loss = torch.zeros(1).to(self._device)
        self._n = 0
