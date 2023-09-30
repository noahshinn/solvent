import torch

from solvent.utils import mae, mse
from ._loss import LossMixin

from solvent.types import NACPredMetrics


class NACLoss(LossMixin):
    def __init__(self, device: str = 'cuda') -> None:
        self._device = device

        self._c_mae = torch.zeros(1).to(device)
        self._c_mse = torch.zeros(1).to(device)
        self._n = 0

    def __call__(self, nac_pred: torch.Tensor, nac_target: torch.Tensor) -> None:
        self._c_mae += mae(nac_pred, nac_target).detach()
        self._c_mse += mse(nac_pred, nac_target).detach()
        self._n += 1

    def compute_metrics(self) -> NACPredMetrics:
        return NACPredMetrics(self._c_mae / self._n, self._c_mse / self._n)

    def compute_loss(self) -> torch.Tensor:
        return self._c_mse
    
    def reset(self) -> None:
        self._c_mae = torch.zeros(1).to(self._device)
        self._c_mse = torch.zeros(1).to(self._device)
        self._n = 0
