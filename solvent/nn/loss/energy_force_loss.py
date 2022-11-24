"""
STATUS: NOT TESTED

"""

import torch

from solvent.utils import mae, mse
from solvent.types import QMPredMAE
from ._loss import LossMixin


class EnergyForceLoss(LossMixin):
    def __init__(
            self,
            energy_contribution: float = 0.0,
            force_contribution: float = 0.0,
            device: str = 'cuda'
        ) -> None:
        self._e_contrib = energy_contribution
        self._f_contrib = force_contribution
        self._device = device

        self._c_e_mae = torch.zeros(1).to(device)
        self._c_e_mse = torch.zeros(1).to(device)
        self._c_f_mae = torch.zeros(1).to(device)
        self._c_f_mse = torch.zeros(1).to(device)
        self._n = 0

    def __call__(
            self,
            e_pred: torch.Tensor,
            e_target: torch.Tensor,
            f_pred: torch.Tensor,
            f_target: torch.Tensor,
        ) -> None:
        self._c_e_mae += mae(e_pred, e_target).detach()
        self._c_e_mse += mse(e_pred, e_target).detach()
        self._c_f_mae += mae(f_pred, f_target).detach()
        self._c_f_mse += mse(f_pred, f_target).detach()
        self._n += 1

    def compute_metrics(self) -> QMPredMAE:
        e_mae = self._c_e_mae / self._n
        f_mae = self._c_f_mae / self._n
        return QMPredMAE(e_mae, f_mae)

    def compute_loss(self) -> torch.Tensor:
        e_loss = self._e_contrib * self._c_e_mse
        f_loss = self._f_contrib * self._c_f_mse
        self._c_e_mse = torch.zeros(1).to(self._device)
        self._c_f_mse = torch.zeros(1).to(self._device)
        return e_loss + f_loss
    
    def reset(self) -> None:
        self._c_e_mse = torch.zeros(1).to(self._device)
        self._c_f_mse = torch.zeros(1).to(self._device)
        self._c_e_mae = torch.zeros(1).to(self._device)
        self._c_f_mae = torch.zeros(1).to(self._device)
        self._n = 0
