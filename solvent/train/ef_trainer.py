"""
STATUS: NOT TESTED

"""

import time
import torch
from torch.optim import (
    Adam,
    SGD
)
from torch.optim.lr_scheduler import (
    ExponentialLR,
    ReduceLROnPlateau
)

from solvent.nn import EnergyForceLoss, force_grad
from solvent.types import EnergyForcePrediction, QMPredMAE
from solvent.train import Trainer

from typing import Dict, Union
from torch_geometric.loader import DataLoader
from torch_geometric.data.data import Data


class EFTrainer(Trainer):
    def __init__(
            self,
            root: str,
            run_name: str,
            model: torch.nn.Module,
            train_loader: Union[DataLoader, str],
            test_loader: Union[DataLoader, str],
            optim: Union[Adam, SGD, None] = None,
            scheduler: Union[ExponentialLR, ReduceLROnPlateau, None] = None,
            energy_contribution: float = 1.0,
            force_contribution: float = 1.0,
            energy_scale: float = 1.0,
            force_scale: float = 1.0,
            nmol: int = 1,
            units: str = 'hartree',
            start_epoch: int = 0,
            start_lr: float = 1e-2,
            chkpt_freq: int = 1,
            description: str = ''
        ) -> None:
        super().__init__(
            root=root,
            run_name=run_name,
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            optim=optim,
            scheduler=scheduler,
            start_epoch=start_epoch,
            start_lr=start_lr,
            chkpt_freq=chkpt_freq,
            description=description
        )
        self._loss = EnergyForceLoss(
            energy_contribution=energy_contribution,
            force_contribution=force_contribution,
            device='cuda:0' if torch.cuda.is_available() else 'cpu'
        )
        self._e_scale = energy_scale
        self._f_scale = force_scale
        self._nmol = nmol 
        self._units = units 

    def pred(self, structure: Union[Dict, Data]) -> EnergyForcePrediction:
        """
        Evaluates the model.

        N: Number of atoms in the system.
        M: Number of unique chemical species types
        K: Number of electronic states.

        Args:
            structure (Union[Dict, Data]): An atomic system represented as either
                a Python dictionary or torch-geometric Data object with the following
                data fields:
                    `x`: one-hot vector of size (M)
                    `pos`: coordinates of size (N, 3)
                    `energies`: energy vector of size (K)
                    `forces`: force vector of size (K, N, 3)

        Returns:
            e (torch.Tensor), f (torch.Tensor): energy and force tensor of size (K)
                and (K, N, 3), respectively

        """
        e = self._model(structure)
        f = force_grad(e, structure['pos'], self._device)
        return EnergyForcePrediction(e, f)

    def evaluate(self, loader: DataLoader, mode: str) -> QMPredMAE:
        """
        Full pass through a data set.

        Args:
            loader (DataLoader): An iterable for a series of structures.
            mode (str): One of 'TRAIN' or 'TEST'

        Returns:
            e_mae (torch.Tensor), f_mae (torch.Tensor): Energy force mean absolute
                error.

        Asserts:
            - `mode` is one of 'TRAIN' or 'TEST'

        """
        assert mode == 'TRAIN' or mode == 'TEST'
        for structure in loader:
            structure['pos'].requires_grad = True
            structure.to(self._device)
            e, f = self.pred(structure)
            self._loss(
                e_pred=e,
                e_target=structure['energies'].to(self._device),
                f_pred=f,
                f_target=structure['forces'].to(self._device)
            )
            if mode == 'TRAIN':
                self.step(loss=self._loss.compute_loss())

        e_mae, f_mae = self._loss.compute_metrics()
        return QMPredMAE(e_mae, f_mae)

    def log_metrics(
            self,
            e_train_mae: torch.Tensor,
            e_test_mae: torch.Tensor,
            f_train_mae: torch.Tensor,
            f_test_mae: torch.Tensor,
        ) -> None:
        """
        Formats the energy and force metrics for proper logging.
            - scales the energy error according to the number of molecules in
                the system
            - returns the energy and force in terms of the given unit of energy

        Args:
            e_train_mae (torch.Tensor): Train set energy mean absolute error.
            e_test_mae (torch.Tensor): Test set energy mean absolute error.
            f_train_mae (torch.Tensor): Train set force mean absolute error.
            f_test_mae (torch.Tensor): Test set force mean absolute error.

        Returns:
            (None)

        """
        self._logger.log_epoch(
            epoch=self._epoch,
            lr=self._lr,
            e_train_mae=e_train_mae * self._e_scale / self._nmol,
            e_test_mae=e_test_mae * self._e_scale / self._nmol,
            f_train_mae=f_train_mae * self._f_scale,
            f_test_mae=f_test_mae * self._f_scale,
            duration=time.perf_counter() - self._walltime
        )

    def update(self, loss: torch.Tensor) -> None:
        """
        Updates after every epoch.
            - resets wall time
        
        Args:
            loss (torch.Tensor): Loss tensor from forward pass

        Returns:
            (None)

        """
        self._walltime = time.perf_counter()
        if isinstance(self._scheduler, ReduceLROnPlateau):
            self._scheduler.step(metrics=loss)
        else:
            self._scheduler.step()
        self._loss.reset()
        self._epoch += 1
        self._cur_chkpt_count += 1
        self._lr = self._optim.param_groups[0]['lr']

    def fit(self) -> str:
        """
        TODO

        Args:
            None

        Returns:
            (str): exit code

        """
        while not self.should_terminate():
            e_train_mae, f_train_mae = self.evaluate(loader=self._train_loader, mode='TRAIN')
            e_test_mae, f_test_mae = self.evaluate(loader=self._test_loader, mode='TEST')

            self.log_metrics(
                e_train_mae=e_train_mae,
                e_test_mae=e_test_mae,
                f_train_mae=f_train_mae,
                f_test_mae=f_test_mae,
            )

            self.update(self._loss.compute_loss())

            if self._cur_chkpt_count == self._chkpt_freq:
                self.chkpt()

        return self._exit_code
