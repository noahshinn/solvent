"""
STATUS: NOT TESTED

"""

import time
import torch
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import ExponentialLR, ReduceLROnPlateau

from solvent.train import Trainer
from solvent.data import DataLoader
from solvent.nn import NACLoss
from solvent.utils import mse, normalize
from solvent.logger import NACLogger

from typing import Union, Dict, Optional
from torch_geometric.data.data import Data
from solvent.types import NACPredMetrics 


class NACTrainer(Trainer):
    def __init__(
            self,
            root: str,
            run_name: str,
            model: torch.nn.Module,
            train_loader: Union[DataLoader, str],
            test_loader: Union[DataLoader, str],
            optim: Union[Adam, SGD, None] = None,
            scheduler: Union[ExponentialLR, ReduceLROnPlateau, None] = None,
            mu: Union[float, torch.Tensor] = 0.0,
            std: Union[float, torch.Tensor] = 1.0,
            start_epoch: int = 0,
            start_lr: float = 0.01,
            chkpt_freq: int = 1,
            description: str = '',
            ncores: Optional[int] = None
        ) -> None:
        super().__init__(root, run_name, model, train_loader, test_loader, optim, scheduler, start_epoch, start_lr, chkpt_freq, description, ncores)
        self._loss = NACLoss(self._device)
        self._logger = NACLogger(self._log_dir, self._is_resume)
        
        # for target value scaling and shifting
        self._mu = mu
        self._std = std
    
    def log_metrics(
            self,
            train_nac_mae: torch.Tensor,
            test_nac_mae: torch.Tensor,
            train_nac_mse: torch.Tensor,
            test_nac_mse: torch.Tensor,
        ) -> None:
        self._logger.log_epoch(
            epoch=self._epoch,
            lr=self._lr,
            train_mae=train_nac_mae,
            test_mae=test_nac_mae,
            train_mse=train_nac_mse,
            test_mse=test_nac_mse,
            duration=time.perf_counter() - self._walltime
        )

    def pred(self, structure: Union[Dict, Data]) -> torch.Tensor:
        """
        Evaluates the model.

        N: Number of atoms in the system.
        M: Number of unique chemical species types

        Args:
            structure (Union[Dict, Data]): An atomic system represented as either
                a Python dictionary or torch-geometric Data object with the following
                data fields:
                    `x`: one-hot vector of size (M)
                    `pos`: coordinates of size (N, 3)
                    `nacs`: energy vector of size (N * 3)

        Returns:
            (torch.Tensor): Derivative coupling vector of size (N, 3)

        """
        return self._model(structure)

    def evaluate(self, loader: DataLoader, mode: str) -> NACPredMetrics:
        """
        Full pass through a data set.

        Args:
            loader (DataLoader): An iterable for a series of structures.
            mode (str): One of 'TRAIN' or 'TEST'

        Returns:
            nac_mae (torch.Tensor), nac_mse (torch.Tensor)

        Asserts:
            - `mode` is one of 'TRAIN' or 'TEST'

        """
        assert mode == 'TRAIN' or mode == 'TEST'
        for structure in loader:
            structure.to(self._device)
            pred = self.pred(structure)
            self._loss(
                pred,
                normalize(structure['nacs'].to(self._device), self._mu, self._std)
            )
            if mode == 'TRAIN':
                loss = mse(pred, normalize(structure['nacs'].to(self._device), self._mu, self._std))
                loss.backward()
                self.step()
        nac_mae, nac_mse = self._loss.compute_metrics()
        return NACPredMetrics(nac_mae, nac_mse)

    def step(self) -> None:
        self._optim.step()
        self._optim.zero_grad()

    def update(self, loss: torch.Tensor) -> None:
        self._walltime = time.perf_counter()
        if isinstance(self._scheduler, ReduceLROnPlateau):
            self._scheduler.step(metrics=loss)
        else:
            self._scheduler.step()
        self._loss.reset()
        self._epoch += 1
        self._cur_chkpt_count += 1
        self._lr = self._optim.param_groups[0]['lr']

    # TODO: figure out type errors
    def fit(self) -> str:
        """
        Args:
            None

        Returns:
            (str): exit code

        """
        while not self.should_terminate():
            nac_mae_train, nac_mse_train = self.evaluate(loader=self._train_loader, mode='TRAIN') # type: ignore
            nac_mae_test, nac_mse_test = self.evaluate(loader=self._test_loader, mode='TEST') # type: ignore

            self.log_metrics(
                train_nac_mae=nac_mae_train,
                test_nac_mae=nac_mae_test,
                train_nac_mse=nac_mse_train,
                test_nac_mse=nac_mse_test,
            )

            if self._cur_chkpt_count == self._chkpt_freq:
                self.chkpt()

            self.update(self._loss.compute_loss())

        return self._exit_code
