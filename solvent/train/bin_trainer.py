"""
STATUS: NOT TESTED

"""

import time
import torch
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import ExponentialLR, ReduceLROnPlateau

from solvent.train import Trainer
from solvent.data import DataLoader
from solvent.nn import BinLoss
from solvent.logger import BinLogger
from solvent.utils import to_bin
from solvent.types import BinPredMetrics

from typing import Union, Dict
from torch_geometric.data.data import Data


class BinTrainer(Trainer):
    def __init__(
            self,
            root: str,
            run_name: str,
            model: torch.nn.Module,
            train_loader: Union[DataLoader, str],
            test_loader: Union[DataLoader, str],
            optim: Union[Adam, SGD, None] = None,
            scheduler: Union[ExponentialLR, ReduceLROnPlateau, None] = None,
            start_epoch: int = 0,
            start_lr: float = 0.01,
            description: str = ''
        ) -> None:
        super().__init__(root, run_name, model, train_loader, test_loader, optim, scheduler, start_epoch, start_lr, description)
        self._loss = BinLoss(self._device)
        self._logger = BinLogger(self._log_dir, self._is_resume)
    
    def log_metrics(
            self,
            accuracy_train: torch.Tensor,
            accuracy_test: torch.Tensor,
            precision_train: torch.Tensor,
            precision_test: torch.Tensor,
            recall_train: torch.Tensor,
            recall_test: torch.Tensor,
            f1_train: torch.Tensor,
            f1_test: torch.Tensor
        ) -> None:
        self._logger.log_epoch(
            epoch=self._epoch,
            lr=self._lr,
            accuracy_train=accuracy_train,
            accuracy_test=accuracy_test,
            precision_train=precision_train,
            precision_test=precision_test,
            recall_train=recall_train,
            recall_test=recall_test,
            f1_train=f1_train,
            f1_test=f1_test,
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
                    `is_like_zero`: binary classification 

        Returns:
            (torch.Tensor): A binary label.

        """
        out = self._model(structure)
        return to_bin(out)

    def evaluate(self, loader: DataLoader, mode: str) -> BinPredMetrics:
        """
        Full pass through a data set.

        Args:
            loader (DataLoader): An iterable for a series of structures.
            mode (str): One of 'TRAIN' or 'TEST'

        Returns:
            acc (torch.Tensor), prec (torch.Tensor), rec (torch.Tensor), f1 (torch.Tensor)

        Asserts:
            - `mode` is one of 'TRAIN' or 'TEST'

        """
        assert mode == 'TRAIN' or mode == 'TEST'
        for structure in loader:
            structure.to(self._device)
            label = self.pred(structure)
            self._loss(
                label,
                structure['is_like_zero'].to(self._device)
            )
            if mode == 'TRAIN':
                self.step(loss=self._loss.compute_loss())
        acc, prec, rec, f1 = self._loss.compute_metrics()
        return BinPredMetrics(acc, prec, rec, f1)

    def update(self, loss: torch.Tensor) -> None:
        self._walltime = time.perf_counter()
        if isinstance(self._scheduler, ReduceLROnPlateau):
            self._scheduler.step(metrics=loss)
        else:
            self._scheduler.step()
        self._epoch += 1
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
            acc_train, prec_train, rec_train, f1_train = self.evaluate(loader=self._train_loader, mode='TRAIN') # type: ignore
            acc_test, prec_test, rec_test, f1_test = self.evaluate(loader=self._test_loader, mode='TEST') # type: ignore

            self.log_metrics(
                accuracy_train=acc_train,
                accuracy_test=acc_test,
                precision_train=prec_train,
                precision_test=prec_test,
                recall_train=rec_train,
                recall_test=rec_test,
                f1_train=f1_train,
                f1_test=f1_test
            )

            self.update(self._loss.compute_loss())

            if (1 - acc_test) < self._best_metric:
                self._best_metric = acc_test
                self.chkpt()

        return self._exit_code
