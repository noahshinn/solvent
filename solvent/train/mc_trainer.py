"""
STATUS: NOT TESTED

"""

import time
import torch
from torch.optim import Adam, SGD
from torch.nn import CrossEntropyLoss
from torch.optim.lr_scheduler import ExponentialLR, ReduceLROnPlateau

from solvent.train import Trainer
from solvent.data import DataLoader
from solvent.nn import MCLoss
from solvent.utils import to_mc
from solvent.logger import MCLogger

from typing import Union, Dict, Optional
from torch_geometric.data.data import Data
from solvent.types import MCPredMetrics


class MCTrainer(Trainer):
    def __init__(
            self,
            root: str,
            run_name: str,
            model: torch.nn.Module,
            train_loader: Union[DataLoader, str],
            test_loader: Union[DataLoader, str],
            nclasses: int,
            optim: Union[Adam, SGD, None] = None,
            scheduler: Union[ExponentialLR, ReduceLROnPlateau, None] = None,
            start_epoch: int = 0,
            start_lr: float = 0.01,
            description: str = '',
            ncores: Optional[int] = None
        ) -> None:
        super().__init__(root, run_name, model, train_loader, test_loader, optim, scheduler, start_epoch, start_lr, description, ncores)
        self._loss = MCLoss(self._device)
        self.__loss = CrossEntropyLoss()
        self._logger = MCLogger(self._log_dir, self._is_resume)
        self._nclasses = nclasses
    
    def log_metrics(
            self,
            train_acc: torch.Tensor,
            test_acc: torch.Tensor,
            train_loss: torch.Tensor,
            test_loss: torch.Tensor,
        ) -> None:
        self._logger.log_epoch(
            epoch=self._epoch,
            lr=self._lr,
            accuracy_train=train_acc,
            accuracy_test=test_acc,
            loss_train=train_loss,
            loss_test=test_loss,
            duration=time.perf_counter() - self._walltime
        )

    # TODO: return size
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
                    `bin`: class

        Returns:
            (torch.Tensor): classification tensor of size (?)

        """
        return self._model(structure)

    def evaluate(self, loader: DataLoader, mode: str) -> MCPredMetrics:
        """
        Full pass through a data set.

        Args:
            loader (DataLoader): An iterable for a series of structures.
            mode (str): One of 'TRAIN' or 'TEST'

        Returns:
            acc (torch.Tensor),
            prec (torch.Tensor),
            rec (torch.Tensor),
            f1 (torch.Tensor)

        Asserts:
            - `mode` is one of 'TRAIN' or 'TEST'

        """
        assert mode == 'TRAIN' or mode == 'TEST'
        for structure in loader:
            structure.to(self._device)
            pred = self.pred(structure)
            print(f'pred: {pred}')
            print(pred.dtype)
            print(f"target: {to_mc(structure['bin'].to(self._device), self._nclasses)}")
            print(to_mc(structure['bin'].to(self._device), self._nclasses).dtype)
            self._loss(
                pred,
                to_mc(structure['bin'].to(self._device), self._nclasses)
            )
            if mode == 'TRAIN':
                loss = self.__loss(pred, to_mc(structure['bin'].to(self._device), self._nclasses))
                loss.backward()
                self.step()
        acc, _loss = self._loss.compute_metrics()
        return MCPredMetrics(acc, _loss)

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
            acc_train, loss_train = self.evaluate(loader=self._train_loader, mode='TRAIN') # type: ignore
            acc_test, loss_test = self.evaluate(loader=self._test_loader, mode='TEST') # type: ignore

            self.log_metrics(
                train_acc=acc_train,
                test_acc=acc_test,
                train_loss=loss_train,
                test_loss=loss_test
            )

            if acc_test < self._best_metric:
                self._best_metric = acc_test
                self.chkpt()

            self.update(self._loss.compute_loss())

        return self._exit_code
