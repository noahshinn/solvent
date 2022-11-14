"""
STATUS: DEV

"""

import torch
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import ExponentialLR, ReduceLROnPlateau

from solvent.train import Trainer
from solvent.data import DataLoader
from solvent.nn import BinLoss
from solvent.logger import Logger

from typing import Union


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
            chkpt_freq: int = 1,
            description: str = ''
        ) -> None:
        super().__init__(root, run_name, model, train_loader, test_loader, optim, scheduler, start_epoch, start_lr, chkpt_freq, description)
        self._loss = BinLoss()
    
    # TODO: implement
    def log_metrics(self, *args) -> None:
        self._logger.
        return super().log_metrics(*args)

    # TODO: implement
    def update(self, loss: torch.Tensor) -> None:
        return super().update(loss)

    # TODO: implement
    def fit(self) -> str:
        return super().fit()