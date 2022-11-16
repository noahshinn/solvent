"""
STATUS: NOT TESTED

"""

import os
import abc
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

from solvent import constants
from solvent.utils import InvalidFileType, set_exit_handler
from solvent.logger import Logger

from typing import Union
from torch_geometric.loader import DataLoader


class Trainer:
    """
    Start training:

    >>> from solvent import train, models
    >>> model = models.Model(*args, **kwargs)
    >>> train_loader = ...
    >>> test_loader = ...
    >>> trainer = train.Trainer(
    ...     model=model,
    ...     train_loader=train_loader,
    ...     test_loader=test_loader,
    ...     *args, **kwargs
    ... )
    >>> trainer.fit()

    Resume training:

    >>> import torch
    >>> from solvent import train, models
    >>> params = torch.load('params.pt')
    >>> model = models.Model(*args, **kwargs)
    >>> model.load_state_dict(params['model'])
    >>> train_loader = ...
    >>> test_loader = ...
    >>> trainer = train.Trainer(
    ...     model=model,
    ...     train_loader=train_loader,
    ...     test_loader=test_loader,
    ...     optim_params=params['optim'],
    ...     scheduler_params=params['scheduler'],
    ...     start_epoch=params['epoch'],
    ...     log_file='resume.log'
    ...     *args, **kwargs)
    >>> trainer.fit()

    Example shown in `/solvent/demo/example_training.py`

    """

    __metaclass__ = abc.ABCMeta

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
            start_lr: float = 1e-2,
            chkpt_freq: int = 1,
            description: str = ''
        ) -> None:
        torch.set_default_dtype(torch.float32)
        if torch.cuda.is_available():
            self._device = 'cuda:0'
        else:
            self._device = 'cpu'

        self._model = model
        self._model.to(self._device)

        if isinstance(train_loader, str):
            if not train_loader.endswith('.pt'):
                raise InvalidFileType('not given .pt file!')
            self._train_loader = torch.load(train_loader)
        else:
            self._train_loader = train_loader
        if isinstance(test_loader, str):
            if not test_loader.endswith('.pt'):
                raise InvalidFileType('not given .pt file!')
            self._train_loader = torch.load(train_loader)
        else:
            self._test_loader = test_loader 

        self._epoch = start_epoch
        self._log_dir = os.path.join(root, run_name)
        self._chkpt_freq = chkpt_freq
        self._cur_chkpt_count = 0
        self._exit_code = 'NOT TERMINATED'

        if not optim is None:
            self._optim = optim
        else:
            self._optim = torch.optim.Adam(model.parameters(), lr=start_lr)

        if not scheduler is None:
            self._scheduler = scheduler
        else:
            # TODO: move to init args
            self._scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer=self._optim,
                mode='min',
                factor=0.8,
                patience=50,
                threshold=1e-4,
                threshold_mode='rel',
                cooldown=0,
                min_lr=0.0,
                eps=1e-8,
                verbose=False
            )

        self._is_resume = not optim is None and not scheduler is None
        self._lr = self._optim.param_groups[0]['lr']

        self._description = description
        self._walltime = self._srt_time = time.perf_counter()

    def step(self, loss: torch.Tensor) -> None:
        """
        Propagates the training by one step.
        * only called during mode='TRAIN'

        Args:
            loss (torch.Tensor): Loss tensor from forward pass.

        Returns:
            (None)

        """
        loss.backward()
        self._optim.step()
        self._optim.zero_grad()

    def chkpt(self) -> None:
        """Logs a checkpoint to the logging directory."""
        save_path = os.path.join(self._log_dir, f'{str(self._epoch)}.pt')
        chkpt = {
            'model': self._model.state_dict(),
            'optim': self._optim.state_dict(),
            'scheduler': self._scheduler.state_dict(),
            'epoch': self._epoch
        }
        torch.save(chkpt, save_path)
        self._logger.log_chkpt(path=save_path)
        self._cur_chkpt_count = 0

    def should_terminate(self) -> bool:
        """
        Determines if the current training should be terminated.

        Args:
            None

        Returns:
            (bool)

        """
        if self._epoch >= 10000:
            self._exit_code = 'MAX EPOCH'
            return True
        elif time.perf_counter() - self._srt_time > constants.SEVEN_DAYS:
            self._exit_code = 'WALL TIME'
            return True
        elif self._lr < 1e-6:
            self._exit_code = 'LR'
            return True
        return False

    def set_loaders(self, train_loader: DataLoader, test_loader: DataLoader) -> None:
        """Set loaders"""
        self._train_loader = train_loader
        self._test_loader = test_loader

    def train(self) -> None:
        """Run training"""
        set_exit_handler(self._logger.log_premature_termination)
        if not self._is_resume:
            self._logger.log_header(
                description=self._description,
                device=self._device
            )
        else:
            self._logger.log_resume(self._epoch)
        res = self.fit()
        self._logger.log_termination(
            exit_code=res,
            duration=time.perf_counter() - self._srt_time
        )

    @abc.abstractmethod
    def log_metrics(
            self,
            *args
        ) -> None:
        """Abstract method"""
        return
    
    @abc.abstractmethod
    def update(self, loss: torch.Tensor) -> None:
        """Update after every epoch."""

    @abc.abstractmethod
    def fit(self) -> str:
        """Abstract method"""
        return
