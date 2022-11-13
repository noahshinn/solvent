"""
STATUS: DEV

"""

import os
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
from solvent.nn import EnergyForceLoss, force_grad
from solvent.utils import InvalidFileType, set_exit_handler
from solvent.logger import Logger
from solvent.types import EnergyForcePrediction, QMPredMAE

from typing import Dict, Union
from torch_geometric.loader import DataLoader
from torch_geometric.data.data import Data


class Trainer:
    """
    Start training:

    >>> from solvent import train, models
    >>> model = models.EModel(*args, **kwargs)
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
    def __init__(
            self,
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
            log_dir: str = 'train-log',
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
        self._log_dir = log_dir 
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

        self._loss = EnergyForceLoss(
            energy_contribution=energy_contribution,
            force_contribution=force_contribution,
            device=self._device
        )
        self._e_scale = energy_scale
        self._f_scale = force_scale 
        self._nmol = nmol
        self._logger = Logger(
            log_dir=log_dir,
            is_resume=self._is_resume,
            units=units
        )
        self._description = description
        self._walltime = self._srt_time = time.perf_counter()

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

    def _evaluate(self, loader: DataLoader, mode: str) -> QMPredMAE:
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
        
    def update(self, loss: torch.Tensor) -> None:
        """
        Updates after every epoch.
            - resets wall time
        

        Args:
            loss (torch.Tensor): Loss tensor from forward pass

        Returns:
            (None): TODO

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

    def chkpt(self) -> None:
        """
        TODO

        Args:
            None

        Returns:
            TODO (None): TODO

        """
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
        TODO

        Args:
            None

        Returns:
            TODO (bool): TODO

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

    def fit(self) -> str:
        """
        TODO

        Args:
            None

        Returns:
            (str): exit code

        """
        while not self.should_terminate():
            e_train_mae, f_train_mae = self._evaluate(loader=self._train_loader, mode='TRAIN')
            e_test_mae, f_test_mae = self._evaluate(loader=self._test_loader, mode='TEST')

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

    def set_loaders(self, train_loader: DataLoader, test_loader: DataLoader) -> None:
        self._train_loader = train_loader
        self._test_loader = test_loader

    def train(self) -> None:
        """
        TODO

        Args:
            None

        Returns:
            (None)

        """
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
