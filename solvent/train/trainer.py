"""
STATUS: DEV

"""

import os
import time
import torch

from solvent import (
    nn,
    logger,
    types,
    constants
)

from typing import Dict, Optional, Union
from torch_geometric.loader import DataLoader
from torch_geometric.data.data import Data


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

    Example shown in `solvent/demo/run.py`

    """
    def __init__(
            self,
            model: torch.nn.Module,
            train_loader: DataLoader,
            test_loader: DataLoader,
            optim_params: Optional[Dict] = None,
            scheduler_params: Optional[Dict] = None,
            energy_contribution: float = 1.0,
            force_contribution: float = 1.0,
            start_epoch: int = 0,
            start_lr: float = 1e-2,
            log_file: str = 'train.log',
            nn_save_dir: str = 'nn',
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
        self._train_loader = train_loader
        self._test_loader = test_loader 
        self._epoch = start_epoch
        self._nn_save_dir = nn_save_dir
        self._chkpt_freq = chkpt_freq
        self._cur_chkpt_count = 0
        self._exit_code = 'NOT TERMINATED'

        self._optim = torch.optim.Adam(model.parameters(), lr=start_lr)
        if not optim_params is None:
            self._optim.load_state_dict(optim_params)
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
        if not scheduler_params is None:
            self._scheduler.load_state_dict(scheduler_params)
        self._lr = self._optim.param_groups[0]['lr']

        if not optim_params is None and not scheduler_params is None:
            self._is_resume = True
        else:
            self._is_resume = False
        
        self._loss = nn.EnergyForceLoss(
            energy_contribution=energy_contribution,
            force_contribution=force_contribution
        )
        self._logger = logger.Logger(file=log_file, is_resume=self._is_resume)
        self._description = description
        self._walltime = self._srt_time = time.perf_counter()

    def _pred(self, structure: Union[Dict, Data]) -> types.EnergyForcePrediction:
        e = self._model(structure)
        f = nn.force_grad(e, structure['pos'])
        return types.EnergyForcePrediction(e, f)

    def _evaluate(self, loader: DataLoader, mode: str) -> types.QMPredMAE:
        assert mode == 'TRAIN' or mode == 'TEST'
        for structure in loader:
            structure['pos'].requires_grad = True
            structure.to(self._device)
            e = self._model(structure)
            f = nn.force_grad(energies=e, pos=structure['pos'])
            self._loss(
                e_pred=e,
                e_target=structure['energies'],
                f_pred=f,
                f_target=structure['forces']
            )
            if mode == 'TRAIN':
                self._step(loss=self._loss.compute_loss())

        e_mae, f_mae = self._loss.compute_metrics()

        return types.QMPredMAE(e_mae, f_mae)

    def _log_metrics(
            self,
            e_train_mae: torch.Tensor,
            e_test_mae: torch.Tensor,
            f_train_mae: torch.Tensor,
            f_test_mae: torch.Tensor,
        ) -> None:
        self._logger.log_epoch(
            epoch=self._epoch,
            lr=self._lr,
            e_train_mae=e_train_mae,
            e_test_mae=e_test_mae,
            f_train_mae=f_train_mae,
            f_test_mae=f_test_mae,
            duration=time.perf_counter() - self._walltime
        )

    def _step(self, loss: torch.Tensor) -> None:
        loss.backward()
        self._optim.step()
        self._optim.zero_grad()
        
    def _update(self, loss: torch.Tensor) -> None:
        self._walltime = time.perf_counter()
        self._scheduler.step(metrics=loss)
        self._loss.reset()
        self._epoch += 1
        self._cur_chkpt_count += 1
        self._lr = self._optim.param_groups[0]['lr']

    def _chkpt(self) -> None:
        save_path = os.path.join(self._nn_save_dir, f'{str(self._epoch)}.pt')
        chkpt = {
            'epoch': self._epoch,
            'model': self._model.state_dict(),
            'optim': self._optim.state_dict(),
            'scheduler': self._scheduler.state_dict()
        }
        torch.save(chkpt, save_path)
        self._logger.log_chkpt(path=save_path)
        self._cur_chkpt_count = 0

    def _should_terminate(self) -> bool:
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

    def _fit(self) -> str:
        while not self._should_terminate():
            e_train_mae, f_train_mae = self._evaluate(loader=self._train_loader, mode='TRAIN')
            e_test_mae, f_test_mae = self._evaluate(loader=self._test_loader, mode='TEST')

            self._log_metrics(
                e_train_mae=e_train_mae,
                e_test_mae=e_test_mae,
                f_train_mae=f_train_mae,
                f_test_mae=f_test_mae,
            )

            self._update(self._loss.compute_loss())

            if self._cur_chkpt_count == self._chkpt_freq:
                self._chkpt()

        return self._exit_code

    def fit(self) -> None:
        if not self._is_resume:
            self._logger.log_header(
                description=self._description,
                device=self._device
            )
        else:
            self._logger.log_resume(self._epoch)
        res = self._fit()
        self._logger.log_termination(
            exit_code=res,
            duration=time.perf_counter() - self._srt_time
        )
