"""
STATUS: DEV

"""

import os
import torch
import datetime

from solvent.utils import PriorityQueue


class Logger:
    def __init__(
            self,
            log_dir: str,
            is_resume: bool,
            units: str = 'hartrees'
        ) -> None:
        """
        Args:
            log_dir (str): Logging root directory.
            is_resume (bool): Determines if the respective training has started
                from a previous checkpoint.
            units (str): Unit of energy.

        Returns:
            (None)

        """
        assert units.lower() == 'hartree' \
            or units.lower() == 'hartrees' \
            or units.lower() == 'ev' \
            or units.lower() == 'evs' \
            or units.lower() == 'kcal' \
            or units.lower() == 'kcals' \
            or units.lower() == 'kcal/mol' \
            or units.lower() == 'kcals/mol'
        self._units = units
        self._dir = log_dir
        self._file = os.path.join(log_dir, 'out.log')
        self._performance_queue = PriorityQueue()
        if not is_resume:
            if not os.path.exists(self._dir):
                os.makedirs(self._dir)
            open(self._file, 'w').close()

    def _log(self, msg: str) -> None:
        """
        Logs a message to the log file.

        Args:
            msg (str): Message to log.

        Returns:
            (None)

        """
        with open(self._file, 'a') as f:
            f.write(msg)

    def log_header(
            self,
            description: str,
            device: str
        ) -> None:
        """
        Logs the header.

        Args:
            description (str): Description of the training given from the trainer.
            device (str): The device on which the training is occurring.

        Returns:
            (None)

        """
        s = f"""
 *---------------------------------------------------*
 |                                                   |
 |               Neural Network Training             |
 |                                                   |
 *---------------------------------------------------*

 Description: {description if description != '' else 'None given'}
 Device: {device}

"""
        self._log(s)

    def log_resume(self, epoch: int) -> None:
        """
        Logs the start of a training from a checkpoint.

        Args:
            epoch (int): Resume epoch

        Returns:
            (None)

        """
        s = f"""
----------- Resume Training -----------
Epoch: {epoch}

"""
        self._log(s)

    def log_epoch(
            self,
            epoch: int,
            lr: float,
            e_train_mae: torch.Tensor,
            e_test_mae: torch.Tensor,
            f_train_mae: torch.Tensor,
            f_test_mae: torch.Tensor,
            duration: float
        ) -> None:
        """
        Logs a message after an epoch is complete.

        Args:
            epoch (int): Current training epoch.
            lr (float): Current learning rate.
            e_train_mae (torch.Tensor): Train energy MAE.
            e_test_mae (torch.Tensor): Test energy MAE.
            f_train_mae (torch.Tensor): Train force MAE.
            f_test_mae (torch.Tensor): Test force MAE.
            duration (float): Elapsed wall time for the given epoch.

        Returns:
            (None)

        """
        self._performance_queue.push({
            'epoch': epoch,
            'e_test_mae': e_test_mae,
            'f_test_mae': f_test_mae,
        }, priority=e_test_mae)
        s = f"""EPOCH {epoch}:
Energy train mae: {e_train_mae.item()} ({self._units}/molecule)
Energy test mae: {e_test_mae.item()} ({self._units}/molecule)
Force train mae: {f_train_mae.item()} ({self._units}/Angstrom)
Force test mae: {f_test_mae.item()} ({self._units}/Angstrom)
Learning rate: {lr:.5f}
Wall time: {duration:.2f} (s)

"""
        self._log(s)

    def log_chkpt(self, path: str) -> None:
        """
        Logs a message after a set of checkpoint parameters are saved.

        Args:
            path (str): The path to the file.

        Returns:
            (None)

        """
        self._log(f'Checkpoint saved to {path}\n\n')

    def _format_best_params(self) -> str:
        """
        Formats the log message for the top 5 model performances.

        Args:
            None 

        Returns:
            s (str): Message to log.

        """
        s = ''
        n = 5
        while n > 0:
            p = self._performance_queue.pop()
            s += f"""
Epoch: {p['epoch']}
Energy test mae: {p['e_test_mae']}
Force test mae: {p['f_test_mae']}

"""
            n -= 1
        return s

    def log_termination(self, exit_code: str, duration: float) -> None:
        """
        Logs a message after a training session is complete.

        Args:
            exit_code (str): One of 'MAX EPOCH' | 'WALL TIME' | 'LR'
            duration (float): Wall time for the entire training.

        Returns:
            (None)

        """
        t_s = str(datetime.timedelta(seconds=duration)).split(':')
        best_param_s = self._format_best_params()
        s = f"""
Training completed with exit code: {exit_code}
Duration: {t_s[0]} hrs {t_s[1]} min {t_s[2]:.2f} sec

* Best parameter sets *

{best_param_s}

*** HAPPY LANDING ***

"""
        self._log(s)
