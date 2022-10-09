"""
STATUS: DEV

"""

import torch
import datetime

from solvent import utils


class Logger:
    def __init__(self, file: str, is_resume: bool) -> None:
        self._file = file
        self._performance_queue = utils.PriorityQueue()
        if not is_resume:
            open(file, 'w').close()

    def _log(self, msg: str) -> None:
        with open(self._file, 'a') as f:
            f.write(msg)

    def log_header(
            self,
            description: str,
            device: str
        ) -> None:
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
        self._performance_queue.push({
            'epoch': epoch,
            'e_test_mae': e_test_mae,
            'f_test_mae': f_test_mae,
        }, priority=e_test_mae)
        s = f"""EPOCH {epoch}:
Energy train mae: {e_train_mae.item()} (eV/molecule)
Energy test mae: {e_test_mae.item()} (eV/molecule)
Force train mae: {f_train_mae.item()} (eV/Angstrom)
Force test mae: {f_test_mae.item()} (eV/Angstrom)
Learning rate: {lr:.5f}
Wall time: {duration:.2f} (s)

"""
        self._log(s)

    def log_chkpt(self, path: str) -> None:
        self._log(f'Checkpoint saved to {path}\n\n')

    def _format_best_params(self) -> str:
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
