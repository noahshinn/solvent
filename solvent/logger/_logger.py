"""
STATUS: DEV

"""

import torch
import datetime

from solvent import utils


class Logger:
    def __init__(self, file: str) -> None:
        self._file = file
        self._performance_queue = utils.PriorityQueue()
        open(file, 'w').close()

    def _log(self, msg: str) -> None:
        with open(self._file, 'a') as f:
            f.write(msg)

    def log_header(self, description: str) -> None:
        s = f"""
 *---------------------------------------------------*
 |                                                   |
 |               Neural Network Training             |
 |                                                   |
 *---------------------------------------------------*

 Description: {description if description != '' else 'None given'}

"""
        self._log(s)

    def log_resume(self) -> None:
        NotImplemented()

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
ENERGY TRAIN MAE: {e_train_mae.item()} (eV/molecule)
ENERGY TEST MAE: {e_test_mae.item()} (eV/molecule)
FORCE TRAIN MAE: {f_train_mae.item()} (eV/Angstrom)
FORCE TEST MAE: {f_test_mae.item()} (eV/Angstrom)
LEARNING RATE: {lr:.5f}
WALLTIME: {duration:.2f} (s)

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
EPOCH: {p['epoch']}
ENERGY TEST MAE: {p['e_test_mae']}
FORCE TEST MAE: {p['f_test_mae']}

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
