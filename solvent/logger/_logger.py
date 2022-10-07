"""
STATUS: DEV

"""

import torch
import datetime


class Logger:
    def __init__(self, file: str) -> None:
        self._file = file
        open(file, 'w').close()

    def _log(self, msg: str) -> None:
        with open(self._file, 'a') as f:
            f.write(msg)

    def log_header(self) -> None:
        NotImplemented()

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

    def log_termination(self, exit_code: str, duration: float) -> None:
        t_s = str(datetime.timedelta(seconds=duration)).split(':')
        s = f"""
*** HAPPY LANDING ***
Training completed with exit code: {exit_code}
Duration: {t_s[0]} hrs {t_s[1]} min {t_s[2]:.2f} sec

"""
        self._log(s)
