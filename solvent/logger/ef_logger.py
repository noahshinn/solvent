"""
STATUS: NOT TESTED

"""

import torch

from solvent.logger import Logger


class EFLogger(Logger):
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
        super().__init__(log_dir, is_resume)
        self._units = units

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
        self.log(s)

    def format_best_params(self) -> str:
        """
        Formats the log message for up to the top 5 model performances.
        Args:
            None 
        Returns:
            s (str): Message to log.
        """
        s = ''
        n = 5
        while n > 0 and not self._performance_queue.isEmpty():
            p = self._performance_queue.pop()
            s += f"""
Epoch: {p['epoch']}
Energy test mae: {p['e_test_mae']}
Force test mae: {p['f_test_mae']}
"""
            n -= 1
        return s
