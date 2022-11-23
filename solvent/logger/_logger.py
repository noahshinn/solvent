"""
STATUS: NOT TESTED

"""

import os
import abc
import datetime

from solvent.utils import PriorityQueue


class Logger:
    __metaclass__ = abc.ABCMeta

    def __init__(
            self,
            log_dir: str,
            is_resume: bool,
        ) -> None:
        """
        Args:
            log_dir (str): Logging root directory.
            is_resume (bool): Determines if the respective training has started
                from a previous checkpoint.

        Returns:
            (None)

        """
        self._dir = log_dir
        self._file = os.path.join(log_dir, 'out.log')
        self._performance_queue = PriorityQueue()
        if not is_resume:
            if not os.path.exists(self._dir):
                os.makedirs(self._dir)
            open(self._file, 'w').close()

    def log(self, msg: str) -> None:
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
        self.log(s)

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
        self.log(s)

    def log_chkpt(self, path: str) -> None:
        """
        Logs a message after a set of checkpoint parameters are saved.

        Args:
            path (str): The path to the file.

        Returns:
            (None)

        """
        self.log(f'Checkpoint saved to {path}\n\n')

    @abc.abstractmethod
    def format_best_params(self) -> None:
        """Formats the top 5 results for logging."""
        return

    def log_premature_termination(self) -> None:
        """
        Logs a message after a premature termination.
        *activated with CTRL+C

        Args:
            None

        Returns:
            (None)

        """
        best_param_s = self.format_best_params()
        s = f"""
Training completed with exit code: PREMATURE EXIT

* Best parameter sets *

{best_param_s}

*** TERMINATED ***

"""
        self.log(s)

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
        best_param_s = self.format_best_params()
        s = f"""
Training completed with exit code: {exit_code}
Duration: {t_s[0]} hrs {t_s[1]} min {t_s[2]:.2f} sec

* Best parameter sets *

{best_param_s}

*** HAPPY LANDING ***

"""
        self.log(s)

    @abc.abstractmethod
    def log_epoch(self, *args, **kwargs) -> None:
        """Logs a message after an epoch is complete."""
        return
