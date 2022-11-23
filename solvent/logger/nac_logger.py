import torch

from solvent.logger import Logger


class NACLogger(Logger):
    def __init__(self, log_dir: str, is_resume: bool) -> None:
        super().__init__(log_dir, is_resume)

    # TODO: implement performance queue update
    def log_epoch(
            self,
            epoch: int,
            lr: float,
            train_mae: torch.Tensor,
            test_mae: torch.Tensor,
            train_mse: torch.Tensor,
            test_mse: torch.Tensor,
            duration: float
        ) -> None:
        """Logs a message after an epoch is complete."""
        self._performance_queue.push({
            'epoch': epoch,
            'nac_test_mae': test_mae,
            'nac_test_mse': test_mse,
        }, priority=test_mse)
        s = f"""EPOCH {epoch}:
Train MAE: {train_mae.item():.4f}
Test MAE: {test_mae.item():.4f}
Train MSE: {train_mse.item():.4f}
Test MSE: {test_mse.item():.4f}
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
NAC test mae: {p['nac_test_mae']}
NAC test mse: {p['nac_test_mse']}
"""
            n -= 1
        return s
