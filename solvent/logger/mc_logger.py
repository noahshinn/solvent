import torch

from solvent.logger import Logger


class MCLogger(Logger):
    def __init__(self, log_dir: str, is_resume: bool, verbose: bool = True) -> None:
        super().__init__(log_dir, is_resume, verbose)

    def log_epoch(
            self,
            epoch: int,
            lr: float,
            accuracy_train: torch.Tensor,
            accuracy_test: torch.Tensor,
            loss_train: torch.Tensor,
            loss_test: torch.Tensor,
            duration: float
        ) -> None:
        """Logs a message after an epoch is complete."""
        self._performance_queue.push({
            'epoch': epoch,
            'mc_test_acc': accuracy_test,
            'mc_test_loss': loss_test,
        }, priority=accuracy_test)
        s = f"""EPOCH {epoch}:
Train accuracy: {accuracy_train.item():.4f}
Test accuracy: {accuracy_test.item():.4f}
Train loss: {loss_train.item():.4f}
Test loss: {loss_test.item():.4f}
Learning rate: {lr:.5f}
Wall time: {duration:.2f} (s)

"""
        self.log(s)

        if self._verbose:
            self.verbose_logger(epoch, f'Accuracy: {accuracy_test.item():.4f}, Loss: {loss_test.item():.4f}')

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
MC test acc: {p['mc_test_acc'].item():.4f}
MC test loss: {p['mc_test_loss'].item():.4f}
"""
            n -= 1
        return s
