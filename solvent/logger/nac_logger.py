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
        s = f"""EPOCH {epoch}:
Train MAE: {train_mae:.4f}
Test MAE: {test_mae:.4f}
Train MSE: {train_mse:.4f}
Test MSE: {test_mse:.4f}
Learning rate: {lr:.5f}
Wall time: {duration:.2f} (s)

"""
        self.log(s)
