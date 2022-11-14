import torch

from solvent.logger import Logger


class BinLogger(Logger):
    def __init__(self, log_dir: str, is_resume: bool) -> None:
        super().__init__(log_dir, is_resume)

    # TODO: implement performance queue update
    def log_epoch(
            self,
            epoch: int,
            lr: float,
            accuracy: torch.Tensor,
            precision: torch.Tensor,
            recall: torch.Tensor,
            duration: float
        ) -> None:
        """Logs a message after an epoch is complete."""
        s = f"""EPOCH {epoch}:
Accuracy: {accuracy:.4f}
Precision: {precision:.4f}
Recall: {recall:.4f}
Learning rate: {lr:.5f}
Wall time: {duration:.2f} (s)

"""
        self.log(s)
