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
            accuracy_train: torch.Tensor,
            accuracy_test: torch.Tensor,
            precision_train: torch.Tensor,
            precision_test: torch.Tensor,
            recall_train: torch.Tensor,
            recall_test: torch.Tensor,
            f1_train: torch.Tensor,
            f1_test: torch.Tensor,
            duration: float
        ) -> None:
        """Logs a message after an epoch is complete."""
        s = f"""EPOCH {epoch}:
Train accuracy: {accuracy_train:.4f}
Test accuracy: {accuracy_test:.4f}
Train precision: {precision_train:.4f}
Test precision: {precision_test:.4f}
Train recall: {recall_train:.4f}
Test recall: {recall_test:.4f}
Train f1 score: {f1_train:.4f}
Test f1 score: {f1_test:.4f}
Learning rate: {lr:.5f}
Wall time: {duration:.2f} (s)

"""
        self.log(s)
