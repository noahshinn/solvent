import torch

from ._loss import LossMixin

from solvent.types import MCPredMetrics


class MCLoss(LossMixin):
    def __init__(self, device: str = 'cuda') -> None:
        self._device = device
        self._loss = torch.nn.CrossEntropyLoss()

        self._c_loss = torch.zeros(1).to(device)
        self._t = 0
        self._n = 0

    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        pred_label = torch.argmax(pred)
        target_label = torch.argmax(target)
        if pred_label == target_label:
            self._t += 1
        self._c_loss += self._loss(pred, target).detach()
        self._n += 1

    def _compute_acc(self) -> torch.Tensor:
        return torch.tensor(self._t / self._n)

    def compute_metrics(self) -> MCPredMetrics:
        acc = self._compute_acc()
        loss = self._c_loss / self._n
        return MCPredMetrics(acc, loss)

    def compute_loss(self) -> torch.Tensor:
        return self._c_loss
    
    def reset(self) -> None:
        self._c_loss = torch.zeros(1).to(self._device)
        self._t = 0
        self._n = 0
