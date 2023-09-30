import torch

from ._loss import LossMixin
from solvent.types import BinPredMetrics


# FIXME: acc, prec, rec
class BinLoss(LossMixin):
    def __init__(self, device: str = 'cuda') -> None:
        self._loss = torch.nn.BCELoss()
        self._device = device
        self._c_loss = torch.zeros(1).to(device)
        self._n = 0
        self._tp = 0
        self._tn = 0
        self._fp = 0
        self._fn = 0

    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        self._c_loss += self._loss(pred, target).detach()
        if pred == 1:
            if target == 1:
                self._tp += 1
            else:
                self._fp += 1
        else:
            if target == 1:
                self._fn += 1
            else:
                self._tn += 1
        self._n += 1

    def _compute_acc(self) -> torch.Tensor:
        return torch.tensor((self._tp + self._tn) / self._n)

    def _compute_prec(self) -> torch.Tensor:
        return torch.tensor(self._tp / (self._tp + self._fp))

    def _compute_rec(self) -> torch.Tensor:
        return torch.tensor(self._tp / (self._tp + self._fn))

    def _compute_f1(self, prec: torch.Tensor, rec: torch.Tensor) -> torch.Tensor:
        return 2 * prec * rec / (prec + rec)

    def compute_metrics(self) -> BinPredMetrics:
        acc = self._compute_acc()
        prec = self._compute_prec()
        rec = self._compute_rec()
        f1 = self._compute_f1(prec, rec)
        return BinPredMetrics(acc, prec, rec, f1)

    def compute_loss(self) -> torch.Tensor:
        return self._c_loss

    def reset(self) -> None:
        self._c_loss = torch.zeros(1).to(self._device)
        self._tp = 0
        self._fp = 0
        self._fn = 0
        self._tn = 0
        self._n = 0
