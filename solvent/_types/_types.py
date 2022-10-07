"""
STATUS: DEV

"""

import torch
from typing import NamedTuple


class EnergyForcePrediction(NamedTuple):
    e: torch.Tensor
    f: torch.Tensor

class QMPredMAE(NamedTuple):
    e_mae: torch.Tensor
    f_mae: torch.Tensor
