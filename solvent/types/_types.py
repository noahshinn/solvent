"""
STATUS: DEV

"""

import torch
from typing import NamedTuple, NewType

PosInt = NewType('PosInt', int)


class EnergyForcePrediction(NamedTuple):
    e: torch.Tensor
    f: torch.Tensor

class QMPredMAE(NamedTuple):
    e_mae: torch.Tensor
    f_mae: torch.Tensor

class PosIntTuple(NamedTuple):
    num1: PosInt
    num2: PosInt
