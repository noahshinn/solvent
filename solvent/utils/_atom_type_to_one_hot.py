"""
STATUS: DEV

"""

import torch

from typing import Dict


def atom_type_to_one_hot(atom_type: str, one_hot_key: Dict) -> torch.Tensor:
    """
    Converts atom types to one hot vectors.
    *figure out better way to do this
    *limited to 3 atoms: hydrogen, carbon, and oxygen

    Args:
        atom_type (str): element abbreviation

    Returns:
        (torch.Tensor): one-hot vector

    """
    return torch.Tensor(one_hot_key[atom_type])
