"""
STATUS: DEV

"""

import torch

_ONE_HOT = {
    'H': [1., 0., 0.],
    'C': [0., 1., 0.],
    'O': [0., 0., 1.]
}

def atom_type_to_one_hot(atom_type: str) -> torch.Tensor:
    """
    Converts atom types to one hot vectors.
    *figure out better way to do this
    *limited to 3 atoms: hydrogen, carbon, and oxygen

    Args:
        atom_type (str): element abbreviation

    Returns:
        (torch.Tensor): one-hot vector

    """
    return torch.Tensor(_ONE_HOT[atom_type.upper()])
