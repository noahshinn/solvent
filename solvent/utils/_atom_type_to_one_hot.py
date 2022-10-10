"""
STATUS: NOT TESTED

"""

import torch

from typing import Dict, List


def atom_type_to_one_hot(
        species: str,
        one_hot_key: Dict[str, List[float]]
    ) -> torch.Tensor:
    """
    Converts chemical species strings to one hot vectors.
    
    Sample key for H2O: {'H': [1., 0.], 'O': [0., 1.]}

    Args:
        species (str): Element abbreviation.
        one_hot_key (dict(str, list(float))): Key to encode chemical species
            to one-hot vectors.

    Returns:
        (torch.Tensor): one-hot vector

    """
    return torch.Tensor(one_hot_key[species])
