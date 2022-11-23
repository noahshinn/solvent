"""
STATUS: NOT TESTED

"""

import torch

from solvent.data import Dataset
from solvent.utils import atom_type_to_one_hot

from typing import Optional, Dict
from torch_geometric.data.data import Data


# TODO: usage
class NACDataset(Dataset):
    def __init__(
            self,
            json_file: str,
            nstructures: int,
            one_hot_key: Dict,
            ncores: Optional[int] = None
        ) -> None:
        super().__init__(json_file, nstructures, one_hot_key, ncores)
        self._species = self._data['species']
        self._coords = self._data['coords']
        self._nacs = self._data['nacs']
        self._natoms = len(self._data['species'][0])

    def load_structure(self, idx: int) -> Data:
        """Load a single structure."""
        one_hot_vecs = torch.stack(
            [atom_type_to_one_hot(
                species=self._species[idx][i],
                one_hot_key=self._one_hot_key)
            for i in range(self._natoms)],
            dim=0)
        return Data(
            x=one_hot_vecs,
            pos=torch.Tensor(self._coords[idx]),
            nacs=torch.Tensor(self._nacs[idx]),
        )
