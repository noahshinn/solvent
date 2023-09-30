from torch_geometric.data.data import Data

from solvent.data import Dataset

from typing import Optional, Dict


# TODO: usage
class BinDataset(Dataset):
    def __init__(
            self,
            json_file: str,
            nstructures: int,
            one_hot_key: Dict,
            units: str = 'HARTREE',
            ncores: Optional[int] = None
        ) -> None:
        super().__init__(json_file, nstructures, one_hot_key, units, ncores)
        self._species = self._data['species']
        self._coords = self._data['coords']
        self._e_diffs = self._data['e_diff']
        self._nacs = self._data['nacs']
        self._norms = self._data['norms']
        self._natoms = len(self._data['species'])

    # FIXME: implement
    def load_structure(self, idx: int) -> Data:
        """Load a single structure."""
        NotImplemented()
        return Data()
