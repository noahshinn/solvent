import torch
from torch_geometric.loader import DataLoader
from torch_geometric.data.data import Data

from solvent.data import NACDataset
from solvent.utils import atom_type_to_one_hot

from typing import Dict, Optional


class NACDeployedDataset(NACDataset):
    def __init__(
            self,
            json_file: str,
            nstructures: int,
            one_hot_key: Dict,
            ncores: Optional[int] = None
        ) -> None:
        super().__init__(json_file, nstructures, one_hot_key, ncores)
        self._bin = self._data['bin']

    def gen_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self._dataset, # type: ignore
            batch_size=1,
            shuffle=False
        )

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
            bin=torch.tensor(self._bin[idx]),
            nacs=torch.Tensor(self._nacs[idx]),
        )
