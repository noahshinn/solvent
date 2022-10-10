"""
STATUS: DEV

"""

import json
import torch
import multiprocessing
from joblib import Parallel, delayed
from torch_geometric.data.data import Data

from solvent.utils import (
    distribute,
    flatten,
    DataNotLoadedException,
    atom_type_to_one_hot,
    center_coords,
    hartree_to_ev,
    hartree_to_kcal
)

from solvent.data import DataLoader

from typing import List, Optional, Dict
from solvent import types


class EnergyForceDataset:
    """
    Usage:

    Load an EnergyForceDataset with 100 structures:
    >>> from solvent import data
    >>> ds = data.EnergyForceDataset('file.json', units='kcal')
    ...     json_file='file.json',
    ...     nstructures=100,
    ...     one_hot_key={
    ...         'H': [1., 0., 0.],
    ...         'C': [0., 1., 0.],
    ...         'O': [0., 0., 1.]
    ...     }
    ...     units='kcal'
    ... )
    >>> ds.load()

    Get averages and scale to target values:
    >>> mean_energy = ds.get_energy_mean()
    >>> rms_force = ds.get_force_rms()
    >>> ds.to_target_energy(shift_factor=mean_energy, scale_factor = 1 / rms_force)
    >>> ds.to_target_force(scale_factor = 1 / rms_force)

    Build a DataLoader with 100 structures and a batch size of 1:
    >>> loader = data.DataLoader(ds.get_dataset(), batch_size=1, shuffle=True)

    Generate train and test DataLoaders with 100 total structures,
    a train/test split of 90/10, and a batch size of 1:
    >>> train_loader, test_loader = ds.gen_dataloaders(
    ...     split=0.9,
    ...     batch_size=1,
    ...     should_shuffle=True
    ... )

    """
    def __init__(
            self,
            json_file: str,
            nstructures: int,
            one_hot_key: Dict,
            units: str = 'HARTREE',
            ncores: Optional[int] = None
        ) -> None:
        assert json_file.endswith('.json')
        assert nstructures >= 2
        assert units.upper() == 'HARTREE' \
            or units.upper() == 'EV' \
            or units.upper() == 'KCAL'

        with open(json_file) as f:
            self._data = json.load(f)
        self._xyz = self._data['xyz']
        self._energies = self._data['energy']
        self._forces = self._data['grad']
        self._natoms = len(self._xyz[0])
        self._nstates = len(self._energies[0])
        self._nstructures = nstructures
        self._one_hot_key = one_hot_key
        self._is_loaded = False
        self._dataset: List[Data] = []
        self._units = units.upper()
        if not ncores is None:
            self._ncores = ncores
        else:
            self._ncores = multiprocessing.cpu_count()

    def load(self) -> None:
        distr_config = distribute(self._nstructures, self._ncores)
        res = Parallel(n_jobs=self._ncores)(delayed(self._load_collection)(task_idx_range) for task_idx_range in distr_config)
        self._dataset = flatten(res)
        self._is_loaded = True

    def get_dataset(self) -> List[Data]:
        if not self._is_loaded:
            raise DataNotLoadedException('must load dataset before accessing content.')
        return self._dataset

    def get_energy_mean(self) -> float:
        if not self._is_loaded:
            raise DataNotLoadedException('must load dataset before accessing content.')
        c = torch.zeros(1)
        for i in range(self._nstructures):
            c += self._dataset[i].energies.mean()
        return (c / self._nstructures).item()

    def get_force_rms(self) -> float:
        if not self._is_loaded:
            raise DataNotLoadedException('must load dataset before accessing content.')
        c = torch.zeros(1)
        for i in range(self._nstructures):
            c += self._dataset[i].forces.pow(2).sum()
        return (c / (self._nstructures * self._nstates * self._natoms * 3)).sqrt().item()

    def to_target_energy(self, shift_factor: float, scale_factor: float) -> None:
        if not self._is_loaded:
            raise DataNotLoadedException('must load dataset before accessing content.')
        for i in range(self._nstructures):
            self._dataset[i].energies = (self._dataset[i].energies - shift_factor) * scale_factor

    def to_target_force(self, scale_factor: float) -> None:
        if not self._is_loaded:
            raise DataNotLoadedException('must load dataset before accessing content.')
        for i in range(self._nstructures):
            self._dataset[i].forces *= scale_factor

    def gen_dataloaders(
            self,
            split: float = 0.8,
            batch_size: int = 1,
            should_shuffle: bool = True
        ) -> types.Loaders:
        ntrain = round(self._nstructures * split)
        ntest = self._nstructures - ntrain
        train_loader = DataLoader(
            dataset=self._dataset[:ntrain],
            batch_size=batch_size,
            shuffle=should_shuffle
        )
        test_loader = DataLoader(
            dataset=self._dataset[-ntest:],
            batch_size=batch_size,
            shuffle=should_shuffle
        )
        return types.Loaders(train_loader, test_loader)

    def __len__(self) -> int:
        if not self._is_loaded:
            raise DataNotLoadedException('must load dataset before accessing dataset length.')
        return len(self._dataset)

    def __getitem__(self, idx: int) -> Data:
        if not self._is_loaded:
            raise DataNotLoadedException('must load dataset before accessing content.')
        return self._dataset[idx]

    def _load_collection(self, task_idx_range: types.PosIntTuple) -> List[Data]:
        c = [self._load_structure(i) for i in range(task_idx_range[0], task_idx_range[1])]
        return c 

    def _load_structure(self, idx: int) -> Data:
        one_hot_vecs = torch.stack(
            [atom_type_to_one_hot(
                atom_type=self._xyz[idx][i][0],
                one_hot_key=self._one_hot_key)
            for i in range(self._natoms)],
            dim=0)
        coords = torch.Tensor([atom[1:] for atom in self._xyz[idx]])
        shifted_coords = center_coords(coords)
        energies = torch.Tensor(self._energies[idx])
        forces = torch.Tensor(self._forces[idx])
        if self._units == 'EV':
            energies = hartree_to_ev(energies)
            forces = hartree_to_ev(forces)
        elif self._units == 'KCAL':
            energies = hartree_to_kcal(energies)
            forces = hartree_to_kcal(forces)
        return Data(
            x=one_hot_vecs,
            pos=shifted_coords,
            energies=energies,
            forces=forces
        )
