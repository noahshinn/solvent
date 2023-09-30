import abc
import json
import multiprocessing
from joblib import Parallel, delayed
from torch_geometric.data.data import Data

from solvent.utils import (
    distribute,
    flatten,
    DataNotLoadedException,
)

from solvent.data import DataLoader

from typing import List, Optional, Dict
from solvent.types import (
    PosIntTuple,
    Loaders
)


class Dataset:
    __metaclass__ = abc.ABCMeta

    def __init__(
            self,
            json_file: str,
            nstructures: int,
            one_hot_key: Dict,
            ncores: Optional[int] = None
        ) -> None:
        assert json_file.endswith('.json')
        assert nstructures >= 2

        with open(json_file) as f:
            self._data = json.load(f)
        self._nstructures = nstructures
        self._one_hot_key = one_hot_key
        self._is_loaded = False
        self._dataset: List[Data] = []
        if not ncores is None:
            self._ncores = ncores
        else:
            self._ncores = multiprocessing.cpu_count()

    # TODO: type error in flatten call
    def load(self) -> None:
        """Loads the dataset into a list of Data."""
        distr_config = distribute(self._nstructures, self._ncores)
        res = Parallel(n_jobs=self._ncores)(delayed(self._load_collection)(task_idx_range) for task_idx_range in distr_config)
        self._dataset = flatten(res) # type: ignore
        self._is_loaded = True

    def get_dataset(self) -> List[Data]:
        if not self._is_loaded:
            raise DataNotLoadedException('must load dataset before accessing content.')
        return self._dataset

    def add_structures(self, structures: List[Data]) -> None:
        """Extend this loaded dataset."""
        if not self._is_loaded:
            raise DataNotLoadedException('must load dataset before adding to dataset.')
        self._dataset.extend(structures)

    def gen_dataloaders(
            self,
            split: float = 0.8,
            batch_size: int = 1,
            should_shuffle: bool = True
        ) -> Loaders:
        """Generates train and test DataLoaders for training."""
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
        return Loaders(train_loader, test_loader)

    def __len__(self) -> int:
        """Length of loaded items in dataset."""
        if not self._is_loaded:
            raise DataNotLoadedException('must load dataset before accessing dataset length.')
        return len(self._dataset)

    def __getitem__(self, idx: int) -> Data:
        """Gets item from index from loaded dataset."""
        if not self._is_loaded:
            raise DataNotLoadedException('must load dataset before accessing content.')
        return self._dataset[idx]

    def _load_collection(self, task_idx_range: PosIntTuple) -> List[Data]:
        """Load a portion of dataset."""
        c = [self.load_structure(i) for i in range(task_idx_range[0], task_idx_range[1])]
        return c 

    @abc.abstractmethod
    def load_structure(self, idx: int) -> Data:
        """Load a single structure."""
        return
