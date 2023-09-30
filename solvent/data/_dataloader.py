from torch_geometric.loader import DataLoader as DL

from typing import List, Optional, Union
from torch_geometric.data import Data, Dataset
from torch_geometric.loader.data_list_loader import BaseData


class DataLoader(DL):
    def __init__(self, dataset: Union[Dataset, List[Data], List[BaseData]], batch_size: int = 1, shuffle: bool = False, follow_batch: Optional[List[str]] = None, exclude_keys: Optional[List[str]] = None, **kwargs):
        super().__init__(dataset, batch_size, shuffle, follow_batch, exclude_keys, **kwargs) # type: ignore
