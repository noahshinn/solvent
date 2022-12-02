"""
STATUS: NOT TESTED

"""

import torch
import warnings

from typing import Union, Optional, Dict
from torch_geometric.data.data import Data


class DeployNAC:
    def __init__(
            self,
            nac_model: Optional[torch.nn.Module] = None,
            mc_model: Optional[torch.nn.Module] = None,
            nac_model_map: Optional[Dict[int, torch.nn.Module]] = None,
            shift: Union[float, torch.Tensor] = 0.0,
            scale: Union[float, torch.Tensor] = 1.0,
        ) -> None:
        self._shift = shift
        self._scale = scale
        if nac_model is None:
            assert not mc_model is None or not nac_model_map is None
            self._should_classify = True
            self._mc_model = mc_model
            self._nac_model_map = nac_model_map
        else:
            if not mc_model is None:
                warnings.warn(f'classification model of type `{type(mc_model)}` is given, but will not be used')
            if not nac_model_map is None:
                warnings.warn(f'nac inference model map with keys `{nac_model_map.keys()}` is given, but will not be used')

            self._should_classify = True
            self._nac_model = nac_model 

    def __call__(self, structure: Union[dict, Data]) -> torch.Tensor:
        """Computes the respective derivative coupling vector"""
        assert isinstance(structure, dict) or isinstance(structure, Data)

        if self._should_classify:
            assert not self._mc_model is None and not self._nac_model_map is None
            bin_ = self._mc_model(structure)
            assert bin_ in self._nac_model_map.keys()
            out = self._nac_model_map[bin_](structure)
        else:
            assert not self._nac_model is None
            out = self._nac_model(structure)
        
        return out * self._scale + self._shift
