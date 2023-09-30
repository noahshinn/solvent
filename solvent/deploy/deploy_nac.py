import torch
import warnings

from solvent.utils import from_mc
from solvent.types import NACPred

from typing import Union, Optional, Dict
from torch_geometric.data.data import Data


class DeployNAC:
    def __init__(
            self,
            nac_model: Optional[torch.nn.Module] = None,
            mc_model: Optional[torch.nn.Module] = None,
            nac_model_map: Optional[Dict[int, Dict[str, Union[torch.nn.Module, float, torch.Tensor]]]] = None,
            scale: Optional[Union[float, torch.Tensor]] = None,
            shift: Optional[Union[float, torch.Tensor]] = None,
            nbins: Optional[int] = None
        ) -> None:
        if nac_model is None:
            assert not mc_model is None or not nac_model_map is None
            self._should_classify = True
            self._mc_model = mc_model
            self._nac_model_map = nac_model_map
            self._nbins = nbins
        else:
            if not mc_model is None:
                warnings.warn(f'classification model of type `{type(mc_model)}` is given, but will not be used')
            if not nac_model_map is None:
                warnings.warn(f'nac inference model map with keys `{nac_model_map.keys()}` is given, but will not be used')

            self._should_classify = True
            self._nac_model = nac_model 
            self._scale = scale
            self._shift = shift 

    def __call__(self, structure: Union[dict, Data]) -> Union[torch.Tensor, NACPred]:
        """Computes the respective derivative coupling vector"""
        assert isinstance(structure, dict) or isinstance(structure, Data)

        if self._should_classify:
            assert not self._mc_model is None and not self._nac_model_map is None
            bin_ = from_mc(self._mc_model(structure))

            """ TEMP """

            return NACPred(bin_, torch.tensor(0))

            """ TEMP """

            model = self._nac_model_map[bin_]['model']
            scale = self._nac_model_map[bin_]['scale']
            shift = self._nac_model_map[bin_]['shift']
            assert bin_ in self._nac_model_map.keys()
            assert isinstance(model, torch.nn.Module)
            assert isinstance(scale, float) or isinstance(scale, torch.Tensor)
            assert isinstance(shift, float) or isinstance(shift, torch.Tensor)
            return NACPred(bin_, torch.detach(model(structure) * scale + shift))
        else:
            assert not self._nac_model is None
            assert not self._scale is None
            assert not self._shift is None
            return torch.detach(self._nac_model(structure) * self._scale + self._shift)
