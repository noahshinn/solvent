"""
STATUS: DEV

"""

import torch
from e3nn import o3
from torch_geometric.data import Data
from e3nn.util.jit import compile_mode

from solvent.models import Model

from typing import Union, List, Optional, Dict


@compile_mode("script")
class NACModel(torch.nn.Module):
    """
    An equivariant graph neural network for derivative coupling vector inference.

    """
    def __init__(
            self,
            irreps_in: Union[o3.Irreps, str, None],
            hidden_sizes: List[int],
            natoms: int,
            nlayers: int,
            max_radius: float,
            nbasis_funcs: int,
            nradial_layers: int,
            nradial_neurons: int,
            navg_neighbors: float,
            act: Optional[Dict[int, torch.nn.Module]] = None,
            cache: Optional[str] = None
        ) -> None:
        r"""
        Initializes the network.

        Args:
            irreps_in (e3nn.o3.Irreps | str | None): representation of the input features
                can be set to ``None`` if nodes don't have input features
            hidden_sizes (list(int)): hidden feature sizes
            natoms (int): number of atoms in the system
            nlayers (int): number of gates (non linearities)
            max_radius (float): maximum radius for the convolution
            nbasis_funcs (int): number of basis on which the edge length are
                projected
            nradial_layers (int): number of hidden layers in the radial fully connected
                network
            nradial_neurons (int): number of neurons in the hidden layers of the radial
                fully connected network
            navg_neighbors (float): typical number of nodes at a distance ``max_radius``
            act (torch.nn.Module): activation for gated convolution
            cache (str | None): Default: None. Cache for precomputed values. 

        Returns:
            None

        """
        super().__init__()
        self.base_model = Model(
            irreps_in=irreps_in,
            hidden_sizes=hidden_sizes,
            irreps_out=f'{natoms * 3}x0e',
            nlayers=nlayers,
            max_radius=max_radius,
            nbasis_funcs=nbasis_funcs,
            nradial_layers=nradial_layers,
            nradial_neurons=nradial_neurons,
            navg_neighbors=navg_neighbors,
            act=act,
            cache=cache
        )
        self._natoms = natoms
        self._out_act = torch.nn.Sigmoid()

    def forward(self, data: Union[Data, Dict[str, torch.Tensor]]) -> torch.Tensor:
        """
        Forward pass of the network.

        Args:
            data (torch_geometric.data.Data | dict): data object containing
                - ``pos`` the position of the nodes (atoms)
                - ``x`` the input features of the nodes, optional
                - ``z`` the attributes of the nodes, for instance the atom type, optional
                - ``batch`` the graph to which the node belong, optional
                - *additional attributes

        Returns:
            x (torch.Tensor): output

        """
        out = self.base_model(data)
        return out.reshape(self._natoms, 3)
