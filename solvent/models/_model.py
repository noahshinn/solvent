"""
e3nn gate_points_2101 network

Additional features:
    - use custom activation function in gated convolution
    - hidden size config more readable
    - graph compute caching

"""

import pickle
import torch
from torch_geometric.data import Data
from torch_cluster import radius_graph
from torch_scatter import scatter

from e3nn import o3
from e3nn.nn import Gate
from e3nn.math import soft_one_hot_linspace

from solvent import nn
from solvent.nn import Compose, InteractionBlock

from typing import Dict, Union, Optional, List


class Model(torch.nn.Module):
    """
    An equivariant graph neural network.

    """
    def __init__(
            self,
            irreps_in: Union[o3.Irreps, str, None],
            hidden_sizes: List[int],
            irreps_out: Union[o3.Irreps, str],
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
            irreps_out (e3nn.o3.Irreps | str): representation of the output features
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
        
        if not cache is None:
            with open(cache, 'rb') as h:
                self._cache = pickle.load(h)  
        else:
            self._cache = {}

        self.max_radius = max_radius
        self.number_of_basis = nbasis_funcs 
        self.num_neighbors = navg_neighbors 

        self.irreps_in = o3.Irreps(irreps_in) if irreps_in is not None else None
        _hidden_sizes = [(mul, (l, p)) for l, mul in enumerate(hidden_sizes) for p in [-1, 1]]
        self.irreps_hidden = o3.Irreps(_hidden_sizes)
        self.irreps_out = o3.Irreps(irreps_out)
        self.irreps_node_attr = o3.Irreps("0e")
        self.irreps_edge_attr = o3.Irreps.spherical_harmonics(3)
        irreps = self.irreps_in

        if act is None:
            self.act = {
                1: torch.nn.functional.silu,
                -1: torch.tanh,
            }
        else:
            assert len(act) == 2 and -1 in act and 1 in act
            self.act = {
                1: act[1],
                -1: act[-1]
            }
        act_gates = {
            1: torch.sigmoid,
            -1: torch.tanh,
        }

        self.layers = torch.nn.ModuleList()

        for _ in range(nlayers):
            irreps_scalars = o3.Irreps(
                [
                    (mul, ir)
                    for mul, ir in self.irreps_hidden
                    if ir.l == 0 and computer.tp_path_exists(irreps, self.irreps_edge_attr, ir) # type: ignore
                ]
            )
            irreps_gated = o3.Irreps(
                    [(mul, ir) for mul, ir in self.irreps_hidden if ir.l > 0 and computer.tp_path_exists(irreps, self.irreps_edge_attr, ir)] # type: ignore
            )
            ir = "0e" if computer.tp_path_exists(irreps, self.irreps_edge_attr, "0e") else "0o" # type: ignore
            irreps_gates = o3.Irreps([(mul, ir) for mul, _ in irreps_gated])

            gate = Gate(
                irreps_scalars, # type: ignore
                [self.act[ir.p] for _, ir in irreps_scalars],
                irreps_gates,
                [act_gates[ir.p] for _, ir in irreps_gates],
                irreps_gated
            )
            conv = InteractionBlock(
                irreps, # type: ignore
                self.irreps_node_attr,
                self.irreps_edge_attr,
                gate.irreps_in,
                nbasis_funcs,
                nradial_layers,
                nradial_neurons,
                navg_neighbors 
            )
            irreps = gate.irreps_out
            self.layers.append(Compose(conv, gate))

        self.layers.append(
            InteractionBlock(
                irreps, # type: ignore
                self.irreps_node_attr,
                self.irreps_edge_attr,
                self.irreps_out,
                nbasis_funcs,
                nradial_layers,
                nradial_neurons,
                navg_neighbors 
            )
        )

    def __repr__(self) -> str:
        return f'Equivariant GNN with {len(self.layers)} convolutional layers.'

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
        if "batch" in data:
            batch = data["batch"]
        else:
            batch = data["pos"].new_zeros(data["pos"].shape[0], dtype=torch.long)
        
        # coord_hash = computer.hash_2d_tensor(data['pos'])
        # if coord_hash in self._cache:
            # _structure_data = self._cache[coord_hash]
            # edge_src = _structure_data['edge_src']
            # edge_dst = _structure_data['edge_dst']
        # else: 
            # edge_index = radius_graph(data["pos"], self.max_radius, batch)
            # edge_src = edge_index[0]
            # edge_dst = edge_index[1]
            # self._cache[coord_hash] = {
                # 'edge_src': edge_src,
                # 'edge_dst': edge_dst,
            # }
        edge_index = radius_graph(data["pos"], self.max_radius, batch)
        edge_src = edge_index[0]
        edge_dst = edge_index[1]

        edge_vec = data["pos"][edge_src] - data["pos"][edge_dst]
        _edge_sh = o3.spherical_harmonics(self.irreps_edge_attr, edge_vec, True, normalization="component") # type: ignore
        _edge_length = edge_vec.norm(dim=1)
        edge_length_embedded = soft_one_hot_linspace(
            x=_edge_length,
            start=0.0,
            end=self.max_radius,
            number=self.number_of_basis,
            basis="gaussian",
            cutoff=False
        ).mul(self.number_of_basis**0.5)
        edge_attr = nn.basis_cutoff(_edge_length, self.max_radius)[:, None] * _edge_sh

        x = data["x"]
        z = data["pos"].new_ones((data["pos"].shape[0], 1))

        for lay in self.layers:
            x = lay(x, z, edge_src, edge_dst, edge_attr, edge_length_embedded)

        return scatter(x, batch, dim=0).div(self.num_neighbors ** 0.5).flatten()
