"""
STATUS: NOT TESTED

"""

import math
import torch
from torch_scatter import scatter

from e3nn import o3
from e3nn.nn import FullyConnectedNet
from e3nn.util.jit import compile_mode

from typing import Optional


@compile_mode("script")
class InteractionBlock(torch.nn.Module):
    """
    Equivariant convolutional layer.

    """
    def __init__(
            self,
            irreps_in: o3.Irreps,
            irreps_node_attr: o3.Irreps,
            irreps_edge_attr: o3.Irreps,
            irreps_out: Optional[o3.Irreps],
            number_of_basis: int,
            radial_layers: int,
            radial_neurons: int,
            num_neighbors: float
        ) -> None:
        """
        Initializes the layer.

        Args:
            irreps_in (e3nn.o3.Irreps): representation of the input node features
            irreps_node_attr (e3nn.o3.Irreps): representation of the node
                attributes
            irreps_edge_attr (e3nn.o3.Irreps): representation of the edge
                attributes
            irreps_out (e3nn.o3.Irreps | None): representation of the output node
                features
            number_of_basis (int): number of basis on which the edge length are
                projected
            radial_layers (int): number of hidden layers in the radial fully
                connected network
            radial_neurons (int): number of neurons in the hidden layers of the
                radial fully connected network
            num_neighbors (float): typical number of nodes convolved over

        Returns:
            None

        """
        super().__init__()
        self.irreps_in = o3.Irreps(irreps_in)
        self.irreps_node_attr = o3.Irreps(irreps_node_attr)
        self.irreps_edge_attr = o3.Irreps(irreps_edge_attr)
        self.irreps_out = o3.Irreps(irreps_out)
        self.num_neighbors = num_neighbors

        self.sc = o3.FullyConnectedTensorProduct(
            self.irreps_in,
            self.irreps_node_attr,
            self.irreps_out
        )

        self.lin1 = o3.FullyConnectedTensorProduct(
            self.irreps_in,
            self.irreps_node_attr,
            self.irreps_in
        )

        irreps_mid = []
        instructions = []
        for i, (mul, ir_in) in enumerate(self.irreps_in):
            for j, (_, ir_edge) in enumerate(self.irreps_edge_attr):
                for ir_out in ir_in * ir_edge:
                    if ir_out in self.irreps_out:
                        k = len(irreps_mid)
                        irreps_mid.append((mul, ir_out))
                        instructions.append((i, j, k, "uvu", True))
        irreps_mid = o3.Irreps(irreps_mid)
        irreps_mid, p, _ = irreps_mid.sort() # type: ignore

        instructions = [(i_1, i_2, p[i_out], mode, train) for i_1, i_2, i_out, mode, train in instructions]

        tp = o3.TensorProduct(
            self.irreps_in, # type: ignore
            self.irreps_edge_attr,
            irreps_mid,
            instructions,
            internal_weights=False,
            shared_weights=False,
        )
        self.fc = FullyConnectedNet(
            [number_of_basis] + radial_layers * [radial_neurons] + [tp.weight_numel], # type: ignore
            torch.nn.functional.silu
        )
        self.tp = tp

        self.lin2 = o3.FullyConnectedTensorProduct(
            irreps_mid,
            self.irreps_node_attr,
            self.irreps_out
        )

    def forward(
            self,
            node_input: torch.Tensor,
            node_attr: torch.Tensor,
            edge_src: torch.Tensor,
            edge_dst: torch.Tensor,
            edge_attr: torch.Tensor,
            edge_length_embedded: torch.Tensor
        ) -> torch.Tensor:
        """
        Forward pass of the interaction block.

        """
        weight = self.fc(edge_length_embedded)

        x = node_input

        s = self.sc(x, node_attr)
        x = self.lin1(x, node_attr)

        edge_features = self.tp(x[edge_src], edge_attr, weight)
        x = scatter(
            edge_features,
            edge_dst,
            dim=0,
            dim_size=x.shape[0]
        ).div(self.num_neighbors ** 0.5)

        x = self.lin2(x, node_attr)

        c_s = math.sin(math.pi / 8)
        c_x = math.cos(math.pi / 8)
        m = self.sc.output_mask
        c_x = 1 - m + c_x * m # type: ignore
        return c_s * s + c_x * x

