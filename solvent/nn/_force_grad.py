"""
STATUS: DEV

improve with vmap

"""

import torch


def force_grad(energies: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
    """
    Computes the gradient of energy with respect to coordinate position
    for every atom for every electronic state for every batch.

    N: number of atoms
    K: number of electronic states

    Args:
        energies (torch.Tensor): Energy tensor of size (K)
        pos (torch.Tensor): Position tensor of size (N, 3)

    Returns:
        forces (torch.Tensor): Force tensor of size (K, N, 3)

    """
    nstates = energies.size(dim=0)
    if not pos.requires_grad:
        pos.requires_grad = True
    forces = []
    for i in range(nstates):
        f = torch.autograd.grad(
            inputs=-energies[i],
            outputs=pos,
            create_graph=True,
            retain_graph=True
        )[0]
        forces.append(f)
    forces = torch.stack(forces, dim=0)

    return forces
