import torch


def force_grad_with_vmap(
        energies: torch.Tensor,
        pos: torch.Tensor,
        device: str
    ) -> torch.Tensor:
    from functorch import vmap
    """
    IN DEVELOPMENT: vmap build does not currently work with torch-geometric
        requirements.

    Computes the gradient of energy with respect to coordinate position
    for every atom for every electronic state for every batch.

    N: number of atoms
    K: number of electronic states

    Args:
        energies (torch.Tensor): Energy tensor of size (K)
        pos (torch.Tensor): Position tensor of size (N, 3)

    Returns:
        jac (torch.Tensor): Jacobian maxtrix of force tensor of size (K, N, 3)

    """
    nstates = energies.size(dim=0)
    basis_vecs = torch.eye(nstates).to(device=device)
    def _vjp(v: torch.Tensor) -> torch.Tensor:
        return torch.autograd.grad(
            outputs=-energies,
            inputs=pos,
            grad_outputs=v
        )[0]
    jac = vmap(_vjp)(basis_vecs)
    return jac
    
def force_grad(
        energies: torch.Tensor,
        pos: torch.Tensor,
        device: str
    ) -> torch.Tensor:
    """
    Computes the gradient of energy with respect to coordinate position
    for every atom for every electronic state for every batch.

    N: number of atoms
    K: number of electronic states

    Args:
        energies (torch.Tensor): Energy tensor of size (K)
        pos (torch.Tensor): Position tensor of size (N, 3)

    Returns:
        jac (torch.Tensor): Jacobian maxtrix of force tensor of size (K, N, 3)

    """
    nstates = energies.size(dim=0)
    basis_vecs = torch.eye(nstates).to(device=device)
    jac_rows = [torch.autograd.grad(-energies, pos, v, retain_graph=True)[0] for v in basis_vecs.unbind()]
    jac = torch.stack(jac_rows, dim=0)
    return jac
