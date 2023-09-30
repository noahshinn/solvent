import torch


def center_coords(coords: torch.Tensor) -> torch.Tensor:
    """
    Shifts the xyz coordinates of an atomic system to be centered at the
    center of position.

    N: Number of atoms
    
    Args:
        coords (torch.Tensor): Tensor of size (N, 3).
          
    Returns:
        (torch.Tensor): Centered coordinates.

    """
    center = coords.mean(dim=0)
    return coords - center
