import torch


def from_mc(out: torch.Tensor) -> int:
    """Converts a tensor to a multi-class label."""
    return int(torch.argmax(out).item())
