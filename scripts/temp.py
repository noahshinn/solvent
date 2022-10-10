import torch

def func(a: torch.Tensor, b: str) -> torch.Tensor:
    return str(a) + b
