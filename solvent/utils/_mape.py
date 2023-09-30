import torch


def mape(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    """Mean absolute percentage error"""
    return torch.div(x2 - x1, x2).abs().sum() / x1.numel()
