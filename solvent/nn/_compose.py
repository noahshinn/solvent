import torch


class Compose(torch.nn.Module):
    def __init__(self, first: torch.nn.Module, second: torch.nn.Module) -> None:
        super().__init__()
        self.first = first
        self.second = second
        self.irreps_in = self.first.irreps_in
        self.irreps_out = self.second.irreps_out

    def forward(self, *input: torch.Tensor) -> torch.Tensor:
        x = self.first(*input)
        return self.second(x)
