import torch
from torch import nn

from fla import GatedDeltaNet


class GatedDeltaNetAdapter(nn.Module):
    def __init__(self, dim, max_length):
        super().__init__()
        self.model = GatedDeltaNet(dim, expand_v=1, head_dim=dim, num_heads=1)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        y, _, _ = self.model(x, **kwargs)
        return y
