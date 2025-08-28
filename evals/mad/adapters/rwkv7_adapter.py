import torch
from torch import nn

from fla import RWKV7Attention


class RWKV7Adapter(nn.Module):
    def __init__(self, dim, max_length):
        super().__init__()
        self.model = RWKV7Attention(hidden_size=dim, head_dim=dim, num_heads=1, layer_idx=0, num_hidden_layers=2)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        y, _, _, _ = self.model(x.to(torch.bfloat16), **kwargs)
        return y
