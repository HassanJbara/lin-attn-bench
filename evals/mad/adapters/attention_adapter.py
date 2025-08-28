import torch
from torch import nn

from fla.layers.attn import Attention


class AttentionAdapter(nn.Module):
    def __init__(
        self, dim, num_heads, qk_norm, max_length, window_size,
    ):
        super().__init__()
        self.model = Attention(
            hidden_size=dim,
            num_heads=num_heads,
            qk_norm=qk_norm,
            max_position_embeddings=max_length,
            window_size=window_size,
        )

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        y, _, _ = self.model(x, **kwargs)
        return y
