import torch
from torch import nn

from fla.layers.path_attn import PaTHAttention


class PaTHAttentionAdapter(nn.Module):
    def __init__(
        self, dim, num_heads, use_forget_gate, use_qk_norm, use_w_shortconv, max_length
    ):
        super().__init__()
        self.model = PaTHAttention(
            hidden_size=dim,
            num_heads=num_heads,
            use_forget_gate=use_forget_gate,
            use_qk_norm=use_qk_norm,
            use_w_shortconv=use_w_shortconv,
        )

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        y, _, _ = self.model(x, **kwargs)
        return y
