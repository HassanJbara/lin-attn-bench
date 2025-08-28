import torch
from torch import nn

from fla.layers.gla import GatedLinearAttention


class GatedLinearAttentionAdapter(nn.Module):
    def __init__(
        self,
        dim,
        max_length, 
        expand_k: float = 0.5,
        expand_v: float = 1.0,
        num_heads: int = 1,
        use_short_conv: bool = False,
        conv_size: int = 4,
        conv_bias: bool = False,
        use_output_gate: bool = True,
        gate_fn: str = 'swish',
        norm_eps: float = 1e-5,
        gate_logit_normalizer: int = 16,
        gate_low_rank_dim: int = 16,
        fuse_norm: bool = True,
        layer_idx: int = None,
    ):
        super().__init__()
        self.model = GatedLinearAttention(
            hidden_size=dim,
            expand_k=expand_k,
            expand_v=expand_v,
            num_heads=num_heads,
            use_short_conv=use_short_conv,
            conv_size=conv_size,
            conv_bias=conv_bias,
            use_output_gate=use_output_gate,
            gate_fn=gate_fn,
            norm_eps=norm_eps,
            gate_logit_normalizer=gate_logit_normalizer,
            gate_low_rank_dim=gate_low_rank_dim,
            fuse_norm=fuse_norm,
            layer_idx=layer_idx,
        )

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        y, _, _ = self.model(x, **kwargs)
        return y
