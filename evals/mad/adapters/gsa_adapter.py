import torch
from torch import nn

from fla.layers.gsa import GatedSlotAttention


class GatedSlotAttentionAdapter(nn.Module):
    def __init__(
        self,
        dim,
        max_length, 
        expand_k: float = 1.,
        expand_v: float = 1.,
        num_heads: int = 1,
        use_short_conv: bool = False,
        conv_size: int = 4,
        conv_bias: bool = False,
        norm_eps: float = 1e-5,
        gate_logit_normalizer: int = 8,
        feature_map: str = 'swish',
        use_output_gate: bool = False,
        use_norm: bool = True,
    ):
        super().__init__()
        self.model = GatedSlotAttention(
            hidden_size=dim,
            expand_k=expand_k,
            expand_v=expand_v,
            num_heads=num_heads,
            use_short_conv=use_short_conv,
            conv_size=conv_size,
            conv_bias=conv_bias,
            norm_eps=norm_eps,
            gate_logit_normalizer=gate_logit_normalizer,
            feature_map=feature_map,
            use_output_gate=use_output_gate,
            use_norm=use_norm,
        )

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        y, _, _ = self.model(x, **kwargs)
        return y
