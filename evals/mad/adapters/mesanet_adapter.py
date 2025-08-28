import torch
from torch import nn

from fla.layers.mesa_net import MesaNet


class MesaNetAdapter(nn.Module):
    def __init__(
        self,         
        max_length,
        dim: int = 2048,
        num_heads: int = 16,
        expand_v: float = 1,
        use_gate: bool = False,
        use_short_conv: bool = True,
        conv_size: int = 4,
        conv_bias: bool = False,
        norm_eps: float = 1e-5,
        lambda_lower_bound: float = 0.25,
        max_cg_step_training: int = 30,
    ):
        super().__init__()
        self.model = MesaNet(
            hidden_size=dim,
            num_heads=num_heads,
            exapnd_v=expand_v,
            use_gate=use_gate,
            use_short_conv=use_short_conv,
            conv_size=conv_size,
            conv_bias=conv_bias,
            norm_eps=norm_eps,
            lambda_lower_bound=lambda_lower_bound,
            max_cg_step_training=max_cg_step_training,
        )

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        y, _, _ = self.model(x, **kwargs)
        return y
