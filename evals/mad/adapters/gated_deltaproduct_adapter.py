import torch
from torch import nn

from fla.layers.gated_deltaproduct import GatedDeltaProduct


class GatedDeltaProductAdapter(nn.Module):
    def __init__(
        self,
        dim,
        max_length,
        allow_neg_eigval: bool = False,  # when true (Gated) DeltaProduct [-1, 1], when false (Gated) DeltaProduct [0, 1]
        num_heads: int = 1,
        num_householder: int = 2,  # New parameter for number of householder transformations
    ):
        super().__init__()
        self.model = GatedDeltaProduct(
            dim,
            expand_v=1,
            head_dim=dim,
            num_heads=num_heads,
            allow_neg_eigval=allow_neg_eigval,
            num_householder=num_householder,
        )

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        y, _, _ = self.model(x, **kwargs)
        return y
