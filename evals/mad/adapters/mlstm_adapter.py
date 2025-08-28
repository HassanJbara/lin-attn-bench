import torch
from torch import nn

from fla.models.xlstm import mLSTMLayerConfig, mLSTMLayer
from fla.models.xlstm.blocks.mlstm.backends import parallel_sigmoid_simple, parallel_sigmoid_tensor


class mLSTMLayerAdapter(nn.Module):
    def __init__(self, bias=False, max_length=2048, conv1d_kernel_size=4, dropout=0.0, dim=1024,
                 num_heads=4, proj_factor=2.0, qkv_proj_blocksize=4, round_proj_up_dim_up=True,
                 round_proj_up_to_multiple_of=64, use_pscan=False):
        super().__init__()
        config = mLSTMLayerConfig(
            bias=bias,
            context_length=max_length,
            conv1d_kernel_size=conv1d_kernel_size,
            dropout=dropout,
            embedding_dim=dim,
            num_heads=num_heads,
            proj_factor=proj_factor,
            qkv_proj_blocksize=qkv_proj_blocksize,
            round_proj_up_dim_up=round_proj_up_dim_up,
            round_proj_up_to_multiple_of=round_proj_up_to_multiple_of
        )
        self.model = mLSTMLayer(config)
        if use_pscan:
            self.model.mlstm_cell.backend_fn = parallel_sigmoid_simple

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.model(x, **kwargs)
