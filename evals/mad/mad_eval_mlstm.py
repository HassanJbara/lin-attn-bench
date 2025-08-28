from evals.mad.adapters.mlstm_adapter import mLSTMLayerAdapter
from evals.mad.wrappers.base_train_script import run_training
from mad.model.layers import SwiGLU

if __name__ == "__main__":
    dim = 128
    config = {"dim": dim}
    mlstm_config = {
        "bias": False,
        "conv1d_kernel_size": 4,
        "dropout": 0.0,
        "dim": dim,
        "num_heads": 1,
        "proj_factor": 2.0,
        "qkv_proj_blocksize": 4,
        "round_proj_up_dim_up": True,
        "round_proj_up_to_multiple_of": 64,
        "use_pscan": False
    }
    layers = [mLSTMLayerAdapter, SwiGLU, mLSTMLayerAdapter, SwiGLU]
    
    run_training("xLSTM", layers=layers, layer_configs=[mlstm_config, config, mlstm_config, config])