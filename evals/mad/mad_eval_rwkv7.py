from evals.mad.adapters.rwkv7_adapter import RWKV7Adapter
from evals.mad.wrappers.base_train_script import run_training
from mad.model.layers import SwiGLU

if __name__ == "__main__":
    dim = 128
    config = {"dim": dim}
    layers = [RWKV7Adapter, SwiGLU, RWKV7Adapter, SwiGLU]
    
    run_training("RWKV7", layers=layers, layer_configs=[config, config, config, config])