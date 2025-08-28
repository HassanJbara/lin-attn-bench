from evals.mad.adapters.mesanet_adapter import MesaNetAdapter
from evals.mad.wrappers.base_train_script import run_training
from mad.model.layers import SwiGLU

if __name__ == "__main__":
    dim = 128
    config = {"dim": dim}
    mesanet_config = {
        "dim": dim,
        "num_heads": 1,
    }
    layers = [MesaNetAdapter, SwiGLU, MesaNetAdapter, SwiGLU]

    run_training(
        "MesaNet",
        layers=layers,
        layer_configs=[mesanet_config, config, mesanet_config, config],
    )
