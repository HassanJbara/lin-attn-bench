from evals.mad.adapters.gated_deltanet_adapter import GatedDeltaNetAdapter
from evals.mad.wrappers.base_train_script import run_training
from mad.model.layers import SwiGLU

if __name__ == "__main__":
    dim = 128
    config = {"dim": dim}
    layers = [GatedDeltaNetAdapter, SwiGLU, GatedDeltaNetAdapter, SwiGLU]

    run_training(
        "GatedDeltaNet",
        layers=layers,
        layer_configs=[config, config, config, config],
    )
