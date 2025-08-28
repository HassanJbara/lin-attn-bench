from evals.mad.adapters.deltanet_adapter import DeltaNetAdapter
from evals.mad.wrappers.base_train_script import run_training
from mad.model.layers import SwiGLU

if __name__ == "__main__":
    dim = 128
    config = {"dim": dim}
    layers = [DeltaNetAdapter, SwiGLU, DeltaNetAdapter, SwiGLU]

    run_training(
        "DeltaNet",
        layers=layers,
        layer_configs=[config, config, config, config],
    )
