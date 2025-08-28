from evals.mad.adapters.gated_deltaproduct_adapter import GatedDeltaProductAdapter
from evals.mad.wrappers.base_train_script import run_training
from mad.model.layers import SwiGLU

if __name__ == "__main__":
    dim = 128
    config = {"dim": dim}
    deltaproduct_config = {
        "dim": dim,
        "allow_neg_eigval": True,
        "num_heads": 1,
        "num_householder": 6,
    }
    layers = [GatedDeltaProductAdapter, SwiGLU, GatedDeltaProductAdapter, SwiGLU]

    run_training(
        "GatedDeltaProduct",
        layers=layers,
        layer_configs=[deltaproduct_config, config, deltaproduct_config, config],
    )
