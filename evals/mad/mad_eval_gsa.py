from evals.mad.adapters.gsa_adapter import GatedSlotAttentionAdapter
from evals.mad.wrappers.base_train_script import run_training
from mad_lab.mad.model.layers import SwiGLU

if __name__ == "__main__":
    dim = 128
    config = {"dim": dim}
    layers = [GatedSlotAttentionAdapter, SwiGLU, GatedSlotAttentionAdapter, SwiGLU]

    run_training(
        "GSA",
        layers=layers,
        layer_configs=[config, config, config, config],
    )
