from evals.mad.adapters.gla_adapter import GatedLinearAttentionAdapter
from evals.mad.wrappers.base_train_script import run_training
from mad.model.layers import SwiGLU

if __name__ == "__main__":
    dim = 128
    config = {"dim": dim}
    layers = [GatedLinearAttentionAdapter, SwiGLU, GatedLinearAttentionAdapter, SwiGLU]

    run_training(
        "GLA",
        layers=layers,
        layer_configs=[config, config, config, config],
    )
