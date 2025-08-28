from evals.mad.adapters.attention_adapter import AttentionAdapter
from evals.mad.wrappers.base_train_script import run_training
from mad.model.layers import SwiGLU

if __name__ == "__main__":
    dim = 128
    config = {"dim": dim}
    attn_config = {
        "dim": dim,
        "num_heads": 2,
        "qk_norm": False,
        "window_size": 128,
    }
    layers = [AttentionAdapter, SwiGLU, AttentionAdapter, SwiGLU]

    run_training(
        "Transformer-window-128",
        layers=layers,
        layer_configs=[attn_config, config, attn_config, config],
    )
