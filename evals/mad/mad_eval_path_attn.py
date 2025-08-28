from evals.mad.mad.adapters.path_attention_adapter import PaTHAttentionAdapter
from evals.mad.wrappers.base_train_script import run_training
from mad.model.layers import SwiGLU

if __name__ == "__main__":
    dim = 128
    config = {"dim": dim}
    path_config = {
        "dim": dim,
        "num_heads": 2,
        "use_forget_gate": False,
        "use_qk_norm": False,
        "use_w_shortconv": True,
    }
    layers = [PaTHAttentionAdapter, SwiGLU, PaTHAttentionAdapter, SwiGLU]

    run_training(
        "PathAttn",
        layers=layers,
        layer_configs=[path_config, config, path_config, config],
    )
