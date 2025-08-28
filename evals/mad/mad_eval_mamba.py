from torch import nn

from mad.benchmark import benchmark
from mad.model import AutoEncoder
from mad.model.language_model import LanguageModel
from mad.model.layers import Mamba


def make_model_fn(
        task: str,
        vocab_size: int,
        max_length: int,
) -> nn.Module:
    layers = [Mamba, Mamba, Mamba, Mamba]

    config = {'dim': 128, 'max_length': max_length}
    layer_configs = [config, config, config, config]

    backbone = LanguageModel if task not in {'compression'} else AutoEncoder

    return backbone(
        vocab_size=vocab_size,
        max_length=max_length,
        layers=layers,
        layer_cfgs=layer_configs
    )


mad_scores = benchmark(make_model_fn=make_model_fn, model_id="Mb-Mb-Mb-Mb", log_parameter_counts_and_quit=True)
