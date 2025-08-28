import argparse
import ray
import ray.util.multiprocessing as mp

from evals.mad.wrappers.benchmark_wrapper import benchmark_wrapper
from mad_lab.benchmark import benchmark
from mad_lab.mad.model import AutoEncoder
from mad_lab.mad.model.language_model import LanguageModel


def run_training(
    default_model_id,
    adapter_cls=None,
    adapter_config=None,
    layers=None,
    layer_configs=None,
):
    assert adapter_cls or layers, "Either adapter_cls or layers must be provided"

    layers = [adapter_cls] if layers is None else layers

    config = {"dim": 16} if adapter_config is None else adapter_config.copy()
    layer_configs = [config] if layer_configs is None else layer_configs
    dim = layer_configs[0]["dim"]

    assert all(cfg["dim"] == dim for cfg in layer_configs), (
        "All layers must have the same dimension"
    )

    def make_model_fn(task: str, vocab_size: int, max_length: int):
        for cfg in layer_configs:
            cfg["max_length"] = max_length

        backbone = LanguageModel if task not in {"compression"} else AutoEncoder
        return backbone(
            dim=dim,
            vocab_size=vocab_size,
            max_length=max_length,
            layers=layers,
            layer_cfgs=layer_configs,
        )

    def select_gpu_and_train(args):
        import torch

        job_id, (mad_config, setup_model_and_train), n_gpu = args
        gpu_id = job_id % n_gpu
        torch.cuda.device(gpu_id)

        return setup_model_and_train(mad_config, make_model_fn)

    parser = argparse.ArgumentParser(
        description=f"Train {default_model_id} models with Ray and multiprocessing."
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default=default_model_id,
        help="Model ID for benchmarking.",
    )
    parser.add_argument(
        "--n-gpu", type=int, default=1, help="Number of GPUs available."
    )
    parser.add_argument(
        "--n-tasks-gpu", type=int, default=2, help="Number of tasks per GPU."
    )
    parser.add_argument(
        "--ray-tmp-path", type=str, default="/tmp/ray", help="Temporary path for Ray."
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="./benchmark/data",
        help="Path to benchmark data.",
    )
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--log-parameters", action="store_true")
    args = parser.parse_args()

    n_gpu = args.n_gpu
    n_tasks_gpu = args.n_tasks_gpu

    # Use direct benchmark for single GPU/task case
    if n_gpu == n_tasks_gpu == 1:
        benchmark(make_model_fn=make_model_fn, model_id=args.model_id, log_parameter_counts_and_quit=args.log_parameters, data_path=args.data_path)
    else:
        if not ray.is_initialized():
            ray.init(num_gpus=n_gpu, _temp_dir=args.ray_tmp_path)

        pool = mp.Pool(n_gpu * n_tasks_gpu)

        mad_configs, setup_function = benchmark_wrapper(
            args.model_id, data_path=args.data_path, num_data_workers=args.num_workers
        )

        task_args = [
            (i, (mad_config, setup_function), n_gpu)
            for i, mad_config in enumerate(mad_configs)
        ]
        instances = pool.map(
            ray.remote(num_gpus=1.0 / n_tasks_gpu)(select_gpu_and_train).remote,
            task_args,
        )
        ray.get(instances)
