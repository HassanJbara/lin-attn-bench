from typing import Callable

from mad.benchmark import check_benchmark_data_present
from mad.configs import make_benchmark_mad_configs, MADConfig
from mad.paths import make_log_path
from mad.train import train


def benchmark_wrapper(
    model_id: str,
    data_path: str = "./benchmark/data",
    logs_path: str = "./benchmark/logs",
    log_to_csv: bool = True,
    log_to_wandb: bool = False,
    num_data_workers: int = 2,
    wandb_project: str = "MAD",
    save_checkpoints: bool = True,
    precision: str = "bf16",
    persistent_workers: bool = True,
):
    """
    Benchmark a model on MAD.

    Args:
        make_model_fn (callable): function that returns a PyTorch model
        model_id (str): unique identifier for the model
        mad_configs (list): list of MADConfig objects
        gpus (int): number of gpus to use for training
        cpus (int): number of cpus to use for training
        num_trials_gpu (int): number of trials to run per gpu
        num_cpus_trial (int): number of cpus to allocate to each trial
        logs_path (str): path where logs are stored
        log_to_csv (bool): if True, training metrics are locally saved to csv
        log_to_wandb (bool): if True, training metrics are logged to Weights & Biases
        wandb_project (str): project name to use when logging to Weights & Biases
        save_checkpoints (bool): if True, last and best model checkpoints of each training run are saved in the log directory
        ray_tmp_path (str): tmp path to be used by ray

    Returns:
        MAD scores for the model
    """
    # create all MAD configs for benchmark:
    mad_configs = make_benchmark_mad_configs(
        data_path=data_path,
        precision=precision,
        persistent_workers=persistent_workers,
        num_data_workers=num_data_workers,
    )
    check_benchmark_data_present(mad_configs)

    def setup_model_and_train(mad_config: MADConfig, make_model_fn: Callable):
        """Helper to setup model and train it according to MAD config."""
        log_path = make_log_path(
            base_path=logs_path,
            mad_config=mad_config,
            model_id=model_id,
        )
        model = make_model_fn(
            task=mad_config.task,
            vocab_size=mad_config.vocab_size,
            max_length=mad_config.seq_len,
        )
        results = train(
            model=model,
            mad_config=mad_config,
            log_path=log_path,
            log_to_csv=log_to_csv,
            log_to_wandb=log_to_wandb,
            save_checkpoints=save_checkpoints,
            wandb_project=wandb_project,
        )
        return results

    return mad_configs, setup_model_and_train
