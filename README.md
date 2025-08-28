# Linear Attention Benchmarks

## MAD

| Model | Compression | Fuzzy-In-Context | In-Context | Memorization | Noisy-In-Context | Selective-Copying | Average | Model Size | Implementation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| PaTHAttention | 0.419 | 0.647 | 0.999 | 0.784 | 0.999 | 0.999 | 0.808 | ~450K |[fla](https://github.com/fla-org/flash-linear-attention) |
| Transformer | 0.432 | 0.565 | 0.905 | 0.788 | 0.897 | 0.998 | 0.764 | ~400K | [fla](https://github.com/fla-org/flash-linear-attention) |
| MesaNet | 0.347 | 0.515 | 0.999 | 0.750 | 0.999 | 0.894 | 0.751 | ~400K | [fla](https://github.com/fla-org/flash-linear-attention) |
| mLSTM | 0.478 | 0.237 | 1.000 | 0.896 | 1.000 | 0.870 | 0.747 | ~400K | [fla](https://github.com/fla-org/flash-linear-attention) |
| RWKV 7 | 0.462 | 0.180 | 0.999 | 0.886 | 0.999 | 0.949 | 0.746 | ~550K | [fla](https://github.com/fla-org/flash-linear-attention) |
| Mamba | 0.491 | 0.123 | 0.993 | 0.896 | 0.997 | 0.887 | 0.731 | ~400K | [fla](https://github.com/fla-org/flash-linear-attention) |
| GatedDeltaProduct | 0.416 | 0.277 | 0.999 | 0.648 | 0.999 | 0.999 | 0.723 | ~750K | [fla](https://github.com/fla-org/flash-linear-attention) |
| Gated DeltaNet | 0.435 | 0.286 | 0.999 | 0.552 | 0.999 | 0.997 | 0.712 | ~450K | [fla](https://github.com/fla-org/flash-linear-attention) |
| mLSTM | 0.387 | 0.268 | 0.999 | 0.843 | 0.998 | 0.690 | 0.698 | ~500K | |
| DeltaNet | 0.396 | 0.393 | 0.999 | 0.394 | 0.999 | 0.997 | 0.697 | ~450K | [fla](https://github.com/fla-org/flash-linear-attention) |
| Gated Linear Attention | 0.408 | 0.155 | 0.918 | 0.771 | 0.931 | 0.891 | 0.679 | ~425K | [fla](https://github.com/fla-org/flash-linear-attention) |
| Gated Slot Attention | 0.397 | 0.212 | 0.769 | 0.831 | 0.821 | 0.852 | 0.648 | ~450K | [fla](https://github.com/fla-org/flash-linear-attention) |

### Running the benchmarks

You will need a linux machine equipped with NVIDIA GPUs. To setup the environment for running these benchmarks you should do the following:

1. Install pytorch version `2.5.x` or `2.6.x`. This is important, since we use ray for multi-GPU scripts and ray isn't compatible with `triton > 3.1`, which come with any pytorch version newer than `2.6`, but fla requires `pytorch >= 2.5`.
2. Install dependencies in requirements.txt. This may take a while because of `causal-conv1d`.
3. Install [flash-attention](https://github.com/Dao-AILab/flash-attention).
4. Clone the [MAD repo](https://github.com/athms/mad-lab) and install the requirements.
5. Rename `mad-lab` to `mad_lab`.
6. Move `mad_lab/configs/tasks` to `./configs/tasks`.
7. Download the benchmarking data from <https://zenodo.org/records/10843663> and place it in `./benchmark/data/`.
8. Run a script from `./evals/mad/` using the following command:

```bash
PYTHONPATH=.:"$(pwd)/mad_lab" python -m evals.mad.mad_eval_gsa --data-path ./benchmark/data --num-workers X --n-gpu X --n-tasks-gpu X`
```
