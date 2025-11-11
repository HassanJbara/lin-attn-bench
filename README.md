# Linear Attention Benchmarks

## FineWeb EDU Benchmark Results

| Name | Global Avg Loss | Global Max Loss | Size |
| :--- | :---: | :---: | :---: |
| Mamba2 | 2.48171 | 2.5153 | 380M |
| Mamba | 2.52864 | 2.56511 | 398M |
| Gated Linear Attention | 2.73029 | 2.82961 | 374M |
| Transformer++ | 2.52412 | 2.6376 | 373M |
| Gated DeltaNet | 2.45253 | 2.57175 | 512M |
| Gated Deltaproduct | 2.62878 | 2.68645 | 349M |
| MesaNet | 2.68138 | 2.7389 | 348M |
| DeltaNet | 2.59971 | 2.65893 | 374M |

### Running the benchmarks

You will need a linux machine equipped with NVIDIA GPUs. To setup the environment for running these benchmarks you should do the following:

1. Install `fla` according to the instructions [here](https://github.com/fla-org/flame).
   * You might need to install `tryo` to fix `torchtitan` dependency issues.
2. Prepare the data according to the instructions in the `fla` repository.
3. Configure `fla.toml` according to your system. *Important*: We noticed that changing parallelism settings can affect the results, but they are generally within the same ballpark.
4. Use the configurations provided in `fla` to train the models with the command provided in the flame repo.

### Dataset

We run the training for `28600` steps on the `HuggingFaceFW/fineweb-edu` subset `sample-100BT` with seed 42. This is ~15B tokens.
