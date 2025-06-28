# GPU Optimization Benchmarks

This directory contains benchmark scripts to compare GPU optimizations against the baseline implementation.

## Quick Benchmark

For a quick comparison of baseline vs GPU-optimized performance:

```bash
python quick_gpu_benchmark.py
```

This script:
- Uses the test data in `data/` directory
- Runs `train_holdout` with 50 holdout samples
- Compares baseline (no GPU opts) vs GPU-optimized configurations
- Takes ~2-5 minutes to complete
- Outputs results to console and `gpu_benchmark_results.csv`

## Comprehensive Benchmark

For detailed benchmarking with multiple configurations and statistical analysis:

```bash
python benchmark_gpu_optimizations.py --n-runs 5 --holdout-size 50
```

Options:
- `--n-runs`: Number of runs per configuration (default: 5)
- `--holdout-size`: Number of samples to hold out (default: 50)
- `--output-dir`: Directory for results (default: benchmark_results/)

This script tests:
1. **Baseline**: No GPU optimizations
2. **GPU Optimized**: Default GPU optimizations
3. **Large Batch**: Fixed large batch size (256)
4. **GPU + XLA**: GPU optimizations with XLA compilation

Outputs:
- `benchmark_results/benchmark_raw_results.csv`: Raw data from all runs
- `benchmark_results/benchmark_summary.csv`: Statistical summary
- `benchmark_results/benchmark_results.png`: Visualization of results
- `benchmark_results/speedup_comparison.png`: Speedup comparison chart

## Expected Results

With GPU optimizations enabled, you should see:

- **3-5x speedup** in training time (dataset-dependent)
- **2-3x higher throughput** (samples/second)
- **Larger batch sizes** (auto-optimized based on GPU memory)
- **Similar or better model quality** (validation loss, prediction accuracy)

Factors affecting speedup:
- GPU model (consumer vs data center GPUs)
- Dataset size (larger datasets show more benefit)
- Model complexity
- System configuration

## Interpreting Results

The benchmarks measure:

1. **Training Time**: Total time to train the model
2. **Throughput**: Samples processed per second
3. **Batch Size**: Actual batch size used (auto-optimized or fixed)
4. **Model Quality**: Validation loss and holdout prediction error
5. **GPU Memory**: Memory usage during training

Key metrics to compare:
- **Speedup**: Training time reduction vs baseline
- **Throughput Ratio**: Increase in samples/second
- **Efficiency**: GPU utilization and memory usage

## Troubleshooting

If you don't see speedup:

1. **Check GPU availability**:
   ```python
   import tensorflow as tf
   print(tf.config.list_physical_devices('GPU'))
   ```

2. **Verify mixed precision support**:
   ```python
   from locator.gpu_optimizer import GPUOptimizer
   GPUOptimizer.setup_mixed_precision()
   ```

3. **Monitor GPU usage**:
   ```bash
   watch -n 1 nvidia-smi
   ```

4. **Try larger datasets**: GPU benefits increase with data size

## Test Data

The benchmarks use test data in `data/`:
- `test_genotypes.vcf.gz`: 500 samples, subset of SNPs
- `test_sample_data.txt`: Sample coordinates (many with NA)

This is a small dataset for testing. Real-world datasets with >10K samples and >100K SNPs will show more dramatic improvements.