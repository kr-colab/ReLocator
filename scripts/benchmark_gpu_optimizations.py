#!/usr/bin/env python3
"""
GPU Optimization Benchmark for Locator

This script benchmarks the GPU optimization features against baseline performance.
It demonstrates the impact of mixed precision training, optimized batch sizes,
and efficient data pipelines on training speed.

IMPORTANT NOTES:
- GPU optimizations show best results with larger datasets (>10k samples)
- The test dataset (450 samples) is too small to fully showcase GPU benefits
- Large batch sizes may cause convergence issues on small datasets
- Expected speedups: 2-4x on datasets with >10k samples and modern GPUs

Usage:
    # From project root:
    python -m scripts.benchmark_gpu_optimizations [--epochs N] [--output results.json]

    # Or with absolute import:
    python scripts/benchmark_gpu_optimizations.py [--epochs N] [--output results.json]
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

# Add parent directory to path to import locator package
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import tensorflow as tf

from locator import Locator


class GPUBenchmark:
    """Benchmark suite for GPU optimizations."""

    def __init__(self, data_path: str = "data", epochs: int = 20):
        self.data_path = Path(data_path)
        self.epochs = epochs
        self.results = []

    def check_gpu(self) -> bool:
        """Check GPU availability and print info."""
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            print(f"✓ GPU found: {len(gpus)} device(s)")
            for i, gpu in enumerate(gpus):
                print(f"  Device {i}: {gpu.name}")
            return True
        else:
            print("✗ No GPU found - results may not show speedup")
            return False

    def get_configs(self) -> List[Tuple[str, dict]]:
        """Get benchmark configurations."""
        base_config = {
            "sample_data": str(self.data_path / "test_sample_data.txt"),
            "max_epochs": self.epochs,
            "patience": 10,
            "keras_verbose": 1,
            "na_action": "exclude",
        }

        return [
            (
                "Baseline (CPU-optimized)",
                {
                    **base_config,
                    "out": "benchmark_baseline",
                    "use_mixed_precision": False,
                    "gpu_batch_size": 32,
                    "use_efficient_pipeline": False,
                },
            ),
            (
                "GPU Optimized (auto batch)",
                {
                    **base_config,
                    "out": "benchmark_gpu_auto",
                    "use_mixed_precision": True,
                    "gpu_batch_size": "auto",  # Auto-detect optimal size
                    "use_efficient_pipeline": True,
                },
            ),
            (
                "GPU Optimized (fixed large batch)",
                {
                    **base_config,
                    "out": "benchmark_gpu_large",
                    "use_mixed_precision": True,
                    "gpu_batch_size": 256,  # Fixed large batch
                    "use_efficient_pipeline": True,
                },
            ),
        ]

    def run_single_benchmark(self, name: str, config: dict) -> dict:
        """Run a single benchmark configuration."""
        print(f"\n{'='*70}")
        print(f"Running: {name}")
        print(f"{'='*70}")

        # Clear session to ensure clean state
        tf.keras.backend.clear_session()

        # Create Locator
        loc = Locator(config)

        # Load data
        print("Loading data...")
        start = time.time()
        genotypes, samples = loc.load_genotypes(
            vcf=str(self.data_path / "test_genotypes.vcf.gz")
        )
        load_time = time.time() - start

        # Genotypes from VCF are (sites, samples, ploidy)
        if len(genotypes.shape) == 3:
            n_snps, n_samples, _ = genotypes.shape
        else:
            n_samples, n_snps = genotypes.shape
        print(f"  Loaded in {load_time:.2f}s")
        print(f"  Dataset: {n_samples} samples × {n_snps} SNPs")

        # Check batch size vs dataset size
        batch_size = config.get("gpu_batch_size", 32)
        if batch_size > n_samples * 0.1:  # More than 10% of dataset
            print(f"  ⚠️  Batch size {batch_size} is large for {n_samples} samples")
            print(f"     This may cause convergence issues")

        # Train
        print(f"\nTraining with batch_size={batch_size}...")
        start = time.time()
        history = loc.train(genotypes=genotypes, samples=samples)
        train_time = time.time() - start

        # Extract metrics
        epochs_run = len(history.history["loss"])
        final_loss = history.history["loss"][-1]
        best_val_loss = min(history.history["val_loss"])

        # Calculate effective throughput
        # Account for train/val split (default 90/10) and then train/test (90/10)
        n_train = int(n_samples * 0.9 * 0.9)
        total_samples_processed = n_train * epochs_run
        throughput = total_samples_processed / train_time

        # Memory usage (if available)
        memory_info = ""
        if tf.config.list_physical_devices("GPU"):
            try:
                # This would require nvidia-ml-py, just note it
                memory_info = "GPU memory tracking requires nvidia-ml-py"
            except:
                pass

        result = {
            "name": name,
            "config": {
                "batch_size": batch_size,
                "mixed_precision": config.get("use_mixed_precision", False),
                "efficient_pipeline": config.get("use_efficient_pipeline", False),
            },
            "dataset": {
                "n_samples": n_samples,
                "n_snps": n_snps,
            },
            "performance": {
                "load_time": load_time,
                "train_time": train_time,
                "epochs_run": epochs_run,
                "throughput": throughput,
                "samples_per_epoch": n_train,
            },
            "quality": {
                "final_loss": final_loss,
                "best_val_loss": best_val_loss,
            },
            "notes": memory_info,
        }

        # Print summary
        print(f"\nResults:")
        print(f"  Training time: {train_time:.2f}s ({epochs_run} epochs)")
        print(f"  Throughput: {throughput:.0f} samples/s")
        print(f"  Time per epoch: {train_time/epochs_run:.2f}s")
        print(f"  Best validation loss: {best_val_loss:.4f}")

        return result

    def run_all_benchmarks(self) -> List[dict]:
        """Run all benchmark configurations."""
        print("GPU Optimization Benchmark Suite")
        print("=" * 70)

        # Check GPU
        has_gpu = self.check_gpu()

        # Get configs
        configs = self.get_configs()

        # Run benchmarks
        results = []
        for name, config in configs:
            try:
                result = self.run_single_benchmark(name, config)
                results.append(result)
            except Exception as e:
                print(f"\n❌ Error in {name}: {e}")
                import traceback

                traceback.print_exc()

        return results

    def print_summary(self, results: List[dict]) -> None:
        """Print benchmark summary and analysis."""
        if len(results) < 2:
            print("\n⚠️  Not enough results for comparison")
            return

        print("\n" + "=" * 70)
        print("BENCHMARK SUMMARY")
        print("=" * 70)

        # Find baseline
        baseline = next((r for r in results if "Baseline" in r["name"]), results[0])

        # Print comparison table
        print(
            f"\n{'Configuration':<30} {'Time (s)':<10} {'Speedup':<10} {'Val Loss':<10}"
        )
        print("-" * 60)

        for result in results:
            speedup = (
                baseline["performance"]["train_time"]
                / result["performance"]["train_time"]
            )
            print(
                f"{result['name']:<30} "
                f"{result['performance']['train_time']:<10.1f} "
                f"{speedup:<10.2f}x "
                f"{result['quality']['best_val_loss']:<10.4f}"
            )

        # Dataset size analysis
        n_samples = results[0]["dataset"]["n_samples"]
        print(f"\n📊 Dataset Analysis:")
        print(f"   - Size: {n_samples} samples (small dataset)")
        print(f"   - GPU optimizations work best with >10k samples")
        print(f"   - Large batches may hurt convergence on small datasets")

        # Performance insights
        print(f"\n🚀 Performance Insights:")
        if has_gpu:
            print(f"   - Mixed precision can provide 2x speedup on compatible GPUs")
            print(f"   - Larger batches improve GPU utilization but may need tuning")
            print(f"   - Data pipeline optimization reduces CPU bottlenecks")
        else:
            print(f"   - No GPU detected - optimizations have limited effect")
            print(f"   - Consider using GPU for significant speedups")

        # Recommendations
        print(f"\n💡 Recommendations:")
        print(f"   1. For small datasets (<1k samples): use conservative batch sizes")
        print(f"   2. For large datasets (>10k samples): use aggressive GPU settings")
        print(
            f"   3. Monitor validation loss - adjust batch size if convergence suffers"
        )
        print(f"   4. Use mixed precision by default (now enabled in Locator)")

    def save_results(self, results: List[dict], output_path: str) -> None:
        """Save results to JSON file."""
        with open(output_path, "w") as f:
            json.dump(
                {
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "gpu_available": bool(tf.config.list_physical_devices("GPU")),
                    "tensorflow_version": tf.__version__,
                    "results": results,
                },
                f,
                indent=2,
            )
        print(f"\n📁 Results saved to {output_path}")


def main():
    """Main benchmark entry point."""
    parser = argparse.ArgumentParser(
        description="Benchmark GPU optimizations for Locator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--epochs", type=int, default=20, help="Number of epochs to train (default: 20)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="gpu_benchmark_results.json",
        help="Output file for results (default: gpu_benchmark_results.json)",
    )
    parser.add_argument(
        "--data", type=str, default="data", help="Path to data directory (default: data)"
    )

    args = parser.parse_args()

    # Run benchmark
    benchmark = GPUBenchmark(data_path=args.data, epochs=args.epochs)
    results = benchmark.run_all_benchmarks()

    # Print summary
    benchmark.print_summary(results)

    # Save results
    if args.output:
        benchmark.save_results(results, args.output)


if __name__ == "__main__":
    main()
