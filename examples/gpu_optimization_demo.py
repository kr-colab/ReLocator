"""
GPU Optimization Demo for Locator

This script demonstrates the GPU optimization features implemented in Locator.
"""

import time

import numpy as np
import pandas as pd
import tensorflow as tf

from locator import Locator
from locator.gpu_optimizer import GPUOptimizer


def create_demo_data(n_samples=1000, n_snps=5000):
    """Create synthetic data for demonstration."""
    # Create genotype data (0, 1, 2)
    genotypes = np.random.randint(0, 3, size=(n_snps, n_samples))

    # Create sample IDs
    samples = np.array([f"sample_{i}" for i in range(n_samples)])

    # Create location data
    sample_data = pd.DataFrame(
        {
            "sampleID": samples,
            "x": np.random.uniform(-180, 180, n_samples),  # longitude
            "y": np.random.uniform(-90, 90, n_samples),  # latitude
        }
    )

    return genotypes, samples, sample_data


def compare_configurations():
    """Compare different GPU optimization configurations."""

    # Create demo data
    print("Creating demo data...")
    genotypes, samples, sample_data = create_demo_data()

    # Configuration 1: Default (no GPU optimizations)
    config_default = {
        "out": "demo_default",
        "sample_data": sample_data,
        "max_epochs": 10,  # Short demo
        "keras_verbose": 0,
        "use_mixed_precision": False,
        "use_efficient_pipeline": False,
        "gpu_batch_size": 32,  # Default small batch
    }

    # Configuration 2: GPU optimized
    config_optimized = {
        "out": "demo_optimized",
        "sample_data": sample_data,
        "max_epochs": 10,  # Short demo
        "keras_verbose": 0,
        "use_mixed_precision": True,
        "use_efficient_pipeline": True,
        "gpu_batch_size": "auto",  # Dynamic batch size
    }

    # Print GPU info
    print("\nGPU Information:")
    gpu_info = GPUOptimizer.get_gpu_info()
    print(f"Number of GPUs: {gpu_info['gpu_count']}")
    for gpu in gpu_info["gpus"]:
        print(f"  GPU {gpu['index']}: {gpu.get('name', 'Unknown')}")

    # Test default configuration
    print("\n" + "=" * 60)
    print("Testing DEFAULT configuration (no GPU optimizations)")
    print("=" * 60)

    loc_default = Locator(config_default)
    start_time = time.time()
    loc_default.train(genotypes=genotypes, samples=samples)
    default_time = time.time() - start_time

    print(f"\nTraining time: {default_time:.2f} seconds")

    # Clear session to free memory
    tf.keras.backend.clear_session()

    # Test optimized configuration
    print("\n" + "=" * 60)
    print("Testing OPTIMIZED configuration")
    print("=" * 60)

    loc_optimized = Locator(config_optimized)

    # Show optimizations applied
    if loc_optimized.config.get("use_mixed_precision"):
        print("✓ Mixed precision training enabled")
    if loc_optimized.config.get("use_efficient_pipeline"):
        print("✓ Efficient data pipeline enabled")
    if loc_optimized.config.get("gpu_batch_size") == "auto":
        print("✓ Dynamic batch size optimization enabled")

    start_time = time.time()
    loc_optimized.train(genotypes=genotypes, samples=samples)
    optimized_time = time.time() - start_time

    print(f"\nTraining time: {optimized_time:.2f} seconds")

    # Compare results
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"Default configuration time: {default_time:.2f} seconds")
    print(f"Optimized configuration time: {optimized_time:.2f} seconds")
    speedup = default_time / optimized_time
    print(f"Speedup: {speedup:.2f}x")

    if speedup > 1:
        print(f"\n🚀 GPU optimizations provided {speedup:.1f}x speedup!")
    else:
        print("\n⚠️  GPU optimizations did not provide speedup. This could be due to:")
        print("   - Small dataset size (GPU optimizations work better with larger data)")
        print("   - No GPU available (running on CPU)")
        print("   - GPU memory constraints")


def demonstrate_batch_size_optimization():
    """Demonstrate automatic batch size optimization."""
    print("\n" + "=" * 60)
    print("BATCH SIZE OPTIMIZATION DEMO")
    print("=" * 60)

    # Create a simple model for testing
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(1000,)),
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.Dense(2),
        ]
    )

    # Find optimal batch size
    optimal_batch = GPUOptimizer.get_optimal_batch_size(
        model, input_shape=(1000,), target_memory_usage=0.85
    )

    print(f"Optimal batch size for your GPU: {optimal_batch}")
    print(f"(Default batch size: 32)")
    print(f"Improvement factor: {optimal_batch/32:.1f}x larger batches")


if __name__ == "__main__":
    print("Locator GPU Optimization Demo")
    print("=" * 60)

    # Check if GPU is available
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        print(f"✓ GPU detected: {len(gpus)} device(s)")
    else:
        print("⚠️  No GPU detected. Running on CPU.")
        print("   GPU optimizations will have limited effect.")

    # Run comparison
    compare_configurations()

    # Demonstrate batch size optimization if GPU available
    if gpus:
        demonstrate_batch_size_optimization()

    print("\nDemo complete!")
