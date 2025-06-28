#!/usr/bin/env python3
"""
Quick GPU Optimization Benchmark

A focused benchmark comparing baseline vs GPU-optimized Locator performance.
Uses train_holdout pattern with the test data.

Usage:
    python quick_gpu_benchmark.py
"""

import time
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path

from locator import Locator


def run_single_benchmark(config_name: str, config: dict) -> dict:
    """Run a single benchmark and return metrics."""
    print(f"\n{'='*60}")
    print(f"Running {config_name}")
    print(f"{'='*60}")
    
    # Clear session
    tf.keras.backend.clear_session()
    
    # Create Locator instance
    loc = Locator(config)
    
    # Load data
    print("Loading data...")
    start_load = time.time()
    genotypes, samples = loc.load_genotypes(vcf="data/test_genotypes.vcf.gz")
    load_time = time.time() - start_load
    print(f"Data loaded in {load_time:.2f} seconds")
    
    # Print data info
    print(f"Genotypes shape: {genotypes.shape}")
    print(f"Number of samples: {len(samples)}")
    
    # Train with holdout
    print("\nTraining with 50 holdout samples...")
    start_train = time.time()
    
    # Capture more detailed timing
    history = loc.train_holdout(
        genotypes=genotypes,
        samples=samples,
        k=50  # Hold out 50 samples
    )
    
    train_time = time.time() - start_train
    
    # Get actual batch size used
    actual_batch_size = config.get('batch_size', 32)
    if 'gpu_batch_size' in config:
        if config['gpu_batch_size'] == 'auto':
            # Try to infer from training
            actual_batch_size = 'auto-optimized'
        else:
            actual_batch_size = config['gpu_batch_size']
    
    # Predict
    print("\nMaking predictions...")
    start_pred = time.time()
    try:
        predictions = loc.predict(return_df=True)
        pred_time = time.time() - start_pred
    except AttributeError as e:
        print(f"  Prediction error (expected with holdout): {e}")
        # For holdout, we need to check the stored predictions from training
        predictions = None
        pred_time = 0.0
    
    # Calculate metrics
    n_epochs = len(history.history['loss'])
    n_samples = len(samples) - 50  # Training samples
    total_samples = n_samples * n_epochs
    samples_per_sec = total_samples / train_time
    
    # Get final metrics
    final_loss = history.history['loss'][-1]
    best_val_loss = min(history.history['val_loss'])
    
    # Calculate prediction accuracy on holdout
    if predictions is not None:
        holdout_preds = predictions[predictions['prediction'] == True]
        if len(holdout_preds) > 0:
            errors = np.sqrt(
                (holdout_preds['true_x'] - holdout_preds['pred_x'])**2 + 
                (holdout_preds['true_y'] - holdout_preds['pred_y'])**2
            )
            mean_error = errors.mean()
            median_error = errors.median()
        else:
            mean_error = median_error = np.nan
    else:
        # For train_holdout, we can't easily get predictions
        mean_error = median_error = np.nan
    
    # Print results
    print(f"\nResults for {config_name}:")
    print(f"  Training time: {train_time:.2f} seconds")
    print(f"  Epochs trained: {n_epochs}")
    print(f"  Throughput: {samples_per_sec:.0f} samples/second")
    print(f"  Batch size: {actual_batch_size}")
    print(f"  Best validation loss: {best_val_loss:.4f}")
    print(f"  Mean holdout error: {mean_error:.2f}")
    print(f"  Prediction time: {pred_time:.2f} seconds")
    
    return {
        'config': config_name,
        'load_time': load_time,
        'train_time': train_time,
        'pred_time': pred_time,
        'total_time': load_time + train_time + pred_time,
        'epochs': n_epochs,
        'samples_per_sec': samples_per_sec,
        'batch_size': str(actual_batch_size),
        'final_loss': final_loss,
        'best_val_loss': best_val_loss,
        'mean_error': mean_error,
        'median_error': median_error
    }


def main():
    """Run the benchmark comparison."""
    print("GPU Optimization Benchmark")
    print("="*60)
    
    # Check GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"GPU detected: {gpus[0].name}")
        # Print GPU details
        try:
            from locator.gpu_optimizer import GPUOptimizer
            gpu_info = GPUOptimizer.get_gpu_info()
            for gpu in gpu_info['gpus']:
                print(f"  {gpu.get('name', 'Unknown GPU')}")
        except:
            pass
    else:
        print("WARNING: No GPU detected!")
    
    # Define configurations
    configs = {
        "Baseline (No GPU Opts)": {
            "out": "benchmark_baseline",
            "sample_data": "data/test_sample_data.txt",
            "max_epochs": 50,
            "patience": 10,
            "keras_verbose": 0,
            "na_action": "exclude",
            # Disable GPU optimizations
            "use_mixed_precision": False,
            "gpu_batch_size": 32,
            "use_efficient_pipeline": False
        },
        "GPU Optimized (Default)": {
            "out": "benchmark_optimized",
            "sample_data": "data/test_sample_data.txt",
            "max_epochs": 50,
            "patience": 10,
            "keras_verbose": 0,
            "na_action": "exclude",
            # GPU optimizations are default, but let's be explicit
            "use_mixed_precision": True,
            "gpu_batch_size": "auto",
            "use_efficient_pipeline": True
        }
    }
    
    # Run benchmarks
    results = []
    for config_name, config in configs.items():
        try:
            metrics = run_single_benchmark(config_name, config)
            results.append(metrics)
        except Exception as e:
            print(f"\nError running {config_name}: {e}")
            continue
    
    # Summary
    print("\n" + "="*60)
    print("BENCHMARK SUMMARY")
    print("="*60)
    
    if len(results) == 2:
        baseline = results[0]
        optimized = results[1]
        
        # Calculate speedups
        train_speedup = baseline['train_time'] / optimized['train_time']
        throughput_ratio = optimized['samples_per_sec'] / baseline['samples_per_sec']
        
        print(f"\nTraining Time:")
        print(f"  Baseline:  {baseline['train_time']:.2f} seconds")
        print(f"  Optimized: {optimized['train_time']:.2f} seconds")
        print(f"  Speedup:   {train_speedup:.2f}x faster")
        
        print(f"\nThroughput:")
        print(f"  Baseline:  {baseline['samples_per_sec']:.0f} samples/sec")
        print(f"  Optimized: {optimized['samples_per_sec']:.0f} samples/sec")
        print(f"  Improvement: {throughput_ratio:.2f}x")
        
        print(f"\nBatch Size:")
        print(f"  Baseline:  {baseline['batch_size']}")
        print(f"  Optimized: {optimized['batch_size']}")
        
        print(f"\nModel Quality:")
        print(f"  Baseline best val loss:  {baseline['best_val_loss']:.4f}")
        print(f"  Optimized best val loss: {optimized['best_val_loss']:.4f}")
        print(f"  Baseline holdout error:  {baseline['mean_error']:.2f}")
        print(f"  Optimized holdout error: {optimized['mean_error']:.2f}")
        
        # Save results
        results_df = pd.DataFrame(results)
        results_df.to_csv('gpu_benchmark_results.csv', index=False)
        print(f"\nDetailed results saved to gpu_benchmark_results.csv")
        
        # Overall assessment
        print("\n" + "="*60)
        print("CONCLUSION")
        print("="*60)
        if train_speedup > 1.2:
            print(f"✅ GPU optimizations provide {train_speedup:.1f}x speedup!")
            print(f"   This translates to {throughput_ratio:.1f}x higher throughput.")
        else:
            print("⚠️  Limited speedup observed. This could be due to:")
            print("   - Small dataset size (GPU optimizations work better with larger data)")
            print("   - No GPU available or limited GPU memory")
            print("   - CPU bottlenecks in data loading")
    
    else:
        print("Failed to complete both benchmarks")
        
    # Save all results
    if results:
        results_df = pd.DataFrame(results)
        results_df.to_csv('gpu_benchmark_results.csv', index=False)
        print(f"\nResults saved to gpu_benchmark_results.csv")


if __name__ == '__main__':
    main()