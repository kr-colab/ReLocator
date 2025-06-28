#!/usr/bin/env python3
"""
Benchmark GPU Optimizations vs Baseline

This script compares the performance of Locator with GPU optimizations
against the baseline (no GPU optimizations) using the train_holdout pattern.

Usage:
    python benchmark_gpu_optimizations.py [--n-runs 5] [--holdout-size 50]
"""

import argparse
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

from locator import Locator


class BenchmarkResults:
    """Store and analyze benchmark results."""
    
    def __init__(self):
        self.results = []
        
    def add_result(self, config_name: str, run_id: int, metrics: Dict):
        """Add a benchmark result."""
        self.results.append({
            'config': config_name,
            'run': run_id,
            **metrics
        })
        
    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to DataFrame."""
        return pd.DataFrame(self.results)
    
    def summarize(self) -> pd.DataFrame:
        """Summarize results by configuration."""
        df = self.to_dataframe()
        summary = df.groupby('config').agg({
            'training_time': ['mean', 'std', 'min', 'max'],
            'prediction_time': ['mean', 'std'],
            'samples_per_second': ['mean', 'std'],
            'final_loss': ['mean', 'std'],
            'best_val_loss': ['mean', 'std'],
            'epochs_trained': ['mean', 'std'],
            'batch_size_used': ['mean', 'min', 'max'],
            'gpu_memory_mb': ['mean', 'max'],
            'mae_holdout': ['mean', 'std'],
            'r2_longitude': ['mean', 'std'],
            'r2_latitude': ['mean', 'std']
        }).round(3)
        return summary


def get_gpu_memory_usage() -> float:
    """Get current GPU memory usage in MB."""
    try:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            info = tf.config.experimental.get_memory_info('GPU:0')
            return info['current'] / 1024 / 1024  # Convert to MB
    except:
        pass
    return 0.0


def calculate_metrics(true_coords: np.ndarray, pred_coords: np.ndarray) -> Dict:
    """Calculate prediction metrics."""
    from sklearn.metrics import r2_score, mean_absolute_error
    
    mae = mean_absolute_error(true_coords, pred_coords)
    r2_lon = r2_score(true_coords[:, 0], pred_coords[:, 0])
    r2_lat = r2_score(true_coords[:, 1], pred_coords[:, 1])
    
    return {
        'mae_holdout': mae,
        'r2_longitude': r2_lon,
        'r2_latitude': r2_lat
    }


def run_benchmark_iteration(config: Dict, data_path: Path, holdout_size: int, 
                          iteration: int) -> Dict:
    """Run a single benchmark iteration."""
    print(f"\n  Iteration {iteration + 1}")
    
    # Clear any previous session
    tf.keras.backend.clear_session()
    
    # Create Locator instance
    loc = Locator(config)
    
    # Load data
    print("    Loading data...")
    genotypes, samples = loc.load_genotypes(vcf=str(data_path / "test_genotypes.vcf.gz"))
    
    # Start timing
    start_time = time.time()
    
    # Train with holdout
    print(f"    Training with {holdout_size} holdout samples...")
    history = loc.train_holdout(
        genotypes=genotypes,
        samples=samples,
        k=holdout_size
    )
    
    training_time = time.time() - start_time
    
    # Get batch size actually used
    if hasattr(loc, 'config'):
        if loc.config.get('gpu_batch_size') == 'auto':
            # Extract from training logs if possible
            batch_size_used = loc.config.get('batch_size', 32)  # Fallback
            # Try to get actual optimized batch size from model training
            if hasattr(loc, '_last_batch_size'):
                batch_size_used = loc._last_batch_size
        else:
            batch_size_used = loc.config.get('gpu_batch_size', loc.config.get('batch_size', 32))
    else:
        batch_size_used = 32
    
    # Predict on holdout samples
    print("    Predicting on holdout samples...")
    pred_start = time.time()
    predictions = loc.predict(return_df=True)
    prediction_time = time.time() - pred_start
    
    # Calculate samples per second
    n_samples = len(samples)
    epochs = len(history.history['loss'])
    total_samples_processed = n_samples * epochs
    samples_per_second = total_samples_processed / training_time
    
    # Get GPU memory usage
    gpu_memory = get_gpu_memory_usage()
    
    # Get training metrics
    final_loss = history.history['loss'][-1]
    best_val_loss = min(history.history['val_loss'])
    
    # Calculate prediction accuracy on holdout samples
    # Find holdout samples in predictions
    holdout_preds = predictions[predictions['prediction'] == True]
    if len(holdout_preds) > 0:
        true_coords = holdout_preds[['true_x', 'true_y']].values
        pred_coords = holdout_preds[['pred_x', 'pred_y']].values
        accuracy_metrics = calculate_metrics(true_coords, pred_coords)
    else:
        accuracy_metrics = {
            'mae_holdout': np.nan,
            'r2_longitude': np.nan,
            'r2_latitude': np.nan
        }
    
    # Compile results
    results = {
        'training_time': training_time,
        'prediction_time': prediction_time,
        'samples_per_second': samples_per_second,
        'final_loss': final_loss,
        'best_val_loss': best_val_loss,
        'epochs_trained': epochs,
        'batch_size_used': batch_size_used,
        'gpu_memory_mb': gpu_memory,
        **accuracy_metrics
    }
    
    print(f"    Training time: {training_time:.2f}s")
    print(f"    Throughput: {samples_per_second:.0f} samples/second")
    print(f"    Batch size: {batch_size_used}")
    print(f"    Best val loss: {best_val_loss:.4f}")
    
    return results


def create_benchmark_configs() -> Dict[str, Dict]:
    """Create configurations for benchmarking."""
    
    # Base configuration
    base_config = {
        "out": "benchmark",
        "sample_data": "data/test_sample_data.txt",
        "max_epochs": 100,  # Limit for benchmarking
        "patience": 20,
        "keras_verbose": 0,  # Quiet output
        "na_action": "exclude"  # Exclude NA samples for cleaner benchmark
    }
    
    # Configuration 1: Baseline (no GPU optimizations)
    baseline_config = base_config.copy()
    baseline_config.update({
        "out": "benchmark_baseline",
        "use_mixed_precision": False,
        "gpu_batch_size": 32,  # Fixed small batch
        "use_efficient_pipeline": False
    })
    
    # Configuration 2: GPU optimized (default in new version)
    optimized_config = base_config.copy()
    optimized_config.update({
        "out": "benchmark_optimized",
        "use_mixed_precision": True,
        "gpu_batch_size": "auto",
        "use_efficient_pipeline": True
    })
    
    # Configuration 3: GPU optimized with larger fixed batch
    large_batch_config = base_config.copy()
    large_batch_config.update({
        "out": "benchmark_large_batch",
        "use_mixed_precision": True,
        "gpu_batch_size": 256,  # Fixed large batch
        "use_efficient_pipeline": True
    })
    
    # Configuration 4: GPU optimized with XLA
    xla_config = base_config.copy()
    xla_config.update({
        "out": "benchmark_xla",
        "use_mixed_precision": True,
        "gpu_batch_size": "auto",
        "use_efficient_pipeline": True,
        "enable_xla": True
    })
    
    return {
        "Baseline": baseline_config,
        "GPU Optimized": optimized_config,
        "Large Batch": large_batch_config,
        "GPU + XLA": xla_config
    }


def plot_results(results: BenchmarkResults, output_dir: Path):
    """Create visualization of benchmark results."""
    df = results.to_dataframe()
    
    # Set up the plot style
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('GPU Optimization Benchmark Results', fontsize=16)
    
    # 1. Training time comparison
    ax = axes[0, 0]
    sns.boxplot(data=df, x='config', y='training_time', ax=ax)
    ax.set_title('Training Time')
    ax.set_ylabel('Time (seconds)')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    
    # 2. Throughput comparison
    ax = axes[0, 1]
    sns.boxplot(data=df, x='config', y='samples_per_second', ax=ax)
    ax.set_title('Training Throughput')
    ax.set_ylabel('Samples/second')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    
    # 3. Batch size used
    ax = axes[0, 2]
    sns.boxplot(data=df, x='config', y='batch_size_used', ax=ax)
    ax.set_title('Batch Size Used')
    ax.set_ylabel('Batch size')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    
    # 4. Best validation loss
    ax = axes[1, 0]
    sns.boxplot(data=df, x='config', y='best_val_loss', ax=ax)
    ax.set_title('Best Validation Loss')
    ax.set_ylabel('Loss')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    
    # 5. Holdout MAE
    ax = axes[1, 1]
    sns.boxplot(data=df, x='config', y='mae_holdout', ax=ax)
    ax.set_title('Holdout Prediction MAE')
    ax.set_ylabel('Mean Absolute Error')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    
    # 6. GPU Memory Usage
    ax = axes[1, 2]
    sns.boxplot(data=df, x='config', y='gpu_memory_mb', ax=ax)
    ax.set_title('GPU Memory Usage')
    ax.set_ylabel('Memory (MB)')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'benchmark_results.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create speedup comparison plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Calculate speedup relative to baseline
    baseline_time = df[df['config'] == 'Baseline']['training_time'].mean()
    speedup_data = []
    for config in df['config'].unique():
        if config != 'Baseline':
            config_time = df[df['config'] == config]['training_time'].mean()
            speedup = baseline_time / config_time
            speedup_data.append({'config': config, 'speedup': speedup})
    
    speedup_df = pd.DataFrame(speedup_data)
    sns.barplot(data=speedup_df, x='config', y='speedup', ax=ax)
    ax.axhline(y=1.0, color='red', linestyle='--', label='Baseline')
    ax.set_title('Speedup vs Baseline')
    ax.set_ylabel('Speedup Factor')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    
    # Add value labels on bars
    for i, row in speedup_df.iterrows():
        ax.text(i, row['speedup'] + 0.05, f'{row["speedup"]:.2f}x', 
                ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'speedup_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()


def main():
    """Run the benchmark."""
    parser = argparse.ArgumentParser(description='Benchmark GPU optimizations')
    parser.add_argument('--n-runs', type=int, default=5,
                       help='Number of benchmark runs per configuration')
    parser.add_argument('--holdout-size', type=int, default=50,
                       help='Number of samples to hold out')
    parser.add_argument('--output-dir', type=str, default='benchmark_results',
                       help='Directory for output files')
    args = parser.parse_args()
    
    # Setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    data_path = Path('data')
    
    # Check GPU availability
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"GPU detected: {len(gpus)} device(s)")
        for gpu in gpus:
            print(f"  {gpu}")
    else:
        print("WARNING: No GPU detected. Results may not show GPU benefits.")
    
    # Get configurations
    configs = create_benchmark_configs()
    
    # Run benchmarks
    results = BenchmarkResults()
    
    print(f"\nRunning {args.n_runs} iterations for each configuration...")
    print(f"Holdout size: {args.holdout_size} samples")
    print("="*60)
    
    for config_name, config in configs.items():
        print(f"\nBenchmarking: {config_name}")
        print("-"*40)
        
        for run in range(args.n_runs):
            try:
                metrics = run_benchmark_iteration(
                    config, data_path, args.holdout_size, run
                )
                results.add_result(config_name, run, metrics)
            except Exception as e:
                print(f"  Error in run {run + 1}: {e}")
                continue
    
    # Save results
    print("\n" + "="*60)
    print("BENCHMARK COMPLETE")
    print("="*60)
    
    # Save raw results
    results_df = results.to_dataframe()
    results_df.to_csv(output_dir / 'benchmark_raw_results.csv', index=False)
    
    # Save summary
    summary = results.summarize()
    summary.to_csv(output_dir / 'benchmark_summary.csv')
    
    # Print summary
    print("\nSUMMARY STATISTICS:")
    print(summary)
    
    # Calculate and print speedups
    print("\nSPEEDUP vs BASELINE:")
    baseline_time = results_df[results_df['config'] == 'Baseline']['training_time'].mean()
    for config in results_df['config'].unique():
        if config != 'Baseline':
            config_time = results_df[results_df['config'] == config]['training_time'].mean()
            speedup = baseline_time / config_time
            print(f"  {config}: {speedup:.2f}x faster")
    
    # Create plots
    print("\nGenerating plots...")
    plot_results(results, output_dir)
    
    print(f"\nResults saved to {output_dir}/")
    print("  - benchmark_raw_results.csv: Raw results from all runs")
    print("  - benchmark_summary.csv: Summary statistics")
    print("  - benchmark_results.png: Visualization of results")
    print("  - speedup_comparison.png: Speedup comparison")
    
    # Save configuration details
    with open(output_dir / 'benchmark_configs.json', 'w') as f:
        json.dump(configs, f, indent=2)


if __name__ == '__main__':
    main()