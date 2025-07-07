#!/usr/bin/env python
"""
Demonstration of parallel ensemble training using test data.

This script:
1. Loads genotype data from VCF and sample metadata
2. Runs sequential ensemble training for comparison
3. Runs parallel ensemble training across multiple GPUs
4. Compares results and performance
5. Makes predictions using the trained ensemble

All output is saved to a directory named "demo_output".
"""

import os
import time

# Silence TensorFlow and CUDA warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Only show errors
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # Disable oneDNN custom operations
os.environ["XLA_FLAGS"] = (
    "--xla_gpu_cuda_data_dir=/usr/local/cuda"  # Suppress XLA warnings
)

# Suppress other warnings
import warnings

warnings.filterwarnings("ignore")

# Also suppress Ray's verbose output
os.environ["RAY_DEDUP_LOGS"] = "1"
os.environ["RAY_verbose_spill_logs"] = "0"

# Import locator modules
from locator import Locator
from locator.parallel import parallel_train_ensemble
from locator.plotting import plot_predictions


def train_sequential_ensemble(locator_seq, genotypes, samples, k_folds, output_dir):
    """Train ensemble sequentially and return results."""
    print("1. SEQUENTIAL ENSEMBLE TRAINING")
    print("-" * 30)

    print(f"\nTraining {k_folds}-fold ensemble sequentially...")
    start_time = time.time()

    seq_result = locator_seq.train_ensemble(
        genotypes=genotypes,
        samples=samples,
        k=k_folds,
        save_fold_models=True,
        use_model_manager=True,
        verbose=True,
    )

    seq_time = time.time() - start_time
    print(f"\nSequential training completed in {seq_time:.1f} seconds")
    print(f"Average time per fold: {seq_time/k_folds:.1f} seconds")

    # Get averaged normalization parameters
    seq_norm = seq_result["normalization_params"]
    print("\nNormalization parameters:")
    print(f"  Mean longitude: {seq_norm['meanlong']:.2f}")
    print(f"  Mean latitude: {seq_norm['meanlat']:.2f}")

    return seq_result, seq_time, seq_norm


def train_parallel_ensemble(
    locator_par, genotypes, samples, k_folds, gpu_ids, output_dir
):
    """Train ensemble in parallel and return results."""
    print("\n\n2. PARALLEL ENSEMBLE TRAINING")
    print("-" * 30)

    try:
        print(f"\nTraining {k_folds}-fold ensemble in parallel...")

        start_time = time.time()
        par_result = parallel_train_ensemble(
            locator=locator_par,
            genotypes=genotypes,
            samples=samples,
            k=k_folds,
            gpu_ids=gpu_ids,
            save_fold_models=True,
            use_model_manager=True,
            verbose=True,
        )
        par_time = time.time() - start_time

        print(f"\nParallel training completed in {par_time:.1f} seconds")
        print(f"Average time per fold: {par_time/k_folds:.1f} seconds")

        # Get averaged normalization parameters
        par_norm = par_result["normalization_params"]
        print("\nNormalization parameters:")
        print(f"  Mean longitude: {par_norm['meanlong']:.2f}")
        print(f"  Mean latitude: {par_norm['meanlat']:.2f}")

        return par_result, par_time, par_norm

    except Exception as e:
        print(f"\nError during parallel training: {e}")
        import traceback

        traceback.print_exc()
        return None, 0, None


def make_ensemble_predictions(locator_seq, genotypes, samples, output_dir):
    """Make predictions using the trained ensemble."""
    print("\n\n3. ENSEMBLE PREDICTIONS")
    print("-" * 30)

    # Check if we have samples without coordinates
    sample_status = locator_seq.get_sample_status(samples)

    predictions = locator_seq.predict_ensemble(
        genotypes=genotypes,
        samples=samples,
        include_fold_predictions=True,
        return_std=True,
        save_predictions=False,
    )

    if sample_status["n_na"] > 0:
        print(f"\nFound {sample_status['n_na']} samples without coordinates")
        print("Making predictions for these samples...")

        # Show predictions for NA samples
        na_predictions = predictions[
            predictions["sampleID"].isin(sample_status["na_samples"])
        ]
        print("\nPredictions for samples without coordinates:")
        print(na_predictions[["sampleID", "x", "y", "x_std", "y_std"]])
    else:
        print("\nAll samples have coordinates. Making predictions for all samples...")
        print(f"\nGenerated predictions for {len(predictions)} samples")
        print("Mean prediction uncertainty (std dev):")
        print(f"  Longitude: {predictions['x_std'].mean():.2f}")
        print(f"  Latitude: {predictions['y_std'].mean():.2f}")

        # Create prediction plot
        print("\nGenerating prediction plot...")
        plot_predictions(
            predictions=predictions,
            locator=locator_seq,
            out_prefix=os.path.join(output_dir, "ensemble_predictions"),
            plot_border=True,
        )
        print(f"Prediction plot saved to: {output_dir}/ensemble_predictions.png")

    # Save predictions
    pred_file = os.path.join(output_dir, "ensemble_predictions.csv")
    predictions.to_csv(pred_file, index=False)
    print(f"\nSaved predictions to: {pred_file}")

    return predictions


def cleanup_resources(k_folds, output_dir):
    """Clean up Ray and model files."""
    # Cleanup Ray if initialized
    try:
        import ray

        if ray.is_initialized():
            ray.shutdown()
            print("Ray shutdown complete")
    except Exception:
        pass

    # Clean up model files if requested
    if os.environ.get("CLEANUP_MODELS", "0") == "1":
        print("\nCleaning up model files...")
        for i in range(k_folds):
            for prefix in ["ensemble", "ensemble_parallel"]:
                weights_file = os.path.join(output_dir, f"{prefix}_fold{i}.weights.h5")
                if os.path.exists(weights_file):
                    os.remove(weights_file)
        print("Model files cleaned up")


def main():
    vcf_path = "data/test_genotypes.vcf.gz"
    coords_path = "data/test_sample_data.txt"
    output_dir = "demo_output"

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Configuration for Locator
    config = {
        "out": os.path.join(output_dir, "ensemble"),
        "sample_data": coords_path,
        "vcf": vcf_path,
        "batch_size": 32,
        "width": 256,  # Number of units in hidden layers
        "nlayers": 8,  # Number of hidden layers
        "dropout_prop": 0.25,
        "max_epochs": 100,  # Reduced for demo
        "train_split": 0.8,
        "patience": 20,  # Early stopping patience
        "keras_verbose": 0,  # Suppress keras output
        "verbose_splits": True,
    }

    # Number of folds for ensemble
    k_folds = 3  # 3-fold for faster demo

    print("PARALLEL ENSEMBLE TRAINING DEMONSTRATION")
    print("=" * 60)
    print(f"Using data from: {vcf_path}")
    print(f"Sample metadata: {coords_path}")
    print(f"Output directory: {output_dir}")
    print(f"Ensemble size: {k_folds} models")
    print("=" * 60 + "\n")

    # For demo, we'll use CPU to ensure it works everywhere
    gpu_ids = []  # Empty list = CPU only
    print("Running demo in CPU mode for compatibility")
    print("(To use GPUs, modify gpu_ids list, e.g., gpu_ids = [0, 1])")
    print("(To see detailed TensorFlow output, set TF_CPP_MIN_LOG_LEVEL=0)\n")

    # SEQUENTIAL ENSEMBLE TRAINING
    print("1. SEQUENTIAL ENSEMBLE TRAINING")
    print("-" * 30)

    # Create Locator instance for sequential training
    locator_seq = Locator(config.copy())

    # Load genotype data
    print("Loading genotype data from VCF...")
    genotypes, samples = locator_seq.load_genotypes(vcf=vcf_path)
    print(f"Loaded genotypes shape: {genotypes.shape}")
    print(f"Number of samples: {len(samples)}")
    print(f"Number of SNPs: {genotypes.shape[0]}")

    locator_seq.check_data(genotypes, samples)

    # Train sequential ensemble
    seq_result, seq_time, seq_norm = train_sequential_ensemble(
        locator_seq, genotypes, samples, k_folds, output_dir
    )

    # Create Locator instance for parallel training
    locator_par = Locator(config.copy())
    locator_par.config["out"] = os.path.join(output_dir, "ensemble_parallel")

    # Train parallel ensemble
    par_result, par_time, par_norm = train_parallel_ensemble(
        locator_par, genotypes, samples, k_folds, gpu_ids, output_dir
    )

    # Print training summary
    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    print(f"Sequential time: {seq_time:.1f}s")

    if par_result is not None:
        print(f"Parallel time: {par_time:.1f}s")

        if par_time < seq_time:
            speedup = seq_time / par_time
            print(f"Speedup: {speedup:.2f}x")
        else:
            print("Note: CPU mode may not show speedup benefits")

        # Verify results are similar
        norm_diff_long = abs(seq_norm["meanlong"] - par_norm["meanlong"])
        norm_diff_lat = abs(seq_norm["meanlat"] - par_norm["meanlat"])
        print("\nResult consistency check:")
        print(f"  Longitude difference: {norm_diff_long:.6f}")
        print(f"  Latitude difference: {norm_diff_lat:.6f}")

        if norm_diff_long < 0.01 and norm_diff_lat < 0.01:
            print("  ✓ Results are consistent between sequential and parallel")
        else:
            print("  ⚠ Results differ between sequential and parallel")

    # Make predictions using the ensemble
    _ = make_ensemble_predictions(locator_seq, genotypes, samples, output_dir)

    # Cleanup
    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)

    cleanup_resources(k_folds, output_dir)


if __name__ == "__main__":
    main()
