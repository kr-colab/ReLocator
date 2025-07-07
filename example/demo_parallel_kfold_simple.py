#!/usr/bin/env python
"""
Simple demonstration of parallel k-fold holdout analysis using test data.

This script:
1. Loads genotype data from VCF and sample metadata
2. Runs parallel k-fold cross-validation
3. Generates error summary plots from the predictions

All output is saved to a directory named "demo_output".
"""

import os

# Import locator modules
from locator import Locator
from locator.parallel import parallel_k_fold_holdouts
from locator.plotting import plot_error_summary


def main():

    vcf_path = "data/test_genotypes.vcf.gz"
    coords_path = "data/test_sample_data.txt"
    output_dir = "demo_output"

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Configuration for Locator
    config = {
        "out": output_dir,
        "sample_data": coords_path,
        "vcf": vcf_path,
        "batch_size": 32,
        "width": 256,  # Number of units in hidden layers
        "nlayers": 8,  # Number of hidden layers
        "dropout_prop": 0.25,
        "max_epochs": 500,
        "train_split": 0.8,
        "patience": 100,  # Early stopping patience
        "keras_verbose": 0,  # Suppress keras output since verbose=False in k-fold
        "verbose_splits": True,
        "holdout_no_intermediate_saves": True,
    }

    # Create Locator instance
    locator = Locator(config)

    # Load genotype data
    print("\nLoading genotype data from VCF...")
    genotypes, samples = locator.load_genotypes(vcf=vcf_path)
    print(f"Loaded genotypes shape: {genotypes.shape}")
    print(f"Number of samples: {len(samples)}")
    print(f"Number of SNPs: {genotypes.shape[0]}")

    locator.check_data(genotypes, samples)

    # For demo, we'll use CPU to ensure it works everywhere
    gpu_ids = []  # Empty list = CPU only
    print("\nRunning demo in CPU mode for compatibility")

    # Run parallel k-fold cross-validation
    k = 3  # 3-fold for faster demo
    print(f"\nRunning parallel {k}-fold cross-validation...")

    try:
        predictions = parallel_k_fold_holdouts(
            locator=locator,
            genotypes=genotypes,
            samples=samples,
            k=k,
            gpu_ids=gpu_ids,  # CPU only for demo
            gpu_fraction=0.0,  # CPU mode
            return_df=True,
            verbose=True,
            save_full_pred_matrix=False,  # we will save this on our own.
        )

        print(f"\nPredictions completed!")
        print(f"Predictions shape: {predictions.shape}")

        # Save raw predictions
        pred_file = os.path.join(output_dir, "kfold_predictions_raw.csv")
        predictions.to_csv(pred_file, index=False)
        print(f"Saved raw predictions to: {pred_file}")

        print("\nGenerating error summary plot...")

        plot_error_summary(
            predictions=predictions,
            sample_data=coords_path,
            plot_map=False,
            include_training_locs=True,
            show=False,  # Save only
            out_prefix=os.path.join(output_dir, "kfold"),
        )

        print(f"Error summary plot saved to: {output_dir}/kfold_error_summary.png")

    except Exception as e:
        print(f"\nError during analysis: {e}")
        import traceback

        traceback.print_exc()

    finally:
        # Cleanup Ray if initialized
        try:
            import ray

            if ray.is_initialized():
                ray.shutdown()
                print("\nRay shutdown complete")
        except:
            pass

    print("\nDemo complete!")


if __name__ == "__main__":
    main()
