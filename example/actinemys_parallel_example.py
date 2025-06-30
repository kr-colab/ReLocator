#!/usr/bin/env python3
"""
Example of using parallel k-fold with Actinemys data.

This shows how to properly use multiple GPUs when running from a script
(not from a notebook with pre-set CUDA_VISIBLE_DEVICES).
"""

import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from locator import Locator
from locator.parallel import simple_parallel_k_fold, simple_parallel_leave_one_out


def main():
    # Paths
    vcf_path = "/sietch_colab/data_share/turtles_Actinemys/58-Actinemys/QC/58-Actinemys.pruned.vcf.gz"
    coords_path = "/sietch_colab/data_share/turtles_Actinemys/actinemys_locator_metadata.tsv"
    output_dir = "/sietch_colab/data_share/turtles_Actinemys/locator_output_parallel"

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Configuration for Locator
    config = {
        "out": os.path.join(output_dir, "actinemys_parallel"),
        "sample_data": coords_path,
        "vcf": vcf_path,
        "batch_size": 32,
        "width": 256,
        "nlayers": 8,
        "dropout_prop": 0.25,
        "max_epochs": 500,
        "train_split": 0.8,
        "patience": 100,
        "keras_verbose": 0,
        "weight_samples": {
            "enabled": True,
            "method": "KD",
            "xbins": 30,
            "ybins": 30,
        },
        "disable_gpu": False,  # Will be handled by parallel functions
        "gpu_number": 0,  # Will be overridden in workers
        "verbose_splits": False,  # Less verbose for parallel
        "holdout_no_intermediate_saves": True,
    }

    # Create Locator instance
    print("Creating Locator instance...")
    locator = Locator(config)

    # Load genotype data
    print("\nLoading genotype data from VCF...")
    genotypes, samples = locator.load_genotypes(vcf=vcf_path)
    print(f"Loaded genotypes shape: {genotypes.shape}")
    print(f"Number of samples: {len(samples)}")
    print(f"Number of SNPs: {genotypes.shape[0]}")

    # # Example 1: Parallel k-fold cross-validation
    # print("\n" + "="*60)
    # print("Running parallel 10-fold cross-validation on GPUs 0, 1, 2...")
    # print("="*60)
    
    # kfold_results = simple_parallel_k_fold(
    #     locator=locator,
    #     genotypes=genotypes,
    #     samples=samples,
    #     k=10,
    #     gpu_ids=[0, 1, 2],  # Use GPUs 0, 1, and 2
    #     verbose=True
    # )
    
    # print(f"\nK-fold results shape: {kfold_results.shape}")
    # kfold_results.to_csv(os.path.join(output_dir, "kfold_results.csv"), index=False)
    
    # Example 2: Parallel leave-one-out (on smaller subset for speed)
    print("\n" + "="*60)
    print("Running parallel leave-one-out on first 50 samples...")
    print("="*60)
    
    subset_size = 50
    loo_results = simple_parallel_leave_one_out(
        locator=locator,
        genotypes=genotypes[:, :subset_size],
        samples=samples[:subset_size],
        gpu_ids=[1, 2],  # Use GPUs 0, 1, and 2
        verbose=True
    )
    
    print(f"\nLOO results shape: {loo_results.shape}")
    loo_results.to_csv(os.path.join(output_dir, "loo_results_subset.csv"), index=False)
    
    print("\nDone!")


if __name__ == "__main__":
    # DO NOT set CUDA_VISIBLE_DEVICES here - let the parallel functions handle it
    main()