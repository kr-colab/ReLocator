#!/usr/bin/env python3
"""Test script for ensemble refactoring - Phase 1"""

import allel
import numpy as np
import pandas as pd

# Create synthetic test data
n_samples = 100
n_snps = 500

# Generate random genotypes
np.random.seed(42)
genotype_array = np.random.randint(0, 3, size=(n_snps, n_samples))
genotypes = allel.GenotypeArray(genotype_array[:, :, np.newaxis])

# Generate sample IDs
samples = np.array([f"sample_{i:03d}" for i in range(n_samples)])

# Generate sample locations (with some NA values)
coords_df = pd.DataFrame(
    {
        "sampleID": samples,
        "x": np.random.uniform(-120, -70, n_samples),
        "y": np.random.uniform(30, 50, n_samples),
    }
)

# Set some samples to have NA coordinates
na_indices = np.random.choice(n_samples, size=10, replace=False)
coords_df.loc[coords_df.index.isin(na_indices), ["x", "y"]] = np.nan

# Configuration
config = {
    "out": "test_ensemble",
    "sample_data": coords_df,
    "max_epochs": 10,  # Quick test
    "keras_verbose": 0,
    "patience": 5,
    "batch_size": 32,
    "na_action": "separate",
}

print("Testing new ensemble implementation...")
print(f"Samples: {n_samples} (10 with NA coordinates)")
print(f"SNPs: {n_snps}")

# Test 1: Test IndexSet k_fold_split
print("\n1. Testing IndexSet.k_fold_split...")
from locator.data import IndexSet

# Create NA mask
locs = coords_df[["x", "y"]].values
na_mask = np.isnan(locs[:, 0]) | np.isnan(locs[:, 1])

# Create k-fold splits
fold_index_sets = IndexSet.k_fold_split(n=n_samples, k=5, seed=42, na_mask=na_mask)

print(f"Created {len(fold_index_sets)} folds")
for i, index_set in enumerate(fold_index_sets):
    sizes = index_set.split_sizes()
    print(f"  Fold {i}: train={sizes['train']}, test={sizes['test']}")

# Test 2: Test ensemble mixin methods on Locator
print("\n2. Testing Locator ensemble methods...")
from locator import Locator

locator = Locator(config)

# Test create_ensemble_folds
fold_info = locator.create_ensemble_folds(
    genotypes=genotypes, samples=samples, k=5, na_action="separate"
)

print(f"Fold info keys: {list(fold_info.keys())}")
print(
    f"Sample status: {fold_info['sample_status']['n_known']} known, "
    f"{fold_info['sample_status']['n_na']} NA"
)

# Test 3: Test legacy EnsembleLocator wrapper
print("\n3. Testing legacy EnsembleLocator API...")
from locator import EnsembleLocator

# This should show deprecation warning
ensemble = EnsembleLocator(config, k_folds=3)

# Test create_folds
fold_indices = ensemble.create_folds(genotypes, samples, locs)
print(f"Legacy fold indices created: {len(fold_indices)} folds")

print("\nPhase 1 implementation test complete!")
print("\nNote: Full training test skipped for speed. To test training:")
print("  result = locator.train_ensemble(genotypes, samples, k=3)")
print("  predictions = locator.predict_ensemble(genotypes, samples)")
