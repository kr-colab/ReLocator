"""Test that parallel methods correctly handle sample exclusion."""

import os

import allel
import numpy as np
import pandas as pd

# Set Ray to be quiet
os.environ["RAY_DEDUP_LOGS"] = "1"
os.environ["RAY_verbose_spill_logs"] = "0"

from locator import Locator
from locator.parallel import parallel_k_fold_holdouts


def test_parallel_kfold_with_exclusion():
    """Test that parallel k-fold handles excluded samples correctly."""
    # Create test data
    np.random.seed(123)
    n_snps = 500
    n_samples = 30

    # Create genotypes with good variation
    geno_array = np.zeros((n_snps, n_samples, 2), dtype=int)
    for i in range(n_snps):
        p = np.random.uniform(0.2, 0.4)
        geno_array[i, :, :] = np.random.binomial(1, p, size=(n_samples, 2))

    genotypes = allel.GenotypeArray(geno_array)
    samples = np.array([f"sample_{i:03d}" for i in range(n_samples)])

    # Create sample data with specific locations for verification
    sample_data = pd.DataFrame(
        {
            "sampleID": samples,
            "x": np.arange(n_samples) * 1.0,  # Unique x values for each sample
            "y": np.arange(n_samples) * 2.0,  # Unique y values for each sample
        }
    )

    # Initialize with exclusions
    exclude_list = ["sample_005", "sample_015", "sample_025"]
    locator = Locator(
        {
            "out": "test_parallel_exclusion",
            "sample_data": sample_data,
            "exclude_samples": exclude_list,
            "epochs": 2,
            "verbose": 0,
            "min_mac": 2,
        }
    )

    print("Testing parallel k-fold with exclusions...")
    try:
        # Run parallel k-fold
        results = parallel_k_fold_holdouts(
            locator=locator,
            genotypes=genotypes,
            samples=samples,
            k=3,
            gpu_ids=[0],  # Use single GPU
            return_df=True,
            verbose=True,
        )

        print(f"SUCCESS: Got predictions for {len(results)} samples")

        # Verify excluded samples not in results
        for excluded in exclude_list:
            assert excluded not in results["sampleID"].values
        print("Verified: Excluded samples not in predictions")

        # Verify we have predictions for all non-excluded samples
        expected_count = n_samples - len(exclude_list)
        assert len(results["sampleID"].unique()) == expected_count
        print(f"Verified: Got predictions for all {expected_count} non-excluded samples")

        # Verify the data alignment is correct by checking a few predictions
        # The model should learn some correlation between sample index and location
        # If misaligned, predictions would be random
        sample_check = results[results["sampleID"] == "sample_000"].iloc[0]
        print(
            f"Sample 000 prediction: x={sample_check['x_pred']:.2f}, y={sample_check['y_pred']:.2f}"
        )
        print("Sample 000 true: x=0.0, y=0.0")

        return True

    except Exception as e:
        print(f"FAILED with error: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_data_alignment():
    """Test that genotype-location alignment is maintained with exclusions."""
    # Create simple test case with known pattern
    np.random.seed(456)
    n_snps = 100
    n_samples = 10

    # Create genotypes where each sample has a unique pattern
    # Sample i has i% of 1s in their genotypes
    geno_array = np.zeros((n_snps, n_samples, 2), dtype=int)
    for i in range(n_samples):
        prop_ones = i / 10.0
        geno_array[:, i, :] = np.random.binomial(1, prop_ones, size=(n_snps, 2))

    genotypes = allel.GenotypeArray(geno_array)
    samples = np.array([f"sample_{i:03d}" for i in range(n_samples)])

    # Create sample data where location correlates with sample index
    sample_data = pd.DataFrame(
        {
            "sampleID": samples,
            "x": np.arange(n_samples) * 10.0,
            "y": np.arange(n_samples) * 10.0,
        }
    )

    # Exclude samples in the middle
    exclude_list = ["sample_003", "sample_004", "sample_005"]

    # Test WITHOUT exclusion first
    locator1 = Locator(
        {
            "out": "test_no_exclusion",
            "sample_data": sample_data,
            "epochs": 5,
            "verbose": 0,
            "min_mac": 1,
        }
    )

    # Test WITH exclusion
    locator2 = Locator(
        {
            "out": "test_with_exclusion",
            "sample_data": sample_data,
            "exclude_samples": exclude_list,
            "epochs": 5,
            "verbose": 0,
            "min_mac": 1,
        }
    )

    print("\nTesting data alignment with exclusions...")

    # Train both models
    locator1.train(genotypes=genotypes, samples=samples)
    locator2.train(genotypes=genotypes, samples=samples)

    # Check that sample 006 predictions are reasonable
    # (should still correlate with its true location)
    test_idx_no_excl = 6  # sample_006 is at index 6 without exclusion

    # Get predictions for sample_006
    pred1 = locator1.predict(
        genotypes=genotypes[:, [test_idx_no_excl]],
        samples=["sample_006"],
        return_df=True,
    )
    pred2 = locator2.predict(
        genotypes=genotypes[:, [test_idx_no_excl]],
        samples=["sample_006"],
        return_df=True,
    )

    print("Sample 006 true location: x=60.0, y=60.0")
    print(f"Without exclusion: x={pred1.iloc[0]['x']:.1f}, y={pred1.iloc[0]['y']:.1f}")
    print(f"With exclusion: x={pred2.iloc[0]['x']:.1f}, y={pred2.iloc[0]['y']:.1f}")

    # Both should be reasonable predictions (not completely random)
    # If data was misaligned, predictions would be way off
    error1 = abs(pred1.iloc[0]["x"] - 60.0) + abs(pred1.iloc[0]["y"] - 60.0)
    error2 = abs(pred2.iloc[0]["x"] - 60.0) + abs(pred2.iloc[0]["y"] - 60.0)

    print(f"Total error without exclusion: {error1:.1f}")
    print(f"Total error with exclusion: {error2:.1f}")

    # Errors should be reasonable (not > 100 which would indicate misalignment)
    if error1 > 100 or error2 > 100:
        print("ERROR: Predictions indicate data misalignment!")
        return False

    print("Data alignment appears correct")
    return True


if __name__ == "__main__":
    import ray

    # Shut down any existing Ray instance
    if ray.is_initialized():
        ray.shutdown()

    success1 = test_parallel_kfold_with_exclusion()

    # Shutdown Ray before second test
    ray.shutdown()

    success2 = test_data_alignment()

    if success1 and success2:
        print("\nAll tests passed!")
    else:
        print("\nSome tests failed!")
        exit(1)
