"""Test that sample exclusion works correctly with analysis methods after fixes."""

import allel
import numpy as np
import pandas as pd

from locator import Locator


def test_train_holdout_with_exclusion():
    """Test that train_holdout handles excluded samples correctly."""
    # Create test data
    np.random.seed(42)
    n_snps = 1000
    n_samples = 50

    # Create genotype array with good variation
    geno_array = np.zeros((n_snps, n_samples, 2), dtype=int)
    for i in range(n_snps):
        p = np.random.uniform(0.1, 0.5)
        geno_array[i, :, :] = np.random.binomial(1, p, size=(n_samples, 2))

    genotypes = allel.GenotypeArray(geno_array)
    samples = np.array([f"sample_{i:03d}" for i in range(n_samples)])

    # Create sample data
    sample_data = pd.DataFrame(
        {
            "sampleID": samples,
            "x": np.random.uniform(-120, -80, n_samples),
            "y": np.random.uniform(30, 50, n_samples),
        }
    )

    # Initialize Locator with exclusions
    exclude_list = ["sample_005", "sample_015", "sample_025", "sample_035", "sample_045"]
    locator = Locator(
        {
            "out": "test_exclusion",
            "sample_data": sample_data,
            "exclude_samples": exclude_list,
            "epochs": 5,
            "verbose": 0,
            "min_mac": 2,
        }
    )

    # This should work without IndexError
    try:
        locator.train_holdout(genotypes=genotypes, samples=samples, k=5)
        print("SUCCESS: train_holdout completed without errors")

        # Verify samples were filtered
        assert len(locator.samples) == n_samples - len(exclude_list)
        print(
            f"Verified: {len(locator.samples)} samples after exclusion (expected {n_samples - len(exclude_list)})"
        )

        # Make predictions
        preds = locator.predict_holdout(verbose=False, return_df=True)
        print(f"Got predictions for {len(preds)} holdout samples")

        # Check that excluded samples are not in predictions
        for excluded in exclude_list:
            assert excluded not in preds["sampleID"].values
        print("Verified: Excluded samples not in predictions")

        return True

    except Exception as e:
        print(f"FAILED with error: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_run_holdouts_simple():
    """Test run_holdouts with a simple case."""
    # Create minimal test data
    np.random.seed(42)
    n_snps = 500
    n_samples = 30

    # Create genotypes with good variation
    geno_array = np.zeros((n_snps, n_samples, 2), dtype=int)
    for i in range(n_snps):
        p = np.random.uniform(0.2, 0.4)
        geno_array[i, :, :] = np.random.binomial(1, p, size=(n_samples, 2))

    genotypes = allel.GenotypeArray(geno_array)
    samples = np.array([f"sample_{i:03d}" for i in range(n_samples)])

    # Create sample data
    sample_data = pd.DataFrame(
        {
            "sampleID": samples,
            "x": np.random.uniform(-120, -80, n_samples),
            "y": np.random.uniform(30, 50, n_samples),
        }
    )

    # Initialize with exclusions
    exclude_list = ["sample_002", "sample_012", "sample_022"]
    locator = Locator(
        {
            "out": "test_holdouts",
            "sample_data": sample_data,
            "exclude_samples": exclude_list,
            "epochs": 3,
            "verbose": 0,
            "min_mac": 2,
        }
    )

    try:
        # Run holdouts
        results = locator.run_holdouts(
            genotypes=genotypes, samples=samples, k=3, n_reps=1, return_df=True
        )

        print("SUCCESS: run_holdouts completed without errors")
        print(f"Got predictions for {len(results)} samples")

        # Check excluded samples not in results
        for excluded in exclude_list:
            assert excluded not in results["sampleID"].values
        print("Verified: Excluded samples not in holdout predictions")

        return True

    except Exception as e:
        print(f"FAILED with error: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("Testing train_holdout with exclusion...")
    success1 = test_train_holdout_with_exclusion()

    print("\nTesting run_holdouts with exclusion...")
    success2 = test_run_holdouts_simple()

    if success1 and success2:
        print("\nAll tests passed!")
    else:
        print("\nSome tests failed!")
        exit(1)
