"""Test sample exclusion with parallel analysis methods."""

import allel
import numpy as np
import pandas as pd
import pytest

from locator import Locator


class TestSampleExclusionParallel:
    """Test sample exclusion with parallel analysis methods."""

    def setup_method(self):
        """Set up test data."""
        # Create test genotypes
        np.random.seed(42)
        n_snps = 100
        n_samples = 20

        # Create genotype array (diploid) with better variation
        # Make sure SNPs have good allele frequencies to avoid MAC filtering
        geno_array = np.zeros((n_snps, n_samples, 2), dtype=int)
        for i in range(n_snps):
            # Create SNPs with reasonable minor allele frequencies
            # Use binomial distribution to simulate realistic allele frequencies
            p = np.random.uniform(0.1, 0.5)  # MAF between 10% and 50%
            geno_array[i, :, :] = np.random.binomial(1, p, size=(n_samples, 2))

        self.genotypes = allel.GenotypeArray(geno_array)

        # Create sample IDs
        self.samples = np.array([f"sample_{i:03d}" for i in range(n_samples)])

        # Create sample data with locations
        self.sample_data = pd.DataFrame(
            {
                "sampleID": self.samples,
                "x": np.random.uniform(-120, -80, n_samples),
                "y": np.random.uniform(30, 50, n_samples),
            }
        )

    def test_holdouts_with_exclusion(self):
        """Test run_holdouts with excluded samples."""
        # Initialize Locator with sample exclusion
        locator = Locator(
            {
                "out": "test_holdouts_exclusion",
                "sample_data": self.sample_data,
                "exclude_samples": ["sample_002", "sample_005", "sample_008"],
                "min_mac": 1,  # Lower MAC threshold to keep more SNPs
                "epochs": 10,  # Reduce epochs for faster testing
                "patience": 5,
                "verbose": 0,  # Disable verbose to avoid progress bar issues
            }
        )

        # Check data
        locator.check_data(self.genotypes, self.samples)

        # Run holdouts - should work without index errors
        results = locator.run_holdouts(
            self.genotypes,
            self.samples,
            k=2,  # Smaller holdout to leave more training samples
            n_reps=1,  # Just one rep for testing
            return_df=True,
        )

        # Check results
        assert isinstance(results, pd.DataFrame)
        # Should only have predictions for holdout samples (not excluded ones)
        assert "sample_002" not in results["sampleID"].values
        assert "sample_005" not in results["sampleID"].values
        assert "sample_008" not in results["sampleID"].values

    def test_jacknife_holdouts_with_exclusion(self):
        """Test run_jacknife_holdouts with excluded samples."""
        # Initialize Locator with sample exclusion
        locator = Locator(
            {
                "out": "test_jacknife_exclusion",
                "sample_data": self.sample_data,
                "exclude_samples": ["sample_001", "sample_010"],
            }
        )

        # Run jacknife holdouts
        results = locator.run_jacknife_holdouts(
            self.genotypes, self.samples, k=3, prop=0.1, n_boots=2, return_df=True
        )

        # Check results
        assert isinstance(results, pd.DataFrame)
        # Should have jacknife predictions for holdout samples
        assert "x_boot0" in results.columns
        assert "y_boot0" in results.columns
        # Excluded samples should not be in results
        assert "sample_001" not in results["sampleID"].values
        assert "sample_010" not in results["sampleID"].values

    def test_k_fold_holdouts_with_exclusion(self):
        """Test run_k_fold_holdouts with excluded samples."""
        # Initialize Locator with sample exclusion
        locator = Locator(
            {"out": "test_kfold_exclusion", "sample_data": self.sample_data}
        )

        # Exclude some samples interactively
        locator.exclude_samples(
            ["sample_003", "sample_012", "sample_018"], reason="test_exclusion"
        )

        # Run k-fold holdouts
        results = locator.run_k_fold_holdouts(
            self.genotypes, self.samples, k=3, return_df=True
        )

        # Check results
        assert isinstance(results, pd.DataFrame)
        # All non-excluded samples should have predictions
        expected_samples = set(self.samples) - {"sample_003", "sample_012", "sample_018"}
        assert set(results["sampleID"].unique()) == expected_samples

    def test_windows_holdouts_with_exclusion(self):
        """Test run_windows_holdouts with excluded samples."""
        # Initialize Locator with sample exclusion
        locator = Locator(
            {
                "out": "test_windows_exclusion",
                "sample_data": self.sample_data,
                "vcf": "dummy.vcf",  # Will be mocked below
            }
        )

        # Mock positions for window analysis
        locator.positions = np.arange(100) * 10000  # 100 SNPs at 10kb intervals

        # Exclude some samples
        locator.exclude_samples(["sample_004", "sample_007"], reason="window_test")

        # Run windows holdouts on specific samples
        holdout_samples = ["sample_000", "sample_015"]  # Non-excluded samples

        results = locator.run_windows_holdouts(
            self.genotypes,
            self.samples,
            holdout_sample_ids=holdout_samples,
            window_size=50000,  # 50kb windows
            n_windows=2,
            return_df=True,
        )

        # Check results
        assert isinstance(results, pd.DataFrame)
        # Should have predictions for the holdout samples in each window
        assert set(results["sampleID"].unique()) == set(holdout_samples)
        # Should have window information
        assert "window" in results.columns


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
