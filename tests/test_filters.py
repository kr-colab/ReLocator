"""Tests for centralized data filtering and normalization utilities."""

import allel
import numpy as np
import pytest

from locator.data import (
    FilterStats,
    NormalizationParams,
    filter_snps,
    filter_snps_legacy,
    impute_missing,
    normalize_locs,
    normalize_locs_params,
)


class TestNormalization:
    """Test normalization functions."""

    def test_normalize_locs_basic(self):
        """Test basic coordinate normalization."""
        # Create test data
        locs = np.array([[10.0, 20.0], [15.0, 25.0], [20.0, 30.0], [25.0, 35.0]])

        # Test legacy function
        meanlong, sdlong, meanlat, sdlat, unnormed, normed = normalize_locs(locs)

        assert meanlong == np.mean(locs[:, 0])
        assert meanlat == np.mean(locs[:, 1])
        assert np.allclose(np.mean(normed[:, 0]), 0, atol=1e-10)
        assert np.allclose(np.mean(normed[:, 1]), 0, atol=1e-10)
        assert np.allclose(np.std(normed[:, 0]), 1, atol=1e-10)
        assert np.allclose(np.std(normed[:, 1]), 1, atol=1e-10)

    def test_normalize_locs_params(self):
        """Test normalization with params object."""
        locs = np.array([[10.0, 20.0], [15.0, 25.0], [20.0, 30.0], [25.0, 35.0]])

        params, unnormed, normed = normalize_locs_params(locs)

        # Test params object
        assert isinstance(params, NormalizationParams)
        assert params.meanlong == np.mean(locs[:, 0])
        assert params.meanlat == np.mean(locs[:, 1])

        # Test apply and reverse
        normed2 = params.apply(locs)
        assert np.allclose(normed, normed2)

        reversed_locs = params.reverse(normed)
        assert np.allclose(locs, reversed_locs)

    def test_normalize_locs_with_nan(self):
        """Test normalization with NaN values."""
        locs = np.array([[10.0, 20.0], [np.nan, 25.0], [20.0, np.nan], [25.0, 35.0]])

        meanlong, sdlong, meanlat, sdlat, unnormed, normed = normalize_locs(locs)

        # Check that NaN values are preserved
        assert np.isnan(normed[1, 0])
        assert np.isnan(normed[2, 1])

        # Check that stats ignore NaN
        assert meanlong == np.nanmean(locs[:, 0])
        assert meanlat == np.nanmean(locs[:, 1])


class TestFilterSNPs:
    """Test SNP filtering functions."""

    def setup_method(self):
        """Create real GenotypeArray for testing."""
        # Create real genotype array with controlled data
        # 5 SNPs, 10 samples, diploid
        # Make sure to have:
        # - Some biallelic sites (only 0s and 1s)
        # - One triallelic site (has 2s)
        # - Different minor allele counts
        genotype_data = np.array(
            [
                # SNP 0: Biallelic, MAC=5 (5 copies of allele 1)
                [
                    [0, 0],
                    [0, 1],
                    [0, 1],
                    [0, 0],
                    [0, 0],
                    [0, 0],
                    [0, 0],
                    [0, 0],
                    [0, 1],
                    [1, 1],
                ],
                # SNP 1: Biallelic, MAC=8 (8 copies of allele 1)
                [
                    [0, 1],
                    [1, 1],
                    [0, 1],
                    [1, 1],
                    [0, 0],
                    [0, 0],
                    [0, 1],
                    [0, 1],
                    [0, 0],
                    [0, 0],
                ],
                # SNP 2: Triallelic (has allele 2)
                [
                    [0, 0],
                    [0, 1],
                    [1, 2],
                    [0, 0],
                    [0, 0],
                    [0, 2],
                    [0, 0],
                    [0, 0],
                    [0, 0],
                    [0, 0],
                ],
                # SNP 3: Biallelic, MAC=3 (3 copies of allele 1)
                [
                    [0, 0],
                    [0, 1],
                    [0, 0],
                    [0, 0],
                    [0, 0],
                    [0, 0],
                    [0, 0],
                    [0, 0],
                    [0, 1],
                    [0, 1],
                ],
                # SNP 4: Biallelic, MAC=10 (10 copies of allele 1)
                [
                    [0, 1],
                    [1, 1],
                    [0, 1],
                    [1, 1],
                    [0, 1],
                    [0, 1],
                    [0, 0],
                    [0, 0],
                    [0, 0],
                    [0, 0],
                ],
            ],
            dtype=np.int8,
        )

        self.genotypes = allel.GenotypeArray(genotype_data)

    def test_filter_snps_basic(self):
        """Test basic SNP filtering."""
        ac, stats = filter_snps(self.genotypes, min_mac=1)

        assert isinstance(stats, FilterStats)
        assert stats.n_snps_original == 5
        assert stats.n_samples_original == 10
        assert stats.n_biallelic_filtered == 1  # One non-biallelic site
        assert stats.mac_threshold == 1

    def test_filter_snps_with_mac(self):
        """Test filtering with minimum allele count."""
        ac, stats = filter_snps(self.genotypes, min_mac=5)

        assert stats.mac_threshold == 5
        assert isinstance(stats, FilterStats)
        assert isinstance(ac, np.ndarray)
        # We should have 3 SNPs left after filtering (SNPs 0, 1, and 4 have MAC >= 5)
        assert ac.shape[0] == 3
        assert stats.n_mac_filtered == 1  # SNP 3 has MAC=3, filtered out

    def test_filter_snps_legacy(self):
        """Test legacy wrapper returns only allele counts."""
        ac = filter_snps_legacy(self.genotypes)
        assert isinstance(ac, np.ndarray)

    def test_filter_snps_with_max_snps(self):
        """Test random subsampling."""
        ac, stats = filter_snps(self.genotypes, max_snps=2)

        assert ac.shape[0] == 2
        assert stats.n_random_subset > 0


class TestImputation:
    """Test missing data imputation."""

    def test_impute_missing(self):
        """Test basic imputation functionality."""
        # Create genotype array with missing data (-1 indicates missing)
        genotype_data = np.array(
            [
                # SNP 0: has missing data in sample 2
                [[0, 0], [0, 1], [-1, -1], [1, 1], [0, 1]],
                # SNP 1: has missing data in sample 1
                [[0, 1], [-1, -1], [0, 0], [1, 1], [0, 1]],
                # SNP 2: no missing data
                [[0, 0], [0, 1], [1, 1], [0, 1], [1, 1]],
            ],
            dtype=np.int8,
        )

        genotypes = allel.GenotypeArray(genotype_data)

        # Check that we have missing data
        assert genotypes.is_missing().any()

        # Run imputation
        imputed = impute_missing(genotypes)

        # Check that missing values were replaced
        assert imputed.shape == (3, 5)  # 3 SNPs, 5 samples
        # The imputed array should have no negative values
        assert (imputed >= 0).all()
        assert (imputed <= 2).all()  # diploid, so max is 2

    def test_impute_handles_half_and_fully_missing(self):
        """Imputation must not produce NaN AF or AF>1 on edge cases.

        - Half-missing calls (-1, x) overcount in a naive ninds denominator.
        - A fully-missing site yields divide-by-zero in the AF formula.
        Either case previously crashed np.random.binomial.
        """
        # Site 0: half-missing calls. Naive (alt / 2*ninds) = 3 / (2*3) = 0.5 OK
        # Site 1: mostly half-missing alt calls. alt=3, ninds=3, naive=3/6=0.5 OK
        # Site 2: pure alt half-missing. alt=4, ninds=4, naive=4/8=0.5 OK
        # Site 3: mix of (1,1) and (-1, 1). alt=5, ninds=3, naive=5/6>0.83 OK
        # Site 4: severe — all (1, 1) and one (-1, -1). alt=8, ninds=4,
        #         naive=8/8=1.0 (boundary). Add (1, -1) to push naive past 1.
        # Site 5: fully missing — should impute to 0 (AF=0) silently.
        genotype_data = np.array(
            [
                [[-1, 1], [-1, 1], [-1, 1], [-1, 0], [-1, 0]],
                [[-1, 1], [-1, 1], [-1, 1], [-1, 0], [0, 1]],
                [[-1, 1], [-1, 1], [-1, 1], [-1, 1], [0, 1]],
                [[1, 1], [1, 1], [-1, 1], [-1, 1], [0, 0]],
                [[1, 1], [1, 1], [1, 1], [1, -1], [-1, -1]],
                [[-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1]],
            ],
            dtype=np.int8,
        )
        genotypes = allel.GenotypeArray(genotype_data)
        imputed = impute_missing(genotypes)
        assert imputed.shape == (6, 5)
        assert np.isfinite(imputed).all()
        assert (imputed >= 0).all() and (imputed <= 2).all()


def test_imports_backward_compatible():
    """Test that functions can be imported from main locator package."""
    from locator import filter_snps, impute_missing, normalize_locs

    # These should be the legacy versions or wrappers
    assert filter_snps is not None
    assert normalize_locs is not None
    assert impute_missing is not None
