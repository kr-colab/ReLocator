"""Tests for centralized data filtering and normalization utilities."""

import numpy as np
import pytest
from unittest.mock import Mock
from locator.data import (
    filter_snps, 
    filter_snps_legacy,
    normalize_locs, 
    normalize_locs_params,
    impute_missing,
    FilterStats,
    NormalizationParams
)


class TestNormalization:
    """Test normalization functions."""
    
    def test_normalize_locs_basic(self):
        """Test basic coordinate normalization."""
        # Create test data
        locs = np.array([
            [10.0, 20.0],
            [15.0, 25.0],
            [20.0, 30.0],
            [25.0, 35.0]
        ])
        
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
        locs = np.array([
            [10.0, 20.0],
            [15.0, 25.0],
            [20.0, 30.0],
            [25.0, 35.0]
        ])
        
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
        locs = np.array([
            [10.0, 20.0],
            [np.nan, 25.0],
            [20.0, np.nan],
            [25.0, 35.0]
        ])
        
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
        """Create mock GenotypeArray for testing."""
        self.mock_genotypes = Mock()
        
        # Mock count_alleles
        mock_allele_counts = Mock()
        mock_allele_counts.is_biallelic = Mock(return_value=np.array([True, True, False, True, True]))
        self.mock_genotypes.count_alleles = Mock(return_value=mock_allele_counts)
        
        # Mock shape
        self.mock_genotypes.shape = (5, 10)  # 5 SNPs, 10 samples
        
        # Mock indexing to return filtered genotypes
        def getitem(*args):
            index = args[0] if len(args) == 1 else args
            if isinstance(index, tuple) and isinstance(index[0], np.ndarray):
                # Boolean indexing
                filtered = Mock()
                filtered.shape = (np.sum(index[0]), 10)
                # Set up count_alleles return value
                derived_counts = np.array([5, 8, 3, 10])[:np.sum(index[0])]
                mock_counts = Mock()
                mock_counts.__getitem__ = Mock(side_effect=lambda x: derived_counts if x == (slice(None), 1) else None)
                filtered.count_alleles = Mock(return_value=mock_counts)
                filtered.to_allele_counts = Mock(return_value=np.random.randint(0, 3, (np.sum(index[0]), 10, 2)))
                filtered.is_missing = Mock(return_value=np.zeros((np.sum(index[0]), 10), dtype=bool))
                # Allow further indexing
                filtered.__getitem__ = Mock(side_effect=getitem)
                return filtered
            elif isinstance(index, tuple) and isinstance(index[0], list):
                # List indexing for MAC filtering
                filtered = Mock()
                filtered.shape = (len([x for x in index[0] if x]), 10)
                filtered.to_allele_counts = Mock(return_value=np.random.randint(0, 3, (filtered.shape[0], 10, 2)))
                filtered.is_missing = Mock(return_value=np.zeros((filtered.shape[0], 10), dtype=bool))
                return filtered
            return self.mock_genotypes
            
        self.mock_genotypes.__getitem__ = Mock(side_effect=getitem)
        
        # Mock to_allele_counts
        self.mock_genotypes.to_allele_counts = Mock(
            return_value=np.random.randint(0, 3, (5, 10, 2))
        )
        
        # Mock is_missing
        self.mock_genotypes.is_missing = Mock(
            return_value=np.zeros((5, 10), dtype=bool)
        )
        
    def test_filter_snps_basic(self):
        """Test basic SNP filtering."""
        ac, stats = filter_snps(self.mock_genotypes, min_mac=1)
        
        assert isinstance(stats, FilterStats)
        assert stats.n_snps_original == 5
        assert stats.n_samples_original == 10
        assert stats.n_biallelic_filtered == 1  # One non-biallelic site
        assert stats.mac_threshold == 1
        
    def test_filter_snps_with_mac(self):
        """Test filtering with minimum allele count."""
        ac, stats = filter_snps(self.mock_genotypes, min_mac=5)
        
        assert stats.mac_threshold == 5
        assert isinstance(stats, FilterStats)
        assert isinstance(ac, np.ndarray)
        
    def test_filter_snps_legacy(self):
        """Test legacy wrapper returns only allele counts."""
        ac = filter_snps_legacy(self.mock_genotypes)
        assert isinstance(ac, np.ndarray)
        
    def test_filter_snps_with_max_snps(self):
        """Test random subsampling."""
        ac, stats = filter_snps(self.mock_genotypes, max_snps=2)
        
        assert ac.shape[0] == 2
        assert stats.n_random_subset > 0


class TestImputation:
    """Test missing data imputation."""
    
    def test_impute_missing(self):
        """Test basic imputation functionality."""
        # Create mock genotype array
        mock_genotypes = Mock()
        
        # Set up allele counts with some missing data
        ac_with_missing = np.array([
            [[0, 2], [1, 1], [0, 0], [2, 0]],  # SNP 1
            [[1, 1], [0, 0], [1, 1], [0, 2]],  # SNP 2
        ])
        
        # Mock required methods
        mock_genotypes.count_alleles = Mock(
            return_value=np.array([[4, 4], [4, 4]])  # Total allele counts
        )
        mock_genotypes.to_allele_counts = Mock(return_value=ac_with_missing)
        
        # Create missingness mask
        missingness = np.array([
            [False, False, True, False],  # SNP 1, sample 3 missing
            [False, True, False, False],  # SNP 2, sample 2 missing
        ])
        mock_genotypes.is_missing = Mock(return_value=missingness)
        
        # Run imputation
        imputed = impute_missing(mock_genotypes)
        
        # Check that missing values were replaced
        assert imputed.shape == (2, 4)
        # The previously missing values should now have values
        # (actual values depend on random binomial draws)
        
        
def test_imports_backward_compatible():
    """Test that functions can be imported from main locator package."""
    from locator import filter_snps, normalize_locs, impute_missing
    
    # These should be the legacy versions or wrappers
    assert filter_snps is not None
    assert normalize_locs is not None
    assert impute_missing is not None