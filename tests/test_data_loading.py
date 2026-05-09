"""Tests for data loading functionality without mocking"""

import os
import tempfile
from pathlib import Path

import allel
import numpy as np
import pandas as pd
import pytest

from locator import Locator


def test_load_genotypes_from_dataframe():
    """Test loading genotypes from a pandas DataFrame."""
    # Create test data
    geno_df = pd.DataFrame(
        {1001: [0, 1, 2], 2005: [1, 2, 0], 3010: [2, 0, 1]},
        index=["sample1", "sample2", "sample3"],
    )

    sample_df = pd.DataFrame(
        {
            "sampleID": ["sample1", "sample2", "sample3"],
            "x": [10.0, 20.0, 30.0],
            "y": [5.0, 15.0, 25.0],
        }
    )

    # Create Locator instance
    config = {"genotype_data": geno_df, "sample_data": sample_df}
    locator = Locator(config=config)

    # Load genotypes
    genotypes, samples = locator.load_genotypes()

    # Verify results
    assert isinstance(genotypes, allel.GenotypeArray)
    assert genotypes.shape == (3, 3, 2)  # 3 SNPs, 3 samples, diploid
    np.testing.assert_array_equal(
        samples, np.array(["sample1", "sample2", "sample3"], dtype=object)
    )

    # Verify positions were stored
    assert hasattr(locator, "positions")
    np.testing.assert_array_equal(
        locator.positions, np.array([1001, 2005, 3010], dtype=float)
    )


def test_load_genotypes_from_matrix_file():
    """Test loading genotypes from a matrix file."""
    # Create temporary matrix file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("sampleID\t1001\t2005\t3010\n")
        f.write("sample1\t0\t1\t2\n")
        f.write("sample2\t1\t2\t0\n")
        f.write("sample3\t2\t0\t1\n")
        matrix_file = f.name

    try:
        # Create sample data
        sample_df = pd.DataFrame(
            {
                "sampleID": ["sample1", "sample2", "sample3"],
                "x": [10.0, 20.0, 30.0],
                "y": [5.0, 15.0, 25.0],
            }
        )

        config = {"sample_data": sample_df}
        locator = Locator(config=config)

        # Load genotypes from matrix
        genotypes, samples = locator.load_genotypes(matrix=matrix_file)

        # Verify results
        assert isinstance(genotypes, allel.GenotypeArray)
        assert genotypes.shape == (3, 3, 2)  # 3 SNPs, 3 samples, diploid
        np.testing.assert_array_equal(
            samples, np.array(["sample1", "sample2", "sample3"])
        )

    finally:
        os.unlink(matrix_file)


def test_load_genotypes_from_matrix_file_continuous_dosage():
    """Test loading a float-dosage matrix bypasses the GenotypeArray round trip."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("sampleID\t1001\t2005\t3010\n")
        f.write("sample1\t0.10\t0.95\t1.85\n")
        f.write("sample2\t1.40\t1.95\t0.05\n")
        f.write("sample3\t1.99\t0.00\t1.05\n")
        matrix_file = f.name

    try:
        sample_df = pd.DataFrame(
            {
                "sampleID": ["sample1", "sample2", "sample3"],
                "x": [10.0, 20.0, 30.0],
                "y": [5.0, 15.0, 25.0],
            }
        )
        config = {"sample_data": sample_df}
        locator = Locator(config=config)

        genotypes, samples = locator.load_genotypes(matrix=matrix_file)

        # Continuous dosage path returns a float ndarray of shape
        # (n_sites, n_samples) — the same layout that filter_snps emits for
        # the integer path.
        assert isinstance(genotypes, np.ndarray)
        assert genotypes.dtype == np.float32
        assert genotypes.shape == (3, 3)  # 3 SNPs, 3 samples
        # Sample 1's site 1001 dosage should be 0.10 (continuous, not rounded).
        np.testing.assert_allclose(genotypes[0, 0], 0.10, atol=1e-5)
        np.testing.assert_array_equal(
            samples, np.array(["sample1", "sample2", "sample3"])
        )

    finally:
        os.unlink(matrix_file)


def test_filter_genotypes_continuous_dosage_path():
    """Test that _filter_genotypes routes a 2D float ndarray to the dosage filter."""
    n_sites, n_samples = 50, 20
    rng = np.random.default_rng(0)
    # Continuous dosage matrix in [0, 2]; bias toward intermediate to avoid
    # near-monomorphic columns being filtered out by min_mac.
    dosage = rng.beta(2.0, 2.0, size=(n_sites, n_samples)).astype(np.float32) * 2.0

    sample_df = pd.DataFrame(
        {
            "sampleID": [f"s{i}" for i in range(n_samples)],
            "x": np.linspace(0.0, 50.0, n_samples),
            "y": np.linspace(0.0, 50.0, n_samples),
        }
    )
    locator = Locator(config={"sample_data": sample_df, "min_mac": 2})
    out = locator._filter_genotypes(dosage)

    # Shape preserved (no biallelic filtering for continuous dosage); output is
    # contiguous float32 ready to feed the network.
    assert isinstance(out, np.ndarray)
    assert out.dtype == np.float32
    assert out.shape[1] == n_samples
    # Values must remain continuous (not rounded back to 0/1/2).
    assert not np.array_equal(out, np.round(out))


def test_filter_dosage_matrix_skips_legacy_path():
    """Filter free-function handles continuous dosage without invoking ``filter_snps_legacy``.

    ``filter_snps_legacy`` calls ``.count_alleles()`` — which only exists on
    ``allel.GenotypeArray`` — and raises AttributeError on a continuous-dosage
    ndarray. This test confirms that ``filter_dosage_matrix`` (the free function
    used by both training- and prediction-path dispatch in ``locator/data/filters.py``)
    handles this case directly, and that the legacy filter would in fact fail
    if it were dispatched here.
    """
    from locator.data import filter_dosage_matrix, filter_snps_legacy

    n_sites, n_samples = 80, 30
    rng = np.random.default_rng(0)
    dosage = rng.beta(2.0, 2.0, size=(n_sites, n_samples)).astype(np.float32) * 2.0

    out = filter_dosage_matrix(dosage, min_mac=2, max_snps=None)
    assert isinstance(out, np.ndarray)
    assert out.dtype == np.float32
    assert out.ndim == 2
    assert out.shape[1] == n_samples

    # Sanity: the legacy filter must NOT be called on a float ndarray; if it
    # were, this would raise AttributeError on .count_alleles().
    with pytest.raises(AttributeError, match="count_alleles"):
        filter_snps_legacy(dosage, min_mac=2, max_snps=None, impute=False)


def test_filter_dosage_matrix_rejects_nan():
    """NaN in dosage must raise rather than silently masking out every site."""
    from locator.data import filter_dosage_matrix

    dosage = np.array([[0.5, 1.0, 1.5], [np.nan, 0.7, 0.9]], dtype=np.float32)
    with pytest.raises(ValueError, match="NaN"):
        filter_dosage_matrix(dosage, min_mac=1)


def test_load_genotypes_invalid_values():
    """Test that invalid genotype values raise an error."""
    # Create genotype data with invalid values
    geno_df = pd.DataFrame(
        {1001: [0, 1, 3], 2005: [1, 2, 0]},  # 3 is invalid
        index=["sample1", "sample2", "sample3"],
    )

    sample_df = pd.DataFrame(
        {
            "sampleID": ["sample1", "sample2", "sample3"],
            "x": [10.0, 20.0, 30.0],
            "y": [5.0, 15.0, 25.0],
        }
    )

    with pytest.raises(ValueError, match="Genotype values must be 0, 1, or 2"):
        Locator(config={"genotype_data": geno_df, "sample_data": sample_df})


def test_load_genotypes_no_data_provided():
    """Test that an error is raised when no genotype data is provided."""
    sample_df = pd.DataFrame(
        {"sampleID": ["sample1", "sample2"], "x": [10.0, 20.0], "y": [5.0, 15.0]}
    )

    config = {"sample_data": sample_df}
    locator = Locator(config=config)

    with pytest.raises(ValueError, match="No genotype data provided"):
        locator.load_genotypes()


def test_sample_data_property():
    """Test the sample_data property."""
    sample_df = pd.DataFrame(
        {"sampleID": ["sample1", "sample2"], "x": [10.0, 20.0], "y": [5.0, 15.0]}
    )

    locator = Locator(config={"sample_data": sample_df})

    # Access property
    result = locator.sample_data

    # Verify it returns the same data
    pd.testing.assert_frame_equal(result, sample_df)


def test_sample_data_property_from_file():
    """Test the sample_data property when loaded from file."""
    # Create temporary sample file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("sampleID\tx\ty\n")
        f.write("sample1\t10.0\t5.0\n")
        f.write("sample2\t20.0\t15.0\n")
        sample_file = f.name

    try:
        locator = Locator(config={"sample_data": sample_file})

        # Access property (should trigger loading from file)
        result = locator.sample_data

        # Verify data
        assert len(result) == 2
        assert list(result.columns) == ["sampleID", "x", "y"]
        assert result["sampleID"].tolist() == ["sample1", "sample2"]

    finally:
        os.unlink(sample_file)


def test_matrix_file_with_invalid_genotypes():
    """Test loading from matrix file with invalid genotype values."""
    # Create temporary matrix file with invalid values
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("sampleID\t1001\t2005\n")
        f.write("sample1\t0\t4\n")  # 4 is invalid
        f.write("sample2\t1\t2\n")
        matrix_file = f.name

    try:
        sample_df = pd.DataFrame(
            {"sampleID": ["sample1", "sample2"], "x": [10.0, 20.0], "y": [5.0, 15.0]}
        )

        config = {"sample_data": sample_df}
        locator = Locator(config=config)

        with pytest.raises(ValueError, match="Genotype values must be 0, 1, or 2"):
            locator.load_genotypes(matrix=matrix_file)

    finally:
        os.unlink(matrix_file)


def test_vcf_file_not_found():
    """Test loading from non-existent VCF file."""
    sample_df = pd.DataFrame(
        {"sampleID": ["sample1", "sample2"], "x": [10.0, 20.0], "y": [5.0, 15.0]}
    )

    config = {"sample_data": sample_df}
    locator = Locator(config=config)

    # allel.read_vcf raises FileNotFoundError for non-existent files
    with pytest.raises(FileNotFoundError):
        locator.load_genotypes(vcf="nonexistent.vcf")
