"""Shared test fixtures and utilities for locator tests"""

import allel
import numpy as np
import pandas as pd
import pytest


def make_test_genotypes(
    n_snps=100,
    n_samples=50,
    n_known=None,
    seed=None,
    probs=None,
    dtype=np.int8,
):
    """Create test genotype data with optional coordinate DataFrame.

    Parameters
    ----------
    n_snps : int
        Number of SNPs.
    n_samples : int
        Number of samples.
    n_known : int or None
        Number of samples with known coordinates. If None, all samples
        have known coordinates.
    seed : int or None
        Random seed for reproducibility.
    probs : list of float or None
        Probabilities for [0, 1, 2] allele counts. Defaults to
        [0.25, 0.5, 0.25].
    dtype : numpy dtype
        Data type for the genotype array.

    Returns
    -------
    genotypes : allel.GenotypeArray
        Shape (n_snps, n_samples, 2).
    samples : np.ndarray
        Sample ID strings.
    sample_df : pd.DataFrame
        DataFrame with sampleID, x, y columns.
    """
    if seed is not None:
        np.random.seed(seed)
    if probs is None:
        probs = [0.25, 0.5, 0.25]
    if n_known is None:
        n_known = n_samples

    # Vectorized genotype generation
    allele_counts = np.random.choice([0, 1, 2], size=(n_snps, n_samples), p=probs)
    geno_array = np.zeros((n_snps, n_samples, 2), dtype=dtype)
    geno_array[:, :, 0] = (allele_counts >= 1).astype(dtype)
    geno_array[:, :, 1] = (allele_counts >= 2).astype(dtype)

    genotypes = allel.GenotypeArray(geno_array)
    samples = np.array([f"sample_{i}" for i in range(n_samples)])

    # Create coordinate data
    x_coords = np.random.uniform(-180, 180, n_samples).tolist()
    y_coords = np.random.uniform(-90, 90, n_samples).tolist()
    for i in range(n_known, n_samples):
        x_coords[i] = np.nan
        y_coords[i] = np.nan

    sample_df = pd.DataFrame({"sampleID": samples, "x": x_coords, "y": y_coords})

    return genotypes, samples, sample_df


@pytest.fixture
def genotype_data():
    """Create test genotype data"""
    genotypes, samples, sample_df = make_test_genotypes(
        n_snps=100, n_samples=50, n_known=45
    )

    coords = sample_df[["x", "y"]].values
    n_samples = len(samples)
    n_snps = genotypes.shape[0]

    return genotypes, samples, coords, n_samples, n_snps


@pytest.fixture
def sample_data_file(genotype_data, tmp_path):
    """Create sample data file"""
    _, samples, coords, _, _ = genotype_data

    sample_file = tmp_path / "samples.txt"
    content = "sampleID\tx\ty\n"
    for i, sid in enumerate(samples):
        x, y = coords[i]
        content += f"{sid}\t{x}\t{y}\n"

    sample_file.write_text(content)
    return sample_file


@pytest.fixture
def basic_config(tmp_path, sample_data_file):
    """Create basic locator config"""
    return {
        "out": str(tmp_path / "test"),
        "sample_data": str(sample_data_file),
        "max_epochs": 1,
        "keras_verbose": 0,
    }
