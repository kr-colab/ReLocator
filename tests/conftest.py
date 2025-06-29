"""Shared test fixtures and utilities for locator tests"""

import numpy as np
import pytest
import allel


@pytest.fixture
def genotype_data():
    """Create test genotype data"""
    n_samples = 50
    n_snps = 100
    
    # Create genotype data (biallelic)
    geno_array = np.zeros((n_snps, n_samples, 2), dtype=np.int8)
    for i in range(n_snps):
        # Create biallelic genotypes (0/0, 0/1, 1/1)
        for j in range(n_samples):
            allele_count = np.random.choice([0, 1, 2], p=[0.25, 0.5, 0.25])
            if allele_count == 0:
                geno_array[i, j, :] = [0, 0]
            elif allele_count == 1:
                geno_array[i, j, :] = [0, 1]
            else:
                geno_array[i, j, :] = [1, 1]
    
    genotypes = allel.GenotypeArray(geno_array)
    samples = np.array([f"sample_{i}" for i in range(n_samples)])
    
    # Create coordinates (some with NA)
    coords = np.random.uniform(-180, 180, size=(n_samples, 2))
    coords[40:45, :] = np.nan  # Make some samples have NA coordinates
    
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