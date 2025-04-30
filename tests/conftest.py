"""Shared fixtures for testing."""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path


@pytest.fixture
def test_data_dir():
    """Return the path to the test data directory."""
    return Path(__file__).parent / "test_data"


@pytest.fixture
def sample_genotype_data():
    """Create sample genotype data for testing."""
    return np.random.randint(0, 3, size=(10, 100))  # 10 samples, 100 loci


@pytest.fixture
def sample_metadata():
    """Create sample metadata for testing."""
    return pd.DataFrame(
        {
            "sampleID": [f"sample_{i}" for i in range(10)],
            "x": np.random.uniform(0, 50, 10),
            "y": np.random.uniform(0, 50, 10),
            "population": [f"pop_{i%3}" for i in range(10)],
        }
    )
