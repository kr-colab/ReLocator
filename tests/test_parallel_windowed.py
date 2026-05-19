"""Smoke test for the actor-based parallel_windows_holdouts dispatcher."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest
from conftest import make_test_genotypes

from locator import Locator
from locator.parallel import parallel_windows_holdouts


@pytest.fixture
def small_data_with_positions(tmp_path):
    genotypes, samples, sample_df = make_test_genotypes(
        n_snps=100, n_samples=20, n_known=20
    )
    sample_file = tmp_path / "samples.txt"
    content = "sampleID\tx\ty\n"
    for sid, (x, y) in zip(samples, sample_df[["x", "y"]].values, strict=True):
        content += f"{sid}\t{x}\t{y}\n"
    sample_file.write_text(content)

    config = {
        "out": str(tmp_path / "win_test"),
        "sample_data": str(sample_file),
        "max_epochs": 2,
        "patience": 1,
        "batch_size": 4,
        "width": 16,
        "nlayers": 2,
        "keras_verbose": 0,
        "verbose_splits": False,
        "holdout_no_intermediate_saves": True,
        "save_fold_models": False,
        "min_snps_per_window": 5,
    }
    loc = Locator(config)
    # Inject synthetic positions on a single chromosome so the dispatcher
    # doesn't need to load from VCF/zarr.
    loc.positions = np.arange(1, len(genotypes) + 1) * 1000
    return loc, genotypes, samples, tmp_path


@pytest.mark.slow
def test_parallel_windows_holdouts_cpu_smoke(small_data_with_positions):
    """Two windows on CPU; verify wide-format output + chunk file."""
    loc, genotypes, samples, tmp_path = small_data_with_positions
    holdout_indices = list(range(0, 4))  # hold out first 4 samples
    df = parallel_windows_holdouts(
        locator=loc,
        genotypes=genotypes,
        samples=samples,
        window_start=0,
        window_size=50_000,
        respect_chromosomes=False,
        holdout_indices=holdout_indices,
        gpu_ids=[],
        gpu_fraction=0.0,
        return_df=True,
        save_full_pred_matrix=True,
        verbose=False,
    )
    assert df is not None
    assert "sampleID" in df.columns
    # At least one x_<label> / y_<label> pair per window present
    x_cols = [c for c in df.columns if c.startswith("x_")]
    y_cols = [c for c in df.columns if c.startswith("y_")]
    assert len(x_cols) >= 1 and len(y_cols) == len(x_cols)

    chunk_path = str(tmp_path / "win_test_windows_chunks.csv")
    assert os.path.exists(chunk_path)
    chunks = pd.read_csv(chunk_path)
    assert chunks["window_label"].nunique() >= 1
