"""Tests for the actor-based parallel_holdouts dispatcher.

Same setup as test_parallel_kfold but exercising the holdout path: wide-format
output (x_rep0/y_rep0/x_rep1/...), long-format per-replicate checkpointing,
and resume semantics.
"""

from __future__ import annotations

import os

import pandas as pd
import pytest
from conftest import make_test_genotypes

from locator import Locator
from locator.parallel import parallel_holdouts


@pytest.fixture
def small_data(tmp_path):
    genotypes, samples, sample_df = make_test_genotypes(
        n_snps=200, n_samples=20, n_known=20
    )
    sample_file = tmp_path / "samples.txt"
    content = "sampleID\tx\ty\n"
    for sid, (x, y) in zip(samples, sample_df[["x", "y"]].values, strict=True):
        content += f"{sid}\t{x}\t{y}\n"
    sample_file.write_text(content)

    config = {
        "out": str(tmp_path / "holdout_test"),
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
    }
    return Locator(config), genotypes, samples, tmp_path


@pytest.mark.slow
def test_parallel_holdouts_cpu_smoke(small_data):
    """Three replicates on CPU; verify wide-format output + chunk file."""
    loc, genotypes, samples, tmp_path = small_data
    df = parallel_holdouts(
        locator=loc,
        genotypes=genotypes,
        samples=samples,
        k=3,
        n_reps=3,
        gpu_ids=[],
        gpu_fraction=0.0,
        return_df=True,
        save_full_pred_matrix=True,
        verbose=False,
    )
    assert "sampleID" in df.columns
    for rep in range(3):
        assert f"x_rep{rep}" in df.columns
        assert f"y_rep{rep}" in df.columns

    chunk_path = str(tmp_path / "holdout_test_holdouts_chunks.csv")
    assert os.path.exists(chunk_path)
    chunks = pd.read_csv(chunk_path)
    assert set(chunks["rep"].unique()) == {0, 1, 2}


@pytest.mark.slow
def test_parallel_holdouts_resume(small_data):
    """Pre-seeded chunk file → dispatcher skips done replicates and tops up the rest."""
    loc, genotypes, samples, tmp_path = small_data
    chunk_path = str(tmp_path / "holdout_test_holdouts_chunks.csv")

    rows = []
    for rep in (0, 1):
        for sid in samples:
            rows.append(
                {
                    "sampleID": str(sid),
                    "x_pred": -110.0,
                    "y_pred": 35.0,
                    "rep": rep,
                }
            )
    pd.DataFrame(rows).to_csv(chunk_path, index=False)

    df = parallel_holdouts(
        locator=loc,
        genotypes=genotypes,
        samples=samples,
        k=3,
        n_reps=4,
        gpu_ids=[],
        gpu_fraction=0.0,
        return_df=True,
        save_full_pred_matrix=False,
        verbose=False,
        resume=True,
    )
    chunks = pd.read_csv(chunk_path)
    assert sorted(chunks["rep"].unique()) == [0, 1, 2, 3]
    # Reps 0 and 1 should still be the seeded values
    seeded = chunks[chunks["rep"] == 0]
    assert (seeded["x_pred"] == -110.0).all()
    # Wide output covers all four reps
    for rep in range(4):
        assert f"x_rep{rep}" in df.columns
