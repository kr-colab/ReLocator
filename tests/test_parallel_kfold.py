"""Tests for the actor-based k-fold / LOO parallel dispatcher.

These hit the new ``locator.parallel.kfold`` path: per-fold checkpointing,
resume behavior, and the basic Ray ActorPool round-trip. They run on CPU
(``gpu_ids=[]``) to keep the test suite portable.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest
from conftest import make_test_genotypes

from locator import Locator

try:
    import ray  # noqa: F401

    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False

pytestmark = pytest.mark.skipif(not RAY_AVAILABLE, reason="Ray not installed")

from locator.parallel import (  # noqa: E402
    parallel_k_fold_holdouts,
    parallel_leave_one_out,
)


@pytest.fixture
def small_data(tmp_path):
    """Tiny synthetic dataset + persisted sample TSV; CPU-runnable."""
    genotypes, samples, sample_df = make_test_genotypes(
        n_snps=200, n_samples=20, n_known=20
    )
    sample_file = tmp_path / "samples.txt"
    content = "sampleID\tx\ty\n"
    for sid, (x, y) in zip(samples, sample_df[["x", "y"]].values, strict=True):
        content += f"{sid}\t{x}\t{y}\n"
    sample_file.write_text(content)

    config = {
        "out": str(tmp_path / "kfold_test"),
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


def _assert_predictions_shape(df, expected_samples, expected_folds):
    assert set(df.columns) == {"sampleID", "x_pred", "y_pred", "fold"}
    assert df["fold"].nunique() == expected_folds
    assert set(df["sampleID"].astype(str).unique()) == set(map(str, expected_samples))


@pytest.mark.slow
def test_parallel_k_fold_cpu_smoke(small_data):
    """Three folds on CPU; verify output structure + checkpoint file."""
    loc, genotypes, samples, tmp_path = small_data
    df = parallel_k_fold_holdouts(
        locator=loc,
        genotypes=genotypes,
        samples=samples,
        k=3,
        gpu_ids=[],
        gpu_fraction=0.0,
        return_df=True,
        verbose=False,
    )
    _assert_predictions_shape(df, samples, expected_folds=3)
    csv_path = str(tmp_path / "kfold_test_kfold_holdouts_predlocs.csv")
    assert os.path.exists(csv_path)
    on_disk = pd.read_csv(csv_path)
    assert len(on_disk) == len(df)


@pytest.mark.slow
def test_parallel_k_fold_resume(small_data):
    """First run writes 2 folds; second run with k=4 completes remaining 2."""
    loc, genotypes, samples, tmp_path = small_data
    csv_path = str(tmp_path / "kfold_test_kfold_holdouts_predlocs.csv")

    # Seed the checkpoint file with two folds' worth of fake rows so the
    # dispatcher should skip them and only run folds 2 and 3.
    rng = np.random.default_rng(0)
    rows = []
    for fold in (0, 1):
        for sid in samples:
            rows.append(
                {
                    "sampleID": str(sid),
                    "x_pred": float(rng.uniform(-120, -100)),
                    "y_pred": float(rng.uniform(30, 50)),
                    "fold": fold,
                }
            )
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    df = parallel_k_fold_holdouts(
        locator=loc,
        genotypes=genotypes,
        samples=samples,
        k=4,
        gpu_ids=[],
        gpu_fraction=0.0,
        return_df=True,
        verbose=False,
        resume=True,
    )
    # All four folds should be present after resume
    assert sorted(df["fold"].unique()) == [0, 1, 2, 3]
    # Folds 0/1 should still be the seeded rows (untouched)
    seeded_fold_0 = pd.read_csv(csv_path)[lambda d: d["fold"] == 0]
    assert len(seeded_fold_0) == len(samples)


@pytest.mark.slow
def test_parallel_leave_one_out_cpu(small_data):
    """LOO on CPU with a tiny dataset — checks the LOO wrapper path."""
    loc, genotypes, samples, tmp_path = small_data
    df = parallel_leave_one_out(
        locator=loc,
        genotypes=genotypes,
        samples=samples,
        gpu_ids=[],
        gpu_fraction=0.0,
        return_df=True,
        save_full_pred_matrix=True,
    )
    _assert_predictions_shape(df, samples, expected_folds=len(samples))
    # LOO writes its own aggregated CSV under a distinct suffix
    loo_csv = str(tmp_path / "kfold_test_leave_one_out_predlocs.csv")
    assert os.path.exists(loo_csv)
