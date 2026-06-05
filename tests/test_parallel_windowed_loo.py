"""Tests for the full leave-one-out-per-window dispatcher."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from locator import Locator

try:
    import ray  # noqa: F401

    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False


def test_parallel_windows_leave_one_out_is_exported():
    import locator.parallel as lp

    assert "parallel_windows_leave_one_out" in lp.__all__
    assert hasattr(lp, "parallel_windows_leave_one_out")


@pytest.fixture
def small_dosage_loo(tmp_path):
    rng = np.random.default_rng(0)
    n_snps, n_samples = 60, 8
    dosage = rng.uniform(0.0, 2.0, size=(n_snps, n_samples)).astype(np.float32)
    samples = np.array([f"s{i}" for i in range(n_samples)], dtype=object)
    sample_file = tmp_path / "samples.txt"
    content = "sampleID\tx\ty\n"
    for i, sid in enumerate(samples):
        content += f"{sid}\t{float(i)}\t{float(i * 2)}\n"
    sample_file.write_text(content)

    config = {
        "out": str(tmp_path / "loo_test"),
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
        "min_snps_per_window": 3,
    }
    loc = Locator(config)
    loc.positions = np.arange(1, n_snps + 1) * 1000  # single chromosome
    return loc, dosage, samples, tmp_path


@pytest.mark.skipif(not RAY_AVAILABLE, reason="Ray not installed")
@pytest.mark.slow
def test_parallel_windows_leave_one_out_predicts_every_sample_per_window(
    small_dosage_loo,
):
    from locator.parallel import parallel_windows_leave_one_out

    loc, dosage, samples, tmp_path = small_dosage_loo
    df = parallel_windows_leave_one_out(
        locator=loc,
        genotypes=dosage,
        samples=samples,
        window_start=0,
        window_size=20_000,
        respect_chromosomes=False,
        gpu_ids=[],
        gpu_fraction=0.0,
        return_df=True,
        save_full_pred_matrix=True,
        verbose=False,
    )
    assert df is not None
    # Full LOO: every sample appears, in every window column, with no gaps.
    assert len(df) == 8
    x_cols = [c for c in df.columns if c.startswith("x_")]
    y_cols = [c for c in df.columns if c.startswith("y_")]
    n_windows = len(x_cols)
    assert n_windows >= 2
    assert len(y_cols) == n_windows
    assert not df[x_cols + y_cols].isna().any().any()

    chunk_path = str(tmp_path / "loo_test_windows_loo_chunks.csv")
    chunks = pd.read_csv(chunk_path)
    per_window = chunks.groupby("window_label")["sampleID"].nunique()
    assert (per_window == 8).all()  # every window predicted all 8 samples

    # Resume: a second run adds no fits (all windows already checkpointed).
    n_rows_before = len(chunks)
    parallel_windows_leave_one_out(
        locator=loc,
        genotypes=dosage,
        samples=samples,
        window_start=0,
        window_size=20_000,
        respect_chromosomes=False,
        gpu_ids=[],
        gpu_fraction=0.0,
        return_df=False,
        verbose=False,
        resume=True,
    )
    assert len(pd.read_csv(chunk_path)) == n_rows_before


@pytest.mark.skipif(not RAY_AVAILABLE, reason="Ray not installed")
def test_parallel_windows_leave_one_out_rejects_hardcall(small_dosage_loo):
    """Hard-call GenotypeArray input is rejected with a clear error."""
    import allel

    from locator.parallel import parallel_windows_leave_one_out

    loc, _dosage, samples, _tmp = small_dosage_loo
    hardcall = allel.GenotypeArray(np.zeros((60, len(samples), 2), dtype="i1"))
    with pytest.raises(ValueError, match="continuous-dosage"):
        parallel_windows_leave_one_out(
            locator=loc,
            genotypes=hardcall,
            samples=samples,
            gpu_ids=[],
            gpu_fraction=0.0,
            verbose=False,
        )
