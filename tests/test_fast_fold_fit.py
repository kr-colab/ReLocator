"""Tests for the fast fold-fit path (repeat + steps + on-device validation)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from locator.core import Locator
from locator.data.indexset import IndexSet
from locator.data.tf_dataset import make_tf_dataset
from locator.models import euclidean_distance_loss
from locator.training import _OnDeviceValLoss


def _sample_data(path: Path, n: int) -> None:
    rows = ["sampleID\tx\ty"]
    for i in range(n):
        rows.append(f"s{i}\t{float(i)}\t{float(2 * i)}")
    path.write_text("\n".join(rows) + "\n")


def test_on_device_val_loss_matches_keras_evaluate(tmp_path):
    """The callback's val loss must equal model.evaluate on the val split.

    Builds the model through a Locator (which configures TF threading before
    any op runs), matching how every other test in the suite starts TF.
    """
    rng = np.random.default_rng(0)
    n_snps, n_samples = 100, 16
    dosage = rng.uniform(0.0, 2.0, size=(n_snps, n_samples)).astype(np.float32)
    coords = rng.normal(size=(n_samples, 2)).astype(np.float32)
    sd = tmp_path / "samples.tsv"
    _sample_data(sd, n_samples)

    loc = Locator(
        {
            "sample_data": str(sd),
            "width": 32,
            "nlayers": 2,
            "keras_verbose": 0,
            "out": str(tmp_path / "run"),
        }
    )
    loc._filter_genotypes(dosage)
    model = loc._create_model(input_shape=loc.filtered_genotypes.shape[0])

    iset = IndexSet(
        indices={
            "train": np.arange(n_samples - 3),
            "test": np.arange(n_samples - 3, n_samples),
        },
        total_samples=n_samples,
    )
    val_ds = make_tf_dataset(coords, iset, "test", batch_size=32, training=False)
    keras_val = model.evaluate(val_ds, verbose=0)

    val_idx = np.asarray(iset.get_split("test"))
    cb = _OnDeviceValLoss(val_idx, coords[val_idx], euclidean_distance_loss)
    cb.set_model(model)
    cb.on_train_begin()
    cb_val = float(cb._compute())

    assert abs(keras_val - cb_val) < 1e-4


@pytest.mark.slow
@pytest.mark.parametrize("fast", [True, False])
def test_holdout_trains_and_early_stops_both_paths(tmp_path, fast):
    """Fast and slow holdout both train, early-stop, and predict the held-out sample."""
    rng = np.random.default_rng(1)
    n_snps, n_samples = 200, 60
    dosage = rng.uniform(0.0, 2.0, size=(n_snps, n_samples)).astype(np.float32)
    samples = np.array([f"s{i}" for i in range(n_samples)], dtype=object)
    sd = tmp_path / "samples.tsv"
    _sample_data(sd, n_samples)

    loc = Locator(
        {
            "sample_data": str(sd),
            "max_epochs": 40,
            "patience": 5,
            "keras_verbose": 0,
            "out": str(tmp_path / "run"),
            "holdout_no_intermediate_saves": True,
            "save_fold_models": False,
            "fast_fold_fit": fast,
        }
    )
    loc._filter_genotypes(dosage)
    loc.train_holdout(
        genotypes=None,
        samples=samples,
        holdout_indices=[3],
        filtered_genotypes=loc.filtered_genotypes,
    )
    preds = loc.predict_holdout(
        verbose=False,
        return_df=True,
        save_preds_to_disk=False,
        plot_summary=False,
        plot_map=False,
    )
    assert "val_loss" in loc.history.history
    assert len(loc.history.history["loss"]) <= 40  # early stopping may cut short
    assert list(preds["sampleID"]) == ["s3"]
    assert np.isfinite(preds[["x_pred", "y_pred"]].to_numpy()).all()


@pytest.mark.slow
def test_fast_holdout_handles_split_smaller_than_batch(tmp_path):
    """Fast path must not hang when the train split is smaller than a batch."""
    rng = np.random.default_rng(2)
    n_snps, n_samples = 100, 20  # 19 train < default batch 32
    dosage = rng.uniform(0.0, 2.0, size=(n_snps, n_samples)).astype(np.float32)
    samples = np.array([f"s{i}" for i in range(n_samples)], dtype=object)
    sd = tmp_path / "samples.tsv"
    _sample_data(sd, n_samples)

    loc = Locator(
        {
            "sample_data": str(sd),
            "max_epochs": 10,
            "patience": 3,
            "keras_verbose": 0,
            "out": str(tmp_path / "run"),
            "holdout_no_intermediate_saves": True,
            "save_fold_models": False,
            # fast is the default for train_holdout
        }
    )
    loc._filter_genotypes(dosage)
    loc.train_holdout(
        genotypes=None,
        samples=samples,
        holdout_indices=[1],
        filtered_genotypes=loc.filtered_genotypes,
    )
    assert loc.history is not None
    assert "val_loss" in loc.history.history
