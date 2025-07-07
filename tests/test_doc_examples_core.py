"""Test core examples from documentation to ensure they remain functional.

This is a subset of test_doc_examples.py focusing on the most critical
examples that should always work.
"""

import os
import shutil
import tempfile
from pathlib import Path

import allel
import numpy as np
import pandas as pd
import pytest

from locator import Locator
from locator.data import IndexSet
from locator.plotting import plot_error_summary, plot_predictions


@pytest.fixture
def sample_data():
    """Create minimal sample data for testing."""
    np.random.seed(42)
    n_samples = 20
    n_snps = 100  # More SNPs to ensure some pass filtering

    # Create genotype data to ensure we have biallelic SNPs
    gt_array = np.zeros((n_snps, n_samples, 2), dtype=int)

    # Create a mix of genotypes ensuring each SNP has variation
    for i in range(n_snps):
        # Ensure each SNP is biallelic with some variation
        # Create random genotypes with frequencies that ensure MAC > 0
        for j in range(n_samples):
            if np.random.random() < 0.3:  # 30% homozygous ref
                gt_array[i, j, :] = [0, 0]
            elif np.random.random() < 0.6:  # 30% heterozygous
                gt_array[i, j, :] = [0, 1]
            else:  # 40% homozygous alt
                gt_array[i, j, :] = [1, 1]

    # Add a small amount of missing data
    # Only on first SNP to avoid filtering issues
    gt_array[0, 0:2, :] = -1

    genotypes = allel.GenotypeArray(gt_array)

    # Create sample IDs
    samples = np.array([f"sample_{i:03d}" for i in range(n_samples)])

    # Create coordinate data with some NAs
    coords_df = pd.DataFrame(
        {
            "sampleID": samples,
            "x": np.random.uniform(-120, -110, n_samples),
            "y": np.random.uniform(30, 40, n_samples),
        }
    )
    # Make last 2 samples have NA coordinates
    coords_df.loc[18:19, ["x", "y"]] = np.nan

    return genotypes, samples, coords_df


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test outputs."""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    shutil.rmtree(temp_path)


def test_basic_workflow(sample_data, temp_dir):
    """Test the most basic workflow from documentation."""
    genotypes, samples, coords_df = sample_data

    # Save sample data
    sample_file = os.path.join(temp_dir, "samples.txt")
    coords_df.to_csv(sample_file, sep="\t", index=False)

    # Basic usage example
    loc = Locator(
        {
            "out": os.path.join(temp_dir, "my_analysis"),
            "sample_data": sample_file,
            "max_epochs": 2,
            "patience": 1,
            "keras_verbose": 0,
            "min_mac": 0,  # Disable MAC filtering for test data
            "impute_missing": True,  # Handle missing data
        }
    )

    # Train
    loc.train(genotypes=genotypes, samples=samples)

    # Predict
    predictions = loc.predict(return_df=True)

    # Basic validation
    assert isinstance(predictions, pd.DataFrame)
    assert "sampleID" in predictions.columns
    assert "x" in predictions.columns
    assert "y" in predictions.columns
    assert len(predictions) > 0

    # Check that model was saved
    assert os.path.exists(f"{loc.config['out']}.weights.h5")


def test_na_handling_modes(sample_data, temp_dir):
    """Test NA handling modes from documentation."""
    genotypes, samples, coords_df = sample_data

    # Test 'separate' mode (default)
    loc_separate = Locator(
        {
            "out": os.path.join(temp_dir, "na_separate"),
            "sample_data": coords_df,
            "na_action": "separate",
            "max_epochs": 2,
            "patience": 1,
            "keras_verbose": 0,
        }
    )

    # Should train successfully
    loc_separate.train(genotypes=genotypes, samples=samples)
    preds = loc_separate.predict(return_df=True)

    # Should predict for NA samples
    na_samples = coords_df[coords_df["x"].isna()]["sampleID"].tolist()
    pred_samples = preds["sampleID"].tolist()
    assert any(s in pred_samples for s in na_samples)

    # Test 'exclude' mode
    loc_exclude = Locator(
        {
            "out": os.path.join(temp_dir, "na_exclude"),
            "sample_data": coords_df,
            "na_action": "exclude",
            "max_epochs": 2,
            "patience": 1,
            "keras_verbose": 0,
        }
    )

    loc_exclude.train(genotypes=genotypes, samples=samples)

    # Check that only non-NA samples were used
    # With tf.data pipeline, we check trainlocs instead of traingen
    assert hasattr(loc_exclude, "trainlocs")
    # Training locations should come from non-NA samples only
    n_known = coords_df["x"].notna().sum()
    # Account for train/test split (default 90% train)
    expected_train_size = int(n_known * loc_exclude.config.get("train_split", 0.9))
    # Allow for some variance due to random split
    assert abs(len(loc_exclude.trainlocs) - expected_train_size) <= 2

    # Test 'fail' mode
    loc_fail = Locator(
        {
            "out": os.path.join(temp_dir, "na_fail"),
            "sample_data": coords_df,
            "na_action": "fail",
        }
    )

    # Should raise error
    with pytest.raises(ValueError) as excinfo:
        loc_fail.train(genotypes=genotypes, samples=samples)
    assert "samples without coordinates" in str(excinfo.value)


def test_analysis_methods(sample_data, temp_dir):
    """Test key analysis methods from documentation."""
    genotypes, samples, coords_df = sample_data

    # Use only samples with coordinates
    valid_mask = coords_df["x"].notna()
    coords_df_valid = coords_df[valid_mask].copy()
    genotypes_valid = genotypes[:, valid_mask.values]
    samples_valid = samples[valid_mask.values]

    loc = Locator(
        {
            "out": os.path.join(temp_dir, "analysis"),
            "sample_data": coords_df_valid,
            "max_epochs": 2,
            "patience": 1,
            "keras_verbose": 0,
            "min_mac": 0,  # Disable MAC filtering for test data
            "impute_missing": True,  # Handle missing data
            "batch_size": 16,  # Smaller batch size for small dataset
        }
    )

    # Test jacknife
    jack_results = loc.run_jacknife(
        genotypes=genotypes_valid,
        samples=samples_valid,
        prop=0.2,  # This will create 5 replicates (1/0.2)
        return_df=True,
    )
    assert isinstance(jack_results, pd.DataFrame)
    assert "x_0" in jack_results.columns

    # Test bootstrap with fresh instance
    loc_boot = Locator(
        {
            "out": os.path.join(temp_dir, "bootstrap"),
            "sample_data": coords_df_valid,
            "max_epochs": 2,
            "patience": 1,
            "keras_verbose": 0,
            "min_mac": 0,
            "impute_missing": True,
            "batch_size": 16,
        }
    )
    boot_results = loc_boot.run_bootstraps(
        genotypes=genotypes_valid, samples=samples_valid, n_bootstraps=3, return_df=True
    )
    assert isinstance(boot_results, pd.DataFrame)
    assert "x_0" in boot_results.columns

    # Test k-fold with fresh instance
    loc_kfold = Locator(
        {
            "out": os.path.join(temp_dir, "kfold"),
            "sample_data": coords_df_valid,
            "max_epochs": 2,
            "patience": 1,
            "keras_verbose": 0,
            "min_mac": 0,
            "impute_missing": True,
            "batch_size": 16,
        }
    )
    kfold_results = loc_kfold.run_k_fold_holdouts(
        genotypes=genotypes_valid,
        samples=samples_valid,
        k=3,
        return_df=True,
        verbose=False,
    )
    assert isinstance(kfold_results, pd.DataFrame)
    assert "x_pred" in kfold_results.columns
    assert len(kfold_results) == len(samples_valid)


def test_sample_weighting(sample_data, temp_dir):
    """Test sample weighting examples from documentation."""
    genotypes, samples, coords_df = sample_data

    # Use only samples with coordinates
    valid_mask = coords_df["x"].notna()
    coords_df_valid = coords_df[valid_mask].copy()
    genotypes_valid = genotypes[:, valid_mask.values]
    samples_valid = samples[valid_mask.values]

    # Test KDE weighting
    loc_kde = Locator(
        {
            "out": os.path.join(temp_dir, "kde_weights"),
            "sample_data": coords_df_valid,
            "weight_samples": {
                "enabled": True,
                "method": "KD",
                "bandwidth": 5.0,  # Fixed bandwidth for testing
            },
            "max_epochs": 2,
            "patience": 1,
            "keras_verbose": 0,
        }
    )

    loc_kde.train(genotypes=genotypes_valid, samples=samples_valid)
    assert hasattr(loc_kde, "sample_weights")
    assert loc_kde.sample_weights is not None

    # Test histogram weighting
    loc_hist = Locator(
        {
            "out": os.path.join(temp_dir, "hist_weights"),
            "sample_data": coords_df_valid,
            "weight_samples": {
                "enabled": True,
                "method": "histogram",
                "xbins": 3,
                "ybins": 3,
            },
            "max_epochs": 2,
            "patience": 1,
            "keras_verbose": 0,
        }
    )

    loc_hist.train(genotypes=genotypes_valid, samples=samples_valid)
    assert hasattr(loc_hist, "sample_weights")


def test_plotting_functions(sample_data, temp_dir):
    """Test key plotting functions work without errors."""
    genotypes, samples, coords_df = sample_data

    # Use only samples with coordinates
    valid_mask = coords_df["x"].notna()
    coords_df_valid = coords_df[valid_mask].copy()
    genotypes_valid = genotypes[:, valid_mask.values]
    samples_valid = samples[valid_mask.values]

    # Save coords for plot_error_summary
    coords_file = os.path.join(temp_dir, "coords.tsv")
    coords_df_valid.to_csv(coords_file, sep="\t", index=False)

    loc = Locator(
        {
            "out": os.path.join(temp_dir, "plot_test"),
            "sample_data": coords_df_valid,
            "max_epochs": 2,
            "patience": 1,
            "keras_verbose": 0,
        }
    )

    # Generate some predictions
    jack_preds = loc.run_jacknife(
        genotypes=genotypes_valid,
        samples=samples_valid,
        prop=0.3,  # This will create ~3 replicates (1/0.3)
        return_df=True,
    )

    # Test plot_predictions
    plot_predictions(
        jack_preds,
        loc,
        os.path.join(temp_dir, "test_preds"),
        n_samples=3,
        plot_map=False,
        show=False,
    )
    assert os.path.exists(os.path.join(temp_dir, "test_preds_predictions.pdf"))

    # Generate holdout predictions for error plot
    holdout_preds = loc.run_k_fold_holdouts(
        genotypes=genotypes_valid,
        samples=samples_valid,
        k=3,
        return_df=True,
        verbose=False,
    )

    # Test plot_error_summary
    plot_error_summary(
        holdout_preds,
        coords_file,
        os.path.join(temp_dir, "test_errors"),
        plot_map=False,
        use_geodesic=False,
        width=8,
        height=4,
        show=False,
    )
    assert os.path.exists(os.path.join(temp_dir, "test_errors_error_summary.png"))


def test_indexset_data_pipeline(sample_data):
    """Test IndexSet example from data pipeline docs."""
    genotypes, samples, coords_df = sample_data
    n_samples = len(samples)

    # Create custom splits
    index_set = IndexSet.random_split(
        n=n_samples, splits={"train": 0.7, "val": 0.15, "test": 0.15}, seed=42
    )

    # Verify splits sum to total
    total = len(index_set.train) + len(index_set.val) + len(index_set.test)
    assert total == n_samples

    # Verify no overlap
    all_indices = np.concatenate([index_set.train, index_set.val, index_set.test])
    assert len(np.unique(all_indices)) == n_samples

    # Test data access without copying
    train_data = genotypes[:, index_set.train]
    assert train_data.shape[1] == len(index_set.train)


def test_model_persistence(sample_data, temp_dir):
    """Test model saving and loading from documentation."""
    genotypes, samples, coords_df = sample_data

    # Use all data including NA samples (original data has some NA samples)
    # Train and save model
    loc1 = Locator(
        {
            "out": os.path.join(temp_dir, "model1"),
            "sample_data": coords_df,
            "max_epochs": 2,
            "patience": 1,
            "keras_verbose": 0,
            "min_mac": 0,
            "impute_missing": True,
            "batch_size": 16,
        }
    )

    loc1.train(genotypes=genotypes, samples=samples)
    model_path = f"{loc1.config['out']}.weights.h5"
    assert os.path.exists(model_path)

    # Load in new session
    loc2 = Locator({"out": os.path.join(temp_dir, "model2"), "sample_data": coords_df})

    # Make predictions with loaded model using predict_from_weights
    # This will predict on the NA samples
    preds = loc2.predict_from_weights(
        weights_path=model_path, genotypes=genotypes, samples=samples, return_df=True
    )

    # The predict_from_weights method loads metadata internally
    # We can verify it was loaded by checking normalization params
    assert hasattr(loc2, "meanlong")
    assert hasattr(loc2, "sdlong")

    # Should have predictions for NA samples
    assert len(preds) > 0
    # Check that predictions are for NA samples
    na_samples = coords_df[coords_df["x"].isna()]["sampleID"].tolist()
    assert all(sid in na_samples for sid in preds["sampleID"].tolist())


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
