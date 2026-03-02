"""Test examples from documentation to ensure they remain functional.

This module tests code snippets from the documentation to prevent drift
between documented examples and actual API behavior.
"""

import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import tensorflow as tf

from locator import EnsembleLocator, Locator
from locator.data import IndexSet, filter_snps, make_tf_dataset, normalize_locs
from locator.plotting import plot_error_summary, plot_predictions, plot_sample_weights

# Skip plotting tests if cartopy not available
try:
    import cartopy

    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False

# Skip parallel tests if Ray not available
try:
    import ray

    from locator.parallel import (
        parallel_holdouts,
        parallel_k_fold_holdouts,
        parallel_windows_holdouts,
    )

    HAS_RAY = True
except ImportError:
    HAS_RAY = False


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    np.random.seed(42)
    n_samples = 50
    n_snps = 100

    # Vectorized genotype generation with custom probabilities
    # ~30% hom ref, ~30% het, ~40% hom alt
    allele_counts = np.random.choice(
        [0, 1, 2], size=(n_snps, n_samples), p=[0.3, 0.3, 0.4]
    )
    gt_array = np.zeros((n_snps, n_samples, 2), dtype=int)
    gt_array[:, :, 0] = (allele_counts >= 1).astype(int)
    gt_array[:, :, 1] = (allele_counts >= 2).astype(int)

    # Add a small amount of missing data
    gt_array[0:2, 0:2, :] = -1

    # Import allel and create GenotypeArray
    import allel

    genotypes = allel.GenotypeArray(gt_array)

    # Create sample IDs as numpy array
    samples = np.array([f"sample_{i:03d}" for i in range(n_samples)])

    # Create coordinate data with some NAs
    coords_df = pd.DataFrame(
        {
            "sampleID": samples,
            "x": np.random.uniform(-120, -110, n_samples),
            "y": np.random.uniform(30, 40, n_samples),
        }
    )
    # Make some samples have NA coordinates
    coords_df.loc[45:49, ["x", "y"]] = np.nan

    return genotypes, samples, coords_df


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test outputs."""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    shutil.rmtree(temp_path)


class TestBasicExamples:
    """Test basic usage examples from documentation."""

    def test_basic_usage(self, sample_data, temp_dir):
        """Test basic usage example from docs."""
        genotypes, samples, coords_df = sample_data

        # Save sample data to file
        sample_file = os.path.join(temp_dir, "samples.txt")
        coords_df.to_csv(sample_file, sep="\t", index=False)

        # Example from docs/source/examples.rst - Basic Usage
        loc = Locator(
            {
                "out": os.path.join(temp_dir, "my_analysis"),
                "sample_data": sample_file,
                "max_epochs": 2,  # Quick test
                "patience": 1,
            }
        )

        # Modified to use actual genotype array instead of loading
        loc.train(genotypes=genotypes, samples=samples)

        # Make predictions
        predictions = loc.predict(return_df=True)

        # Verify predictions
        assert isinstance(predictions, pd.DataFrame)
        assert "sampleID" in predictions.columns
        assert "x" in predictions.columns
        assert "y" in predictions.columns
        assert len(predictions) > 0

    def test_na_handling_separate_mode(self, sample_data, temp_dir):
        """Test NA handling example with separate mode."""
        genotypes, samples, coords_df = sample_data

        # Example from docs - NA handling with 'separate' mode
        loc = Locator(
            {
                "out": os.path.join(temp_dir, "na_example"),
                "sample_data": coords_df,
                "na_action": "separate",  # Default
                "max_epochs": 2,
                "patience": 1,
            }
        )

        # Check data (should report NA samples)
        loc.check_data(genotypes, samples, verbose=False)

        # Train on samples with coordinates
        loc.train(genotypes=genotypes, samples=samples)
        predictions = loc.predict(return_df=True)

        # Should have predictions for NA samples
        na_samples = coords_df[coords_df["x"].isna()]["sampleID"].tolist()
        pred_samples = predictions["sampleID"].tolist()
        assert any(s in pred_samples for s in na_samples)

    def test_na_handling_exclude_mode(self, sample_data, temp_dir):
        """Test NA handling example with exclude mode."""
        genotypes, samples, coords_df = sample_data

        # Example from docs - 'exclude' mode
        loc_exclude = Locator(
            {
                "out": os.path.join(temp_dir, "exclude_example"),
                "sample_data": coords_df,
                "na_action": "exclude",
                "max_epochs": 2,
                "patience": 1,
            }
        )

        # Only samples with coordinates will be used
        loc_exclude.train(genotypes=genotypes, samples=samples)

        # Verify only non-NA samples used
        # With tf.data pipeline, check trainlocs instead of traingen
        assert hasattr(loc_exclude, "trainlocs")
        n_known = coords_df["x"].notna().sum()
        expected_train_size = int(n_known * loc_exclude.config.get("train_split", 0.9))
        assert abs(len(loc_exclude.trainlocs) - expected_train_size) <= 2

    def test_na_handling_fail_mode(self, sample_data, temp_dir):
        """Test NA handling example with fail mode."""
        genotypes, samples, coords_df = sample_data

        # Example from docs - 'fail' mode
        loc_strict = Locator(
            {
                "out": os.path.join(temp_dir, "strict_example"),
                "sample_data": coords_df,
                "na_action": "fail",
                "max_epochs": 2,
            }
        )

        # This should raise an error
        with pytest.raises(ValueError) as excinfo:
            loc_strict.train(genotypes=genotypes, samples=samples)
        assert "samples without coordinates" in str(excinfo.value)


class TestDataPipelineExamples:
    """Test data pipeline examples from documentation."""

    def test_indexset_usage(self, sample_data):
        """Test IndexSet example from docs."""
        genotypes, samples, coords_df = sample_data
        n_samples = len(samples)

        # Example from docs - Custom splits with IndexSet
        index_set = IndexSet.random_split(
            n=n_samples, splits={"train": 0.7, "val": 0.15, "test": 0.15}
        )

        # Verify splits (allowing for rounding)
        assert abs(len(index_set.train) - int(n_samples * 0.7)) <= 1
        assert abs(len(index_set.val) - int(n_samples * 0.15)) <= 1
        assert (
            len(index_set.train) + len(index_set.val) + len(index_set.test) == n_samples
        )

        # Access data without copying
        train_genotypes = genotypes[:, index_set.train]
        assert train_genotypes.shape[1] == len(index_set.train)

    def test_filter_snps_example(self, sample_data):
        """Test SNP filtering example from docs."""
        genotypes, samples, coords_df = sample_data

        # genotypes is already a GenotypeArray from our fixture
        # Example from docs - Preprocess with tracking
        filtered_geno, filter_stats = filter_snps(
            genotypes, min_mac=2, max_snps=50, impute=True
        )

        # Verify filtering
        assert filter_stats.n_snps_filtered <= filter_stats.n_snps_original
        assert filter_stats.n_snps_filtered <= 50
        assert filtered_geno.shape[0] <= genotypes.shape[0]

    def test_normalize_locs_example(self, sample_data):
        """Test coordinate normalization example from docs."""
        genotypes, samples, coords_df = sample_data

        # Get coordinates
        coords = coords_df[["x", "y"]].values
        valid_coords = coords[~np.isnan(coords[:, 0])]

        # Example from docs - Normalize coordinates
        # normalize_locs returns 6 values: meanlong, sdlong, meanlat, sdlat, unnormedlocs, normedlocs
        meanlong, sdlong, meanlat, sdlat, unnormed_locs, normed_locs = normalize_locs(
            valid_coords
        )

        # Verify normalization
        assert normed_locs.shape == valid_coords.shape
        assert np.abs(normed_locs.mean(axis=0)).max() < 0.01  # Near zero mean
        assert np.abs(normed_locs.std(axis=0) - 1.0).max() < 0.01  # Unit variance

        # Test data integrity
        np.testing.assert_allclose(unnormed_locs, valid_coords, rtol=1e-5)


class TestWeightingExamples:
    """Test sample weighting examples from documentation."""

    def test_kde_weighting(self, sample_data, temp_dir):
        """Test KDE weighting example from docs."""
        genotypes, samples, coords_df = sample_data

        # Remove NA samples for this test
        valid_mask = coords_df["x"].notna()
        coords_df_valid = coords_df[valid_mask].copy()
        genotypes_valid = genotypes[:, valid_mask.values]
        samples_valid = samples[valid_mask.values]

        # Example from docs - KDE weighting
        loc = Locator(
            {
                "out": os.path.join(temp_dir, "weighted_analysis"),
                "sample_data": coords_df_valid,
                "weight_samples": {
                    "enabled": True,
                    "method": "KD",
                    "bandwidth": None,  # Auto-calculate
                },
                "max_epochs": 2,
                "patience": 1,
            }
        )

        # Train with weights
        loc.train(genotypes=genotypes_valid, samples=samples_valid)

        # Verify weights were calculated
        assert hasattr(loc, "sample_weights")
        assert loc.sample_weights is not None

    def test_histogram_weighting(self, sample_data, temp_dir):
        """Test histogram binning weighting example from docs."""
        genotypes, samples, coords_df = sample_data

        # Remove NA samples
        valid_mask = coords_df["x"].notna()
        coords_df_valid = coords_df[valid_mask].copy()
        genotypes_valid = genotypes[:, valid_mask.values]
        samples_valid = samples[valid_mask.values]

        # Example from docs - Histogram weighting
        loc_hist = Locator(
            {
                "out": os.path.join(temp_dir, "hist_weighted"),
                "sample_data": coords_df_valid,
                "weight_samples": {
                    "enabled": True,
                    "method": "histogram",
                    "xbins": 5,  # Fewer bins for small test data
                    "ybins": 5,
                },
                "max_epochs": 2,
                "patience": 1,
            }
        )

        loc_hist.train(genotypes=genotypes_valid, samples=samples_valid)
        assert hasattr(loc_hist, "sample_weights")


class TestAnalysisExamples:
    """Test analysis method examples from documentation."""

    def test_jacknife_analysis(self, sample_data, temp_dir):
        """Test jacknife analysis example."""
        genotypes, samples, coords_df = sample_data

        # Remove NA samples
        valid_mask = coords_df["x"].notna()
        coords_df_valid = coords_df[valid_mask].copy()
        genotypes_valid = genotypes[:, valid_mask.values]
        samples_valid = samples[valid_mask.values]

        loc = Locator(
            {
                "out": os.path.join(temp_dir, "jacknife_test"),
                "sample_data": coords_df_valid,
                "max_epochs": 2,
                "patience": 1,
            }
        )

        # Example from docs - Jacknife analysis
        jacknife_results = loc.run_jacknife(
            genotypes=genotypes_valid,
            samples=samples_valid,
            prop=0.1,  # This will create 10 replicates (1/0.1)
            return_df=True,
        )

        # Verify results
        assert isinstance(jacknife_results, pd.DataFrame)
        assert "sampleID" in jacknife_results.columns
        assert any(col.startswith("x_") for col in jacknife_results.columns)
        assert any(col.startswith("y_") for col in jacknife_results.columns)

    def test_bootstrap_analysis(self, sample_data, temp_dir):
        """Test bootstrap analysis example."""
        genotypes, samples, coords_df = sample_data

        # Remove NA samples
        valid_mask = coords_df["x"].notna()
        coords_df_valid = coords_df[valid_mask].copy()
        genotypes_valid = genotypes[:, valid_mask.values]
        samples_valid = samples[valid_mask.values]

        loc = Locator(
            {
                "out": os.path.join(temp_dir, "bootstrap_test"),
                "sample_data": coords_df_valid,
                "max_epochs": 2,
                "patience": 1,
            }
        )

        # Example from docs - Bootstrap analysis
        bootstrap_results = loc.run_bootstraps(
            genotypes=genotypes_valid,
            samples=samples_valid,
            n_bootstraps=3,  # Fewer for testing
            return_df=True,
        )

        assert isinstance(bootstrap_results, pd.DataFrame)
        assert len(bootstrap_results) == len(samples_valid)

    def test_kfold_holdouts(self, sample_data, temp_dir):
        """Test k-fold cross-validation example."""
        genotypes, samples, coords_df = sample_data

        # Remove NA samples
        valid_mask = coords_df["x"].notna()
        coords_df_valid = coords_df[valid_mask].copy()
        genotypes_valid = genotypes[:, valid_mask.values]
        samples_valid = samples[valid_mask.values]

        loc = Locator(
            {
                "out": os.path.join(temp_dir, "kfold_test"),
                "sample_data": coords_df_valid,
                "max_epochs": 2,
                "patience": 1,
                "keras_verbose": 0,
            }
        )

        # Example from docs - K-fold CV
        kfold_results = loc.run_k_fold_holdouts(
            genotypes=genotypes_valid,
            samples=samples_valid,
            k=3,  # Fewer folds for testing
            return_df=True,
            verbose=False,
        )

        assert isinstance(kfold_results, pd.DataFrame)
        assert "x_pred" in kfold_results.columns
        assert "y_pred" in kfold_results.columns
        # Each sample should appear exactly once
        assert len(kfold_results) == len(samples_valid)


class TestPlottingExamples:
    """Test plotting examples from documentation."""

    def test_plot_predictions_basic(self, sample_data, temp_dir):
        """Test basic plot_predictions example."""
        genotypes, samples, coords_df = sample_data

        # Remove NA samples
        valid_mask = coords_df["x"].notna()
        coords_df_valid = coords_df[valid_mask].copy()
        genotypes_valid = genotypes[:, valid_mask.values]
        samples_valid = samples[valid_mask.values]

        loc = Locator(
            {
                "out": os.path.join(temp_dir, "plot_test"),
                "sample_data": coords_df_valid,
                "max_epochs": 2,
                "patience": 1,
            }
        )

        # Generate predictions
        jack_preds = loc.run_jacknife(
            genotypes=genotypes_valid,
            samples=samples_valid,
            prop=0.2,  # This will create 5 replicates (1/0.2)
            return_df=True,
        )

        # Example from docs - Plot predictions
        plot_predictions(
            jack_preds,
            loc,
            os.path.join(temp_dir, "test_predictions"),
            n_samples=3,
            plot_map=False,  # Don't require cartopy
            show=False,  # Don't display
        )

        # Check output file exists
        assert os.path.exists(os.path.join(temp_dir, "test_predictions_predictions.pdf"))

    def test_plot_error_summary(self, sample_data, temp_dir):
        """Test plot_error_summary example."""
        genotypes, samples, coords_df = sample_data

        # Remove NA samples
        valid_mask = coords_df["x"].notna()
        coords_df_valid = coords_df[valid_mask].copy()
        genotypes_valid = genotypes[:, valid_mask.values]
        samples_valid = samples[valid_mask.values]

        # Save coords to file
        coords_file = os.path.join(temp_dir, "coords.tsv")
        coords_df_valid.to_csv(coords_file, sep="\t", index=False)

        loc = Locator(
            {
                "out": os.path.join(temp_dir, "error_test"),
                "sample_data": coords_df_valid,
                "max_epochs": 2,
                "patience": 1,
                "keras_verbose": 0,
            }
        )

        # Generate predictions
        kfold_preds = loc.run_k_fold_holdouts(
            genotypes=genotypes_valid,
            samples=samples_valid,
            k=3,
            return_df=True,
            verbose=False,
        )

        # Example from docs - Error summary
        plot_error_summary(
            kfold_preds,
            coords_file,
            os.path.join(temp_dir, "error_summary"),
            plot_map=False,  # Don't require cartopy
            use_geodesic=False,  # Simpler for testing
            width=10,
            height=5,
            show=False,
        )

        assert os.path.exists(os.path.join(temp_dir, "error_summary_error_summary.png"))

    def test_plot_sample_weights(self, sample_data, temp_dir):
        """Test plot_sample_weights example."""
        genotypes, samples, coords_df = sample_data

        # Remove NA samples
        valid_mask = coords_df["x"].notna()
        coords_df_valid = coords_df[valid_mask].copy()
        genotypes_valid = genotypes[:, valid_mask.values]
        samples_valid = samples[valid_mask.values]

        # Train with weights
        loc = Locator(
            {
                "out": os.path.join(temp_dir, "weights_test"),
                "sample_data": coords_df_valid,
                "weight_samples": {
                    "enabled": True,
                    "method": "histogram",
                    "xbins": 3,
                    "ybins": 3,
                },
                "max_epochs": 2,
                "patience": 1,
            }
        )

        loc.train(genotypes=genotypes_valid, samples=samples_valid)

        # Example from docs - Plot weights
        plot_sample_weights(
            loc, os.path.join(temp_dir, "sample_weights"), plot_map=False, show=False
        )

        assert os.path.exists(
            os.path.join(temp_dir, "sample_weights_sample_weights.png")
        )


@pytest.mark.skipif(not HAS_RAY, reason="Ray not installed")
class TestParallelExamples:
    """Test parallel analysis examples from documentation."""

    def test_parallel_kfold(self, sample_data, temp_dir):
        """Test parallel k-fold example from docs."""
        genotypes, samples, coords_df = sample_data

        # Remove NA samples
        valid_mask = coords_df["x"].notna()
        coords_df_valid = coords_df[valid_mask].copy()
        genotypes_valid = genotypes[:, valid_mask.values]
        samples_valid = samples[valid_mask.values]

        # Initialize Ray if needed
        if not ray.is_initialized():
            ray.init(num_cpus=2, num_gpus=0)  # CPU only for testing

        try:
            loc = Locator(
                {
                    "out": os.path.join(temp_dir, "parallel_test"),
                    "sample_data": coords_df_valid,
                    "max_epochs": 2,
                    "patience": 1,
                    "keras_verbose": 0,
                    "disable_gpu": True,  # CPU for testing
                }
            )

            # Example from docs - Parallel k-fold (modified for CPU)
            predictions = parallel_k_fold_holdouts(
                loc,
                genotypes_valid,
                samples_valid,
                k=3,
                gpu_ids=[],  # Empty = CPU only
                gpu_fraction=0.0,  # CPU
                return_df=True,
                verbose=False,
            )

            assert isinstance(predictions, pd.DataFrame)
            assert len(predictions) == len(samples_valid)

        finally:
            ray.shutdown()


class TestGPUExamples:
    """Test GPU optimization examples from documentation."""

    def test_gpu_config_examples(self, temp_dir):
        """Test GPU configuration examples."""
        # Example 1: Default GPU optimization
        loc = Locator(
            {
                "out": os.path.join(temp_dir, "gpu_optimized"),
                "sample_data": pd.DataFrame(
                    {"sampleID": ["A", "B"], "x": [1, 2], "y": [3, 4]}
                ),
            }
        )
        # GPU optimizations are enabled by default if GPU is available
        # In CI without GPU, use_mixed_precision will be False
        if tf.config.list_physical_devices("GPU"):
            assert loc.config.get("use_mixed_precision", True) is True
        else:
            # When no GPU is available, mixed precision is disabled
            assert loc.config.get("use_mixed_precision", False) is False

        # Example 2: Memory-constrained GPU
        loc_constrained = Locator(
            {
                "out": os.path.join(temp_dir, "memory_limited"),
                "sample_data": pd.DataFrame(
                    {"sampleID": ["A", "B"], "x": [1, 2], "y": [3, 4]}
                ),
                "gpu_batch_size": 64,
                "gradient_accumulation_steps": 4,
            }
        )
        assert loc_constrained.config["gpu_batch_size"] == 64
        assert loc_constrained.config["gradient_accumulation_steps"] == 4


class TestEnsembleExamples:
    """Test ensemble examples from documentation."""

    def test_ensemble_basic(self, sample_data, temp_dir):
        """Test basic ensemble example."""
        genotypes, samples, coords_df = sample_data

        # Remove NA samples
        valid_mask = coords_df["x"].notna()
        coords_df_valid = coords_df[valid_mask].copy()
        genotypes_valid = genotypes[:, valid_mask.values]
        samples_valid = samples[valid_mask.values]

        # Example from docs - Ensemble
        ensemble = EnsembleLocator(
            base_config={
                "out": os.path.join(temp_dir, "ensemble_analysis"),
                "sample_data": coords_df_valid,
                "max_epochs": 2,
                "patience": 1,
                "keras_verbose": 0,
            },
            k_folds=3,  # Fewer folds for testing
        )

        # Train ensemble
        ensemble.train(genotypes=genotypes_valid, samples=samples_valid)

        # Make predictions
        ensemble_predictions = ensemble.predict()

        assert ensemble_predictions is not None
        assert hasattr(ensemble, "models")
        assert len(ensemble.models) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
