"""Tests for ensemble mixin functionality"""

import numpy as np
import pandas as pd
import pytest

from locator import Locator
from locator.data import IndexSet


class TestEnsembleMixin:
    """Test the EnsembleMixin functionality."""

    def test_k_fold_split(self):
        """Test IndexSet.k_fold_split method."""
        n_samples = 100
        k = 5

        # Test basic k-fold splitting
        fold_sets = IndexSet.k_fold_split(n=n_samples, k=k, seed=42)

        # Verify we get k folds
        assert len(fold_sets) == k

        # Verify each fold has the correct structure
        for fold_set in fold_sets:
            assert "train" in fold_set.indices
            assert "test" in fold_set.indices
            assert fold_set.total_samples == n_samples

        # Verify all samples are covered exactly once as test
        all_test_indices = []
        for fold_set in fold_sets:
            all_test_indices.extend(fold_set.test.tolist())

        assert len(all_test_indices) == n_samples
        assert len(set(all_test_indices)) == n_samples  # No duplicates

        # Verify train/test splits don't overlap within a fold
        for fold_set in fold_sets:
            assert len(np.intersect1d(fold_set.train, fold_set.test)) == 0

    def test_k_fold_split_with_na_mask(self):
        """Test k_fold_split with NA mask."""
        n_samples = 100
        k = 5

        # Create NA mask - exclude 10 samples
        na_mask = np.zeros(n_samples, dtype=bool)
        na_mask[:10] = True

        fold_sets = IndexSet.k_fold_split(n=n_samples, k=k, seed=42, na_mask=na_mask)

        # Verify NA samples are not in any fold
        for fold_set in fold_sets:
            assert not np.any(np.isin(fold_set.train, np.where(na_mask)[0]))
            assert not np.any(np.isin(fold_set.test, np.where(na_mask)[0]))

        # Verify only non-NA samples are used
        all_indices = []
        for fold_set in fold_sets:
            all_indices.extend(fold_set.train.tolist())
            all_indices.extend(fold_set.test.tolist())

        unique_indices = set(all_indices)
        expected_indices = set(np.where(~na_mask)[0])
        assert unique_indices == expected_indices

    def test_create_ensemble_folds(self):
        """Test create_ensemble_folds method."""
        # Create test data
        n_samples = 50
        samples = np.array([f"sample_{i:03d}" for i in range(n_samples)])

        # Create coordinate data with some NA values
        coords_df = pd.DataFrame(
            {
                "sampleID": samples,
                "x": np.random.uniform(-120, -70, n_samples),
                "y": np.random.uniform(30, 50, n_samples),
            }
        )
        coords_df.loc[:5, ["x", "y"]] = np.nan  # First 6 samples have NA coords

        # Create locator instance
        config = {"sample_data": coords_df, "na_action": "separate"}
        locator = Locator(config)

        # Mock genotypes (simplified)
        genotypes = np.random.randint(0, 3, size=(100, n_samples))

        # Test create_ensemble_folds
        fold_info = locator.create_ensemble_folds(
            genotypes=genotypes, samples=samples, k=3, na_action="separate"
        )

        # Verify structure
        assert "index_sets" in fold_info
        assert "fold_indices" in fold_info
        assert "sample_status" in fold_info

        # Verify we have 3 folds
        assert len(fold_info["index_sets"]) == 3
        assert len(fold_info["fold_indices"]) == 3

        # Verify sample status
        assert fold_info["sample_status"]["n_known"] == 44  # 50 - 6 NA
        assert fold_info["sample_status"]["n_na"] == 6

        # Verify NA samples are in pred set for each fold
        na_indices = np.arange(6)  # First 6 samples
        for fold_idx in fold_info["fold_indices"]:
            pred_idx = fold_info["fold_indices"][fold_idx]["pred"]
            assert np.array_equal(pred_idx, na_indices)

    def test_create_ensemble_folds_with_training_indices(self):
        """Test create_ensemble_folds with training_set_indices."""
        # Create test data
        n_samples = 50
        samples = np.array([f"sample_{i:03d}" for i in range(n_samples)])

        coords_df = pd.DataFrame(
            {
                "sampleID": samples,
                "x": np.random.uniform(-120, -70, n_samples),
                "y": np.random.uniform(30, 50, n_samples),
            }
        )

        config = {"sample_data": coords_df, "na_action": "separate"}
        locator = Locator(config)

        genotypes = np.random.randint(0, 3, size=(100, n_samples))

        # Only use first 30 samples for training
        training_indices = np.arange(30)

        fold_info = locator.create_ensemble_folds(
            genotypes=genotypes,
            samples=samples,
            k=3,
            training_set_indices=training_indices,
            na_action="separate",
        )

        # Verify only training indices are used in folds
        for index_set in fold_info["index_sets"]:
            all_fold_indices = np.concatenate([index_set.train, index_set.test])
            assert np.all(np.isin(all_fold_indices, training_indices))

    def test_ensemble_normalization_params(self):
        """Test that normalization parameters are properly computed and stored."""
        # Create simple test data
        n_samples = 30
        samples = np.array([f"sample_{i:03d}" for i in range(n_samples)])

        # Create coordinates in a known range
        x_coords = np.linspace(-100, -80, n_samples)
        y_coords = np.linspace(30, 40, n_samples)

        coords_df = pd.DataFrame({"sampleID": samples, "x": x_coords, "y": y_coords})

        config = {
            "sample_data": coords_df,
            "na_action": "exclude",
            "max_epochs": 1,  # Just test initialization
            "keras_verbose": 0,
        }
        locator = Locator(config)

        # Test _apply_normalization helper
        locs = coords_df[["x", "y"]].values
        norm_params = {"meanlong": -90, "sdlong": 10, "meanlat": 35, "sdlat": 5}

        normalized = locator._apply_normalization(locs, norm_params)

        # Verify normalization
        expected_x = (x_coords - (-90)) / 10
        expected_y = (y_coords - 35) / 5

        np.testing.assert_allclose(normalized[:, 0], expected_x, rtol=1e-5)
        np.testing.assert_allclose(normalized[:, 1], expected_y, rtol=1e-5)

    def test_filter_genotypes(self):
        """Test _filter_genotypes helper method exists and is callable."""
        config = {"min_mac": 2, "max_SNPs": 100, "impute_missing": False}
        locator = Locator(config)

        # Just verify the method exists and is callable
        assert hasattr(locator, "_filter_genotypes")
        assert callable(locator._filter_genotypes)

        # The actual filtering is tested extensively in test_filters.py
