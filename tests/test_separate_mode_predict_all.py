"""Test that 'separate' mode predicts on all samples."""

import allel
import numpy as np
import pandas as pd
import pytest
from conftest import make_test_genotypes

from locator import Locator


class TestSeparateModePredictAll:
    """Test that 'separate' mode predicts on all samples, not just NA samples."""

    def create_test_data(self, n_samples=20, n_known=15):
        """Create test genotype and sample data with some NA coordinates."""
        return make_test_genotypes(
            n_snps=100, n_samples=n_samples, n_known=n_known, seed=42
        )

    def test_separate_mode_predicts_all_samples(self, tmp_path):
        """Test that 'separate' mode predicts on all samples (both known and NA)."""
        # Create test data with 20 samples (15 known, 5 NA)
        genotypes, samples, sample_df = self.create_test_data(n_samples=20, n_known=15)

        # Initialize Locator with 'separate' mode
        locator = Locator(
            {
                "sample_data": sample_df,
                "na_action": "separate",
                "keras_verbose": 0,
                "max_epochs": 2,
                "patience": 1,
                "out": str(tmp_path / "test_separate_all"),
            }
        )

        # Train the model
        history = locator.train(genotypes=genotypes, samples=samples)
        assert history is not None

        # Get predictions
        predictions = locator.predict(return_df=True, save_preds_to_disk=False)

        # Check that we got predictions for ALL samples
        assert len(predictions) == 20, (
            f"Expected 20 predictions but got {len(predictions)}"
        )

        # Check that predictions include all sample IDs
        pred_sample_ids = set(predictions["sampleID"])
        all_sample_ids = set(samples)
        assert pred_sample_ids == all_sample_ids, (
            "Predictions should include all samples"
        )

    def test_separate_mode_with_no_na_samples(self, tmp_path):
        """Test that 'separate' mode works correctly when all samples have coordinates."""
        # Create test data with all samples having known coordinates
        genotypes, samples, sample_df = self.create_test_data(n_samples=10, n_known=10)

        # Initialize Locator with 'separate' mode
        locator = Locator(
            {
                "sample_data": sample_df,
                "na_action": "separate",
                "keras_verbose": 0,
                "max_epochs": 2,
                "patience": 1,
                "out": str(tmp_path / "test_separate_no_na"),
            }
        )

        # Train the model
        history = locator.train(genotypes=genotypes, samples=samples)
        assert history is not None

        # Get predictions
        predictions = locator.predict(return_df=True, save_preds_to_disk=False)

        # Check that we still get predictions for all samples
        assert len(predictions) == 10, (
            f"Expected 10 predictions but got {len(predictions)}"
        )

    def test_exclude_mode_only_predicts_na(self, tmp_path):
        """Test that 'exclude' mode excludes NA samples from both training and prediction."""
        # Create test data with 20 samples (15 known, 5 NA)
        genotypes, samples, sample_df = self.create_test_data(n_samples=20, n_known=15)

        # Initialize Locator with 'exclude' mode
        locator = Locator(
            {
                "sample_data": sample_df,
                "na_action": "exclude",
                "keras_verbose": 0,
                "max_epochs": 2,
                "patience": 1,
                "out": str(tmp_path / "test_exclude"),
            }
        )

        # Train the model
        history = locator.train(genotypes=genotypes, samples=samples)
        assert history is not None

        # In exclude mode, we've excluded NA samples from training,
        # so predgen should be empty
        assert locator.predgen.shape[0] == 0, "In exclude mode, predgen should be empty"

        # If we try to predict, we should get an empty result
        predictions = locator.predict(return_df=True, save_preds_to_disk=False)

        # In exclude mode, there are no samples to predict
        assert len(predictions) == 0, (
            f"Expected 0 predictions in exclude mode but got {len(predictions)}"
        )
