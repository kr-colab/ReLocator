"""Test predict() method with tf.data pipeline"""

import numpy as np
import pandas as pd
import pytest

from locator.core import Locator


class TestPredictTFData:
    """Test predict() method uses tf.data pipeline efficiently"""

    def test_predict_with_genotypes(self, genotype_data, basic_config):
        """Test predict() using genotypes parameter (tf.data approach)"""
        genotypes, samples, coords, _, _ = genotype_data

        # Create locator and train
        locator = Locator(basic_config)
        locator.train(genotypes=genotypes, samples=samples)

        # Test prediction using new tf.data approach
        predictions = locator.predict(
            genotypes=genotypes, samples=samples, return_df=True
        )

        # Verify results
        assert isinstance(predictions, pd.DataFrame)
        assert "sampleID" in predictions.columns
        assert "x" in predictions.columns
        assert "y" in predictions.columns

        # Should predict on samples without coordinates
        na_samples = samples[np.isnan(coords[:, 0])]
        assert len(predictions) == len(na_samples)
        assert all(sid in na_samples for sid in predictions["sampleID"])

    def test_predict_with_custom_indices(self, genotype_data, basic_config):
        """Test predict() with custom indices"""
        genotypes, samples, _, _, _ = genotype_data

        # Create locator and train
        locator = Locator(basic_config)
        locator.train(genotypes=genotypes, samples=samples)

        # Predict on specific samples (first 10)
        custom_indices = np.arange(10)
        predictions = locator.predict(
            genotypes=genotypes, samples=samples, indices=custom_indices, return_df=True
        )

        # Verify results
        assert len(predictions) == 10
        expected_samples = samples[custom_indices]
        assert all(predictions["sampleID"] == expected_samples)

    def test_predict_with_site_order(self, genotype_data, basic_config):
        """Test predict() with site_order for bootstrap/jacknife"""
        genotypes, samples, _, _, n_snps = genotype_data

        # Create locator and train with site_order
        locator = Locator(basic_config)

        # Create site order (subset of SNPs)
        # In real bootstrap/jacknife, this happens during training
        n_sites = genotypes.shape[0]
        site_order = np.random.choice(
            n_sites, n_sites, replace=True
        )  # Bootstrap resampling

        # Train with site_order
        locator.train(genotypes=genotypes, samples=samples, site_order=site_order)

        # Predict with same site_order
        predictions = locator.predict(
            genotypes=genotypes, samples=samples, site_order=site_order, return_df=True
        )

        # Should still return predictions
        assert isinstance(predictions, pd.DataFrame)
        assert len(predictions) > 0

    def test_backward_compatibility(self, genotype_data, basic_config):
        """Test old prediction_genotypes parameter still works with warning"""
        genotypes, samples, coords, _, _ = genotype_data

        # Create locator and train
        locator = Locator(basic_config)
        locator.train(genotypes=genotypes, samples=samples)

        # Only test if we have pred samples
        if not hasattr(locator, "predgen") or locator.predgen is None:
            pytest.skip("No samples without coordinates to predict")

        # Test with old approach (should warn)
        with pytest.warns(DeprecationWarning, match="deprecated"):
            predictions = locator.predict(
                prediction_genotypes=locator.predgen, return_df=True
            )

        # Should still work
        assert isinstance(predictions, pd.DataFrame)

    def test_predict_runs_inner_network_on_feature_array(
        self, genotype_data, basic_config
    ):
        """predict() runs the inner network on a sample-major feature array."""
        genotypes, samples, _, _, _ = genotype_data

        locator = Locator(basic_config)
        locator.train(genotypes=genotypes, samples=samples)

        # predict() unwraps IndexedGenotypeModel and feeds genotype features
        # (shape (n_predict, n_snps)) straight to the inner network.
        n_snps = locator.filtered_genotypes.shape[0]
        captured = {}
        inner = locator.model.inner

        def spy(x, *args, **kwargs):
            arr = np.asarray(x)
            captured["x"] = arr
            return np.random.randn(arr.shape[0], 2)

        inner.predict = spy
        locator.predict(genotypes=genotypes, samples=samples, return_df=True)

        assert "x" in captured
        assert captured["x"].ndim == 2
        assert captured["x"].shape[1] == n_snps
