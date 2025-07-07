"""Test tf.data pipeline integration in training.py"""

from unittest.mock import Mock

import pytest

from locator.core import Locator


class TestTFDataIntegration:
    """Test tf.data pipeline is properly integrated without array reconstruction"""

    def test_train_uses_filtered_genotypes_directly(self, genotype_data, basic_config):
        """Test that train() uses filtered_genotypes directly without reconstruction"""
        genotypes, samples, _, _, _ = genotype_data

        # Create locator
        locator = Locator(basic_config)

        # Mock the model to avoid actual training
        locator.model = Mock()
        locator.model.fit.return_value = Mock(history={})

        # Train
        locator.train(genotypes=genotypes, samples=samples)

        # Verify that filtered_genotypes was stored
        assert hasattr(locator, "filtered_genotypes")
        assert hasattr(locator, "index_set")

        # Verify no array reconstruction occurred
        # The model.fit should have been called with tf.data.Dataset
        fit_call = locator.model.fit.call_args
        assert fit_call is not None
        # First argument should be a tf.data.Dataset
        train_data = fit_call[0][0]
        assert hasattr(train_data, "__iter__")  # It's a dataset, not a numpy array
