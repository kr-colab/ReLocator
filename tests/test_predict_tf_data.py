"""Test predict() method with tf.data pipeline"""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import Mock, patch

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
            genotypes=genotypes,
            samples=samples,
            return_df=True
        )
        
        # Verify results
        assert isinstance(predictions, pd.DataFrame)
        assert 'sampleID' in predictions.columns
        assert 'x' in predictions.columns
        assert 'y' in predictions.columns
        
        # Should predict on samples without coordinates
        na_samples = samples[np.isnan(coords[:, 0])]
        assert len(predictions) == len(na_samples)
        assert all(sid in na_samples for sid in predictions['sampleID'])
    
    def test_predict_with_custom_indices(self, genotype_data, basic_config):
        """Test predict() with custom indices"""
        genotypes, samples, _, _, _ = genotype_data
        
        # Create locator and train
        locator = Locator(basic_config)
        locator.train(genotypes=genotypes, samples=samples)
        
        # Predict on specific samples (first 10)
        custom_indices = np.arange(10)
        predictions = locator.predict(
            genotypes=genotypes,
            samples=samples,
            indices=custom_indices,
            return_df=True
        )
        
        # Verify results
        assert len(predictions) == 10
        expected_samples = samples[custom_indices]
        assert all(predictions['sampleID'] == expected_samples)
    
    def test_predict_with_site_order(self, genotype_data, basic_config):
        """Test predict() with site_order for bootstrap/jacknife"""
        genotypes, samples, _, _, n_snps = genotype_data
        
        # Create locator and train with site_order
        locator = Locator(basic_config)
        
        # Create site order (subset of SNPs) 
        # In real bootstrap/jacknife, this happens during training
        n_sites = genotypes.shape[0]
        site_order = np.random.choice(n_sites, n_sites, replace=True)  # Bootstrap resampling
        
        # Train with site_order
        locator.train(genotypes=genotypes, samples=samples, site_order=site_order)
        
        # Predict with same site_order
        predictions = locator.predict(
            genotypes=genotypes,
            samples=samples,
            site_order=site_order,
            return_df=True
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
        if not hasattr(locator, 'predgen') or locator.predgen is None:
            pytest.skip("No samples without coordinates to predict")
        
        # Test with old approach (should warn)
        with pytest.warns(DeprecationWarning, match="deprecated"):
            predictions = locator.predict(
                prediction_genotypes=locator.predgen,
                return_df=True
            )
        
        # Should still work
        assert isinstance(predictions, pd.DataFrame)
    
    @patch('locator.data.make_tf_dataset')
    def test_predict_uses_make_tf_dataset(self, mock_make_tf_dataset, genotype_data, basic_config):
        """Test that predict() calls make_tf_dataset for tf.data pipeline"""
        genotypes, samples, _, _, _ = genotype_data
        
        # Set up mock to return a mock dataset
        mock_dataset = Mock()
        mock_make_tf_dataset.return_value = mock_dataset
        
        # Create locator and train
        locator = Locator(basic_config)
        locator.train(genotypes=genotypes, samples=samples)
        
        # Mock model predict to avoid actual prediction
        locator.model.predict = Mock(return_value=np.random.randn(5, 2))
        
        # Call predict with new approach
        locator.predict(
            genotypes=genotypes,
            samples=samples,
            return_df=True
        )
        
        # Verify make_tf_dataset was called
        assert mock_make_tf_dataset.called
        call_kwargs = mock_make_tf_dataset.call_args[1]
        
        # Check parameters
        assert 'genotypes' in call_kwargs
        assert 'index_set' in call_kwargs
        assert call_kwargs['split'] == 'predict'
        assert call_kwargs['training'] is False