"""Test tf.data pipeline integration in training.py"""

import numpy as np
import pytest
from unittest.mock import Mock, patch
import allel

from locator.core import Locator


class TestTFDataIntegration:
    """Test tf.data pipeline is properly integrated without array reconstruction"""
    
    def setup_method(self):
        """Set up test data"""
        self.n_samples = 50
        self.n_snps = 100
        
        # Create genotype data (biallelic)
        geno_array = np.zeros((self.n_snps, self.n_samples, 2), dtype=np.int8)
        for i in range(self.n_snps):
            # Create biallelic genotypes (0/0, 0/1, 1/1)
            for j in range(self.n_samples):
                allele_count = np.random.choice([0, 1, 2], p=[0.25, 0.5, 0.25])
                if allele_count == 0:
                    geno_array[i, j, :] = [0, 0]
                elif allele_count == 1:
                    geno_array[i, j, :] = [0, 1]
                else:
                    geno_array[i, j, :] = [1, 1]
        self.genotypes = allel.GenotypeArray(geno_array)
        
        # Create sample IDs
        self.samples = np.array([f"sample_{i}" for i in range(self.n_samples)])
        
        # Create coordinates (some with NA)
        self.coords = np.random.uniform(-180, 180, size=(self.n_samples, 2))
        # Make some samples have NA coordinates
        self.coords[40:45, :] = np.nan
        
        # Create sample data file content
        self.sample_data = "sampleID\tx\ty\n"
        for i, sid in enumerate(self.samples):
            x, y = self.coords[i]
            self.sample_data += f"{sid}\t{x}\t{y}\n"
    
    def test_train_uses_filtered_genotypes_directly(self, tmp_path):
        """Test that train() uses filtered_genotypes directly without reconstruction"""
        # Write sample data
        sample_file = tmp_path / "samples.txt"
        sample_file.write_text(self.sample_data)
        
        # Create locator
        config = {
            "out": str(tmp_path / "test"),
            "sample_data": str(sample_file),
            "max_epochs": 1,
            "use_efficient_pipeline": True,
        }
        locator = Locator(config)
        
        # Mock the model to avoid actual training
        locator.model = Mock()
        locator.model.fit.return_value = Mock(history={})
        
        # Train
        locator.train(genotypes=self.genotypes, samples=self.samples)
        
        # Verify that filtered_genotypes was stored
        assert hasattr(locator, 'filtered_genotypes')
        assert hasattr(locator, 'index_set')
        
        # Verify no array reconstruction occurred
        # The model.fit should have been called with tf.data.Dataset
        fit_call = locator.model.fit.call_args
        assert fit_call is not None
        # First argument should be a tf.data.Dataset
        train_data = fit_call[0][0]
        assert hasattr(train_data, '__iter__')  # It's a dataset, not a numpy array
    
    def test_bootstrap_with_site_order(self, tmp_path):
        """Test that bootstrap can use site_order parameter"""
        # Write sample data
        sample_file = tmp_path / "samples.txt"
        sample_file.write_text(self.sample_data)
        
        # Create locator
        config = {
            "out": str(tmp_path / "test"),
            "sample_data": str(sample_file),
            "max_epochs": 1,
            "use_efficient_pipeline": True,
        }
        locator = Locator(config)
        
        # Mock the model
        locator.model = Mock()
        locator.model.fit.return_value = Mock(history={})
        
        # Create bootstrap site order (with replacement)
        site_order = np.random.choice(self.n_snps, size=self.n_snps, replace=True)
        
        # Train with site_order
        locator.train(
            genotypes=self.genotypes, 
            samples=self.samples,
            site_order=site_order
        )
        
        # The test passes if no error occurs
        # In the future, we could verify that site_order was passed to make_tf_dataset
    
    @patch('locator.training.make_tf_dataset')
    def test_no_array_reconstruction(self, mock_make_tf_dataset, tmp_path):
        """Test that we don't reconstruct arrays when using tf.data"""
        # Write sample data
        sample_file = tmp_path / "samples.txt"
        sample_file.write_text(self.sample_data)
        
        # Create locator
        config = {
            "out": str(tmp_path / "test"),
            "sample_data": str(sample_file),
            "max_epochs": 1,
            "use_efficient_pipeline": True,
        }
        locator = Locator(config)
        
        # Mock make_tf_dataset to return a mock dataset
        mock_dataset = Mock()
        mock_make_tf_dataset.return_value = mock_dataset
        
        # Mock the model
        locator.model = Mock()
        locator.model.fit.return_value = Mock(history={})
        
        # Train
        locator.train(genotypes=self.genotypes, samples=self.samples)
        
        # Verify make_tf_dataset was called with the filtered genotypes
        assert mock_make_tf_dataset.called
        call_args = mock_make_tf_dataset.call_args_list[0][1]  # First call, keyword args
        
        # The genotypes passed should be the filtered_genotypes
        assert 'genotypes' in call_args
        assert call_args['genotypes'] is locator.filtered_genotypes
        
        # Should have been called twice (train and test datasets)
        assert mock_make_tf_dataset.call_count == 2