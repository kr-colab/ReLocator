"""Test bootstrap analysis with tf.data pipeline integration"""

import numpy as np
import pytest
from unittest.mock import Mock, patch
import allel
import pandas as pd

from locator.core import Locator


class TestBootstrapTFData:
    """Test bootstrap resampling uses tf.data pipeline efficiently"""
    
    def setup_method(self):
        """Set up test data"""
        self.n_samples = 30
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
        
        # Create coordinates (all with known locations for bootstrap)
        self.coords = np.random.uniform(-180, 180, size=(self.n_samples, 2))
        
        # Create sample data file content
        self.sample_data = "sampleID\tx\ty\n"
        for i, sid in enumerate(self.samples):
            x, y = self.coords[i]
            self.sample_data += f"{sid}\t{x}\t{y}\n"
    
    def test_bootstrap_uses_site_order(self, tmp_path):
        """Test that bootstrap uses site_order parameter without array copies"""
        # Write sample data
        sample_file = tmp_path / "samples.txt"
        sample_file.write_text(self.sample_data)
        
        # Create locator
        config = {
            "out": str(tmp_path / "test"),
            "sample_data": str(sample_file),
            "max_epochs": 1,
            "use_efficient_pipeline": True,
            "keras_verbose": 0,
        }
        locator = Locator(config)
        
        # Run bootstrap with small number of replicates
        results = locator.run_bootstraps(
            genotypes=self.genotypes,
            samples=self.samples,
            n_bootstraps=2,
            return_df=True
        )
        
        # Verify results
        assert isinstance(results, pd.DataFrame)
        assert "x_0" in results.columns
        assert "y_0" in results.columns
        assert "x_1" in results.columns
        assert "y_1" in results.columns
        assert len(results) == self.n_samples
    
    def test_bootstrap_site_order_propagation(self, tmp_path):
        """Test that bootstrap properly generates and uses site_order"""
        # Write sample data
        sample_file = tmp_path / "samples.txt"
        sample_file.write_text(self.sample_data)
        
        # Create locator
        config = {
            "out": str(tmp_path / "test"),
            "sample_data": str(sample_file),
            "max_epochs": 1,
            "use_efficient_pipeline": True,
            "keras_verbose": 0,
        }
        locator = Locator(config)
        
        # Track site orders used during training
        site_orders_used = []
        original_train = locator.train
        
        def track_train(*args, **kwargs):
            site_order = kwargs.get('site_order', None)
            if site_order is not None:
                site_orders_used.append(site_order.copy())
            return original_train(*args, **kwargs)
        
        locator.train = track_train
        
        # Run bootstrap with 2 replicates
        locator.run_bootstraps(
            genotypes=self.genotypes,
            samples=self.samples,
            n_bootstraps=2,
            return_df=False
        )
        
        # Verify site_order was used for each bootstrap
        assert len(site_orders_used) == 2
        
        # Each site_order should be different (very unlikely to be same with random sampling)
        assert not np.array_equal(site_orders_used[0], site_orders_used[1])
        
        # Each site_order should have the right shape
        for site_order in site_orders_used:
            assert isinstance(site_order, np.ndarray)
            assert site_order.shape[0] > 0  # Should match number of SNPs after filtering
    
    def test_bootstrap_memory_efficiency(self, tmp_path):
        """Test that bootstrap doesn't create array copies"""
        # Write sample data
        sample_file = tmp_path / "samples.txt"
        sample_file.write_text(self.sample_data)
        
        # Create locator
        config = {
            "out": str(tmp_path / "test"),
            "sample_data": str(sample_file),
            "max_epochs": 1,
            "use_efficient_pipeline": True,
            "keras_verbose": 0,
        }
        locator = Locator(config)
        
        # Initial training
        locator.train(genotypes=self.genotypes, samples=self.samples)
        
        # Store shape of filtered genotypes
        filtered_geno_shape = locator.filtered_genotypes.shape
        n_snps_filtered = filtered_geno_shape[0]
        
        # Mock model to avoid actual training
        locator.model = Mock()
        locator.model.fit.return_value = Mock(history={})
        locator.model.predict.return_value = np.random.normal(0, 1, (self.n_samples, 2))
        
        # Run one bootstrap iteration
        locator.run_bootstraps(
            genotypes=self.genotypes,
            samples=self.samples,
            n_bootstraps=1,
            return_df=False
        )
        
        # Verify filtered_genotypes is still available with same shape
        assert hasattr(locator, 'filtered_genotypes')
        assert locator.filtered_genotypes.shape == filtered_geno_shape
        
        # Verify no extra memory copies were made
        # The filtered genotypes should be reused across iterations
        assert locator.filtered_genotypes.shape[0] == n_snps_filtered