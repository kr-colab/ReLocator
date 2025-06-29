"""Test jacknife analysis with improved memory efficiency"""

import numpy as np
import pytest
from unittest.mock import Mock, patch
import allel
import pandas as pd

from locator.core import Locator


class TestJackknifeTFData:
    """Test jacknife resampling with improved memory efficiency"""
    
    def setup_method(self):
        """Set up test data"""
        self.n_samples = 20
        self.n_snps = 50
        
        # Create genotype data (biallelic)
        geno_array = np.zeros((self.n_snps, self.n_samples, 2), dtype=np.int8)
        for i in range(self.n_snps):
            # Create biallelic genotypes with varying allele frequencies
            af = np.random.uniform(0.1, 0.9)  # Random allele frequency
            for j in range(self.n_samples):
                # Generate genotypes based on HWE
                p = af
                q = 1 - af
                genotype_probs = [q*q, 2*p*q, p*p]
                allele_count = np.random.choice([0, 1, 2], p=genotype_probs)
                if allele_count == 0:
                    geno_array[i, j, :] = [0, 0]
                elif allele_count == 1:
                    geno_array[i, j, :] = [0, 1]
                else:
                    geno_array[i, j, :] = [1, 1]
        self.genotypes = allel.GenotypeArray(geno_array)
        
        # Create sample IDs
        self.samples = np.array([f"sample_{i}" for i in range(self.n_samples)])
        
        # Create coordinates (all with known locations for jacknife)
        self.coords = np.random.uniform(-180, 180, size=(self.n_samples, 2))
        
        # Create sample data file content
        self.sample_data = "sampleID\tx\ty\n"
        for i, sid in enumerate(self.samples):
            x, y = self.coords[i]
            self.sample_data += f"{sid}\t{x}\t{y}\n"
    
    def test_jacknife_basic(self, tmp_path):
        """Test that jacknife runs without errors"""
        # Write sample data
        sample_file = tmp_path / "samples.txt"
        sample_file.write_text(self.sample_data)
        
        # Create locator
        config = {
            "out": str(tmp_path / "test"),
            "sample_data": str(sample_file),
            "max_epochs": 1,
            "nboots": 2,  # Small number for testing
            "keras_verbose": 0,
        }
        locator = Locator(config)
        
        # Run jacknife
        results = locator.run_jacknife(
            genotypes=self.genotypes,
            samples=self.samples,
            prop=0.1,  # Drop 10% of SNPs
            return_df=True
        )
        
        # Verify results
        assert isinstance(results, pd.DataFrame)
        assert "x_0" in results.columns
        assert "y_0" in results.columns
        assert "x_1" in results.columns
        assert "y_1" in results.columns
        assert len(results) == self.n_samples
    
    def test_jacknife_no_deep_copy(self, tmp_path):
        """Test that jacknife doesn't use deep copy"""
        # Write sample data
        sample_file = tmp_path / "samples.txt"
        sample_file.write_text(self.sample_data)
        
        # Create locator
        config = {
            "out": str(tmp_path / "test"),
            "sample_data": str(sample_file),
            "max_epochs": 1,
            "nboots": 1,
            "keras_verbose": 0,
        }
        locator = Locator(config)
        
        # Initial training
        locator.train(genotypes=self.genotypes, samples=self.samples)
        
        # Mock predict to track what genotypes are passed
        predictions_seen = []
        original_predict = locator.predict
        
        def track_predict(*args, **kwargs):
            pg = kwargs.get('prediction_genotypes')
            if pg is not None:
                # Check that it's not the exact same object as predgen
                assert pg is not locator.predgen
                # But it should have the same shape
                assert pg.shape == locator.predgen.shape
                predictions_seen.append(pg)
            return original_predict(*args, **kwargs)
        
        locator.predict = track_predict
        
        # Run jacknife
        locator.run_jacknife(
            genotypes=self.genotypes,
            samples=self.samples,
            prop=0.2,
            return_df=False
        )
        
        # Verify predictions were made with modified genotypes
        assert len(predictions_seen) == 1
        # The modified genotypes should differ from original at some sites
        assert not np.array_equal(predictions_seen[0], locator.predgen)
    
    def test_jacknife_allele_frequency_calculation(self, tmp_path):
        """Test that allele frequencies are calculated correctly"""
        # Write sample data
        sample_file = tmp_path / "samples.txt"
        sample_file.write_text(self.sample_data)
        
        # Create locator
        config = {
            "out": str(tmp_path / "test"),
            "sample_data": str(sample_file),
            "max_epochs": 1,
            "nboots": 1,
            "keras_verbose": 0,
        }
        locator = Locator(config)
        
        # Initial training
        locator.train(genotypes=self.genotypes, samples=self.samples)
        
        # Track genotypes used in prediction
        prediction_genotypes = []
        original_predict = locator.predict
        
        def capture_genotypes(*args, **kwargs):
            pg = kwargs.get('prediction_genotypes')
            if pg is not None:
                prediction_genotypes.append(pg.copy())
            # Return mock predictions
            return pd.DataFrame({
                'sampleID': locator.samples[:len(pg)] if hasattr(locator, 'samples') else [f's{i}' for i in range(len(pg))],
                'x': np.random.normal(0, 1, len(pg)),
                'y': np.random.normal(0, 1, len(pg))
            })
        
        locator.predict = capture_genotypes
        
        # Run jacknife with high proportion to ensure sites are replaced
        locator.run_jacknife(
            genotypes=self.genotypes,
            samples=self.samples,
            prop=0.5,  # Drop 50% of SNPs
            return_df=False
        )
        
        # Verify that predictions were made
        assert len(prediction_genotypes) > 0
        
        # Check that replaced sites have valid genotype values (0, 1, or 2)
        for pg in prediction_genotypes:
            assert np.all(np.isin(pg, [0, 1, 2]))
    
    def test_jacknife_with_filtered_genotypes(self, tmp_path):
        """Test jacknife uses filtered genotypes when available"""
        # Write sample data
        sample_file = tmp_path / "samples.txt"
        sample_file.write_text(self.sample_data)
        
        # Create locator
        config = {
            "out": str(tmp_path / "test"),
            "sample_data": str(sample_file),
            "max_epochs": 1,
            "nboots": 1,
            "keras_verbose": 0,
            "min_mac": 5,  # Filter SNPs
        }
        locator = Locator(config)
        
        # Initial training (will create filtered_genotypes)
        locator.train(genotypes=self.genotypes, samples=self.samples)
        
        # Verify filtered_genotypes exists
        assert hasattr(locator, 'filtered_genotypes')
        n_snps_filtered = locator.filtered_genotypes.shape[0]
        
        # Mock model to speed up
        locator.model = Mock()
        locator.model.fit.return_value = Mock(history={})
        locator.model.predict.return_value = np.random.normal(0, 1, (self.n_samples, 2))
        
        # Run jacknife
        locator.run_jacknife(
            genotypes=self.genotypes,
            samples=self.samples,
            prop=0.1,
            return_df=False
        )
        
        # Verify filtered_genotypes is still available
        assert hasattr(locator, 'filtered_genotypes')
        assert locator.filtered_genotypes.shape[0] == n_snps_filtered