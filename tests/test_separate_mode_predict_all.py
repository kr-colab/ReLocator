"""Test that 'separate' mode predicts on all samples."""

import numpy as np
import pandas as pd
import allel
import pytest
from locator import Locator


class TestSeparateModePredictAll:
    """Test that 'separate' mode predicts on all samples, not just NA samples."""
    
    def create_test_data(self, n_samples=20, n_known=15):
        """Create test genotype and sample data with some NA coordinates."""
        np.random.seed(42)
        
        # Create sample IDs
        samples = np.array([f"sample_{i}" for i in range(n_samples)])
        
        # Create genotype data (100 SNPs x n_samples x 2)
        genotype_array = np.zeros((100, n_samples, 2), dtype=np.int8)
        
        # Fill with biallelic genotypes (only 0s and 1s)
        for i in range(100):
            for j in range(n_samples):
                # Generate allele counts between 0 and 2
                allele_count = np.random.randint(0, 3)
                if allele_count == 0:
                    genotype_array[i, j, :] = [0, 0]
                elif allele_count == 1:
                    genotype_array[i, j, :] = [0, 1]
                else:  # allele_count == 2
                    genotype_array[i, j, :] = [1, 1]
        
        # Convert to allel.GenotypeArray
        genotypes = allel.GenotypeArray(genotype_array)
        
        # Create coordinate data with some NAs
        x_coords = [float(i) for i in range(n_known)] + [np.nan] * (n_samples - n_known)
        y_coords = [float(i + 10) for i in range(n_known)] + [np.nan] * (n_samples - n_known)
        
        sample_df = pd.DataFrame({
            'sampleID': samples,
            'x': x_coords,
            'y': y_coords
        })
        
        return genotypes, samples, sample_df
    
    def test_separate_mode_predicts_all_samples(self):
        """Test that 'separate' mode predicts on all samples (both known and NA)."""
        # Create test data with 20 samples (15 known, 5 NA)
        genotypes, samples, sample_df = self.create_test_data(n_samples=20, n_known=15)
        
        # Initialize Locator with 'separate' mode
        locator = Locator({
            'sample_data': sample_df,
            'na_action': 'separate',
            'keras_verbose': 0,
            'max_epochs': 5,
            'out': 'test_separate_all'
        })
        
        # Train the model
        history = locator.train(genotypes=genotypes, samples=samples)
        assert history is not None
        
        # Get predictions
        predictions = locator.predict(return_df=True, save_preds_to_disk=False)
        
        # Check that we got predictions for ALL samples
        assert len(predictions) == 20, f"Expected 20 predictions but got {len(predictions)}"
        
        # Check that predictions include all sample IDs
        pred_sample_ids = set(predictions['sampleID'])
        all_sample_ids = set(samples)
        assert pred_sample_ids == all_sample_ids, "Predictions should include all samples"
        
    def test_separate_mode_with_no_na_samples(self):
        """Test that 'separate' mode works correctly when all samples have coordinates."""
        # Create test data with all samples having known coordinates
        genotypes, samples, sample_df = self.create_test_data(n_samples=10, n_known=10)
        
        # Initialize Locator with 'separate' mode
        locator = Locator({
            'sample_data': sample_df,
            'na_action': 'separate',
            'keras_verbose': 0,
            'max_epochs': 5,
            'out': 'test_separate_no_na'
        })
        
        # Train the model
        history = locator.train(genotypes=genotypes, samples=samples)
        assert history is not None
        
        # Get predictions
        predictions = locator.predict(return_df=True, save_preds_to_disk=False)
        
        # Check that we still get predictions for all samples
        assert len(predictions) == 10, f"Expected 10 predictions but got {len(predictions)}"
        
    def test_exclude_mode_only_predicts_na(self):
        """Test that 'exclude' mode excludes NA samples from both training and prediction."""
        # Create test data with 20 samples (15 known, 5 NA)
        genotypes, samples, sample_df = self.create_test_data(n_samples=20, n_known=15)
        
        # Initialize Locator with 'exclude' mode
        locator = Locator({
            'sample_data': sample_df,
            'na_action': 'exclude',
            'keras_verbose': 0,
            'max_epochs': 5,
            'out': 'test_exclude'
        })
        
        # Train the model
        history = locator.train(genotypes=genotypes, samples=samples)
        assert history is not None
        
        # In exclude mode, we've excluded NA samples from training,
        # so predgen should be empty
        assert locator.predgen.shape[0] == 0, "In exclude mode, predgen should be empty"
        
        # If we try to predict, we should get an empty result
        predictions = locator.predict(return_df=True, save_preds_to_disk=False)
        
        # In exclude mode, there are no samples to predict
        assert len(predictions) == 0, f"Expected 0 predictions in exclude mode but got {len(predictions)}"