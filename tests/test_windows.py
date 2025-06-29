"""Tests for window analysis functionality."""

import pytest
import numpy as np
import pandas as pd
import allel
from locator import Locator
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend


class TestWindowAnalysis:
    """Test suite for window analysis methods."""
    
    def create_test_data_with_positions(self, n_samples=15, n_snps=100, n_known=12):
        """Create test genotype data with position information."""
        # Create sample IDs
        samples = np.array([f'sample_{i}' for i in range(n_samples)])
        
        # Create biallelic genotype data
        genotype_array = np.zeros((n_snps, n_samples, 2), dtype=np.int8)
        
        # Fill with random genotypes
        for i in range(n_snps):
            for j in range(n_samples):
                allele_count = np.random.randint(0, 3)
                if allele_count == 0:
                    genotype_array[i, j, :] = [0, 0]
                elif allele_count == 1:
                    genotype_array[i, j, :] = [0, 1]
                else:
                    genotype_array[i, j, :] = [1, 1]
        
        # Convert to allel.GenotypeArray
        genotypes = allel.GenotypeArray(genotype_array)
        
        # Create positions spanning 1 Mb
        positions = np.sort(np.random.randint(0, 1_000_000, size=n_snps))
        
        # Create coordinate data
        x_coords = [float(i) if i < n_known else np.nan for i in range(n_samples)]
        y_coords = [float(i + 10) if i < n_known else np.nan for i in range(n_samples)]
        
        sample_df = pd.DataFrame({
            'sampleID': samples,
            'x': x_coords,
            'y': y_coords
        })
        
        # Create genotype DataFrame with positions as columns
        ac = genotypes.to_allele_counts()[:, :, 1]
        geno_df = pd.DataFrame(
            ac.T,
            index=samples,
            columns=positions
        )
        
        return genotypes, samples, sample_df, geno_df, positions
    
    def test_run_windows_basic(self):
        """Test basic window analysis functionality."""
        genotypes, samples, sample_df, geno_df, positions = self.create_test_data_with_positions()
        
        locator = Locator({
            'sample_data': sample_df,
            'genotype_data': geno_df,
            'keras_verbose': 0,
            'max_epochs': 5,
            'out': 'test_windows_basic'
        })
        
        # Run window analysis
        result = locator.run_windows(
            genotypes=genotypes,
            samples=samples,
            window_size=300_000,  # 300kb windows
            return_df=True,
            save_full_pred_matrix=False
        )
        
        assert result is not None
        assert isinstance(result, pd.DataFrame)
        assert 'sampleID' in result.columns
        
        # Check that we have predictions for multiple windows
        x_cols = [col for col in result.columns if col.startswith('x_')]
        y_cols = [col for col in result.columns if col.startswith('y_')]
        assert len(x_cols) > 0
        assert len(x_cols) == len(y_cols)
        
        # Check that predictions are numeric
        assert result[x_cols[0]].dtype in [np.float32, np.float64]
        assert result[y_cols[0]].dtype in [np.float32, np.float64]
    
    def test_run_windows_with_na_samples(self):
        """Test window analysis with samples lacking coordinates."""
        genotypes, samples, sample_df, geno_df, positions = self.create_test_data_with_positions(
            n_samples=20, n_snps=100, n_known=15
        )
        
        locator = Locator({
            'sample_data': sample_df,
            'genotype_data': geno_df,
            'na_action': 'separate',
            'keras_verbose': 0,
            'max_epochs': 5,
            'out': 'test_windows_na'
        })
        
        result = locator.run_windows(
            genotypes=genotypes,
            samples=samples,
            window_size=250_000,
            return_df=True,
            save_full_pred_matrix=False
        )
        
        # Should have predictions for all samples
        assert len(result) == len(samples)
        
        # Check that NA samples are included
        na_samples = sample_df[sample_df.x.isna()]['sampleID'].values
        result_samples = result['sampleID'].values
        for na_sample in na_samples:
            assert na_sample in result_samples
    
    def test_run_windows_exclude_mode(self):
        """Test window analysis with exclude mode."""
        genotypes, samples, sample_df, geno_df, positions = self.create_test_data_with_positions(
            n_samples=20, n_snps=100, n_known=15
        )
        
        locator = Locator({
            'sample_data': sample_df,
            'genotype_data': geno_df,
            'na_action': 'exclude',
            'keras_verbose': 0,
            'max_epochs': 5,
            'out': 'test_windows_exclude'
        })
        
        result = locator.run_windows(
            genotypes=genotypes,
            samples=samples,
            window_size=250_000,
            return_df=True,
            save_full_pred_matrix=False
        )
        
        # Should only have predictions for samples with coordinates
        n_known = sample_df.x.notna().sum()
        assert len(result) == n_known
    
    def test_run_windows_window_size(self):
        """Test different window sizes."""
        genotypes, samples, sample_df, geno_df, positions = self.create_test_data_with_positions(
            n_samples=10, n_snps=150, n_known=10
        )
        
        locator = Locator({
            'sample_data': sample_df,
            'genotype_data': geno_df,
            'keras_verbose': 0,
            'max_epochs': 5,
            'out': 'test_windows_size'
        })
        
        # Small windows
        result_small = locator.run_windows(
            genotypes=genotypes,
            samples=samples,
            window_size=100_000,  # 100kb
            return_df=True,
            save_full_pred_matrix=False
        )
        
        # Large windows
        result_large = locator.run_windows(
            genotypes=genotypes,
            samples=samples,
            window_size=500_000,  # 500kb
            return_df=True,
            save_full_pred_matrix=False
        )
        
        # Should have more windows with smaller window size
        x_cols_small = [col for col in result_small.columns if col.startswith('x_')]
        x_cols_large = [col for col in result_large.columns if col.startswith('x_')]
        assert len(x_cols_small) >= len(x_cols_large)
    
    def test_run_windows_holdouts(self):
        """Test window analysis with holdouts."""
        genotypes, samples, sample_df, geno_df, positions = self.create_test_data_with_positions(
            n_samples=15, n_snps=100, n_known=15  # All samples need coordinates for holdout
        )
        
        locator = Locator({
            'sample_data': sample_df,
            'genotype_data': geno_df,
            'keras_verbose': 0,
            'max_epochs': 5,
            'out': 'test_windows_holdouts'
        })
        
        result = locator.run_windows_holdouts(
            genotypes=genotypes,
            samples=samples,
            k=3,  # Hold out 3 samples
            window_size=400_000,
            return_df=True
        )
        
        assert result is not None
        assert isinstance(result, pd.DataFrame)
        
        # Should have window-based prediction columns
        x_cols = [col for col in result.columns if col.startswith('x_pos')]
        y_cols = [col for col in result.columns if col.startswith('y_pos')]
        assert len(x_cols) > 0
        assert len(x_cols) == len(y_cols)
        
        # Should have predictions for holdout samples
        assert len(result) == 3  # k holdout samples
    
    def test_window_analysis_without_positions(self):
        """Test that window analysis fails gracefully without position information."""
        # Create data without position information
        samples = np.array(['s1', 's2', 's3'])
        sample_df = pd.DataFrame({
            'sampleID': samples,
            'x': [1.0, 2.0, 3.0],
            'y': [4.0, 5.0, 6.0]
        })
        
        # Simple genotype array without positions
        genotypes = allel.GenotypeArray(np.zeros((10, 3, 2), dtype=np.int8))
        
        locator = Locator({
            'sample_data': sample_df,
            'keras_verbose': 0,
            'out': 'test_no_positions'
        })
        
        # Should raise error about missing positions
        with pytest.raises(ValueError, match="SNP positions required"):
            locator.run_windows(
                genotypes=genotypes,
                samples=samples,
                window_size=100_000,
                return_df=True
            )
    
    def test_window_start_stop(self):
        """Test window start and stop parameters."""
        genotypes, samples, sample_df, geno_df, positions = self.create_test_data_with_positions(
            n_samples=10, n_snps=100, n_known=10
        )
        
        locator = Locator({
            'sample_data': sample_df,
            'genotype_data': geno_df,
            'keras_verbose': 0,
            'max_epochs': 5,
            'out': 'test_window_range'
        })
        
        # Run with specific start and stop
        result = locator.run_windows(
            genotypes=genotypes,
            samples=samples,
            window_start=200_000,
            window_stop=600_000,
            window_size=200_000,
            return_df=True,
            save_full_pred_matrix=False
        )
        
        # Should have limited number of windows
        x_cols = [col for col in result.columns if col.startswith('x_')]
        assert len(x_cols) <= 2  # At most 2 windows in 400kb range
    
    def test_window_predictions_consistency(self):
        """Test that window predictions are reasonable."""
        genotypes, samples, sample_df, geno_df, positions = self.create_test_data_with_positions(
            n_samples=10, n_snps=100, n_known=10
        )
        
        locator = Locator({
            'sample_data': sample_df,
            'genotype_data': geno_df,
            'keras_verbose': 0,
            'max_epochs': 10,
            'out': 'test_consistency'
        })
        
        result = locator.run_windows(
            genotypes=genotypes,
            samples=samples,
            window_size=300_000,
            return_df=True,
            save_full_pred_matrix=False
        )
        
        # Check that predictions are within reasonable bounds
        x_cols = [col for col in result.columns if col.startswith('x_')]
        y_cols = [col for col in result.columns if col.startswith('y_')]
        
        for x_col, y_col in zip(x_cols, y_cols):
            x_preds = result[x_col].values
            y_preds = result[y_col].values
            
            # Predictions should be finite
            assert np.all(np.isfinite(x_preds))
            assert np.all(np.isfinite(y_preds))
            
            # Predictions should have reasonable variance (not all the same)
            assert np.std(x_preds) > 0.01
            assert np.std(y_preds) > 0.01
    
    def test_run_windows_holdouts_na_handling(self):
        """Test run_windows_holdouts with different NA handling modes."""
        # Test with NA samples (should work with 'exclude' mode)
        genotypes, samples, sample_df, geno_df, positions = self.create_test_data_with_positions(
            n_samples=20, n_snps=100, n_known=15
        )
        
        locator = Locator({
            'sample_data': sample_df,
            'genotype_data': geno_df,
            'keras_verbose': 0,
            'max_epochs': 5,
            'out': 'test_windows_holdouts_na'
        })
        
        # Test exclude mode
        result = locator.run_windows_holdouts(
            genotypes=genotypes,
            samples=samples,
            k=3,
            window_size=400_000,
            na_action='exclude',
            return_df=True,
            save_full_pred_matrix=False
        )
        
        assert result is not None
        # With exclude mode and 15 known samples, we should get holdout samples
        # but the exact number may vary due to random selection from the 15 known samples
        assert len(result) > 0
        assert len(result) <= 3  # At most k holdout samples
        
        # Test separate mode (should behave like exclude for holdouts)
        result_sep = locator.run_windows_holdouts(
            genotypes=genotypes,
            samples=samples,
            k=3,
            window_size=400_000,
            na_action='separate',
            return_df=True,
            save_full_pred_matrix=False
        )
        
        assert result_sep is not None
        assert len(result_sep) > 0
        assert len(result_sep) <= 3
        
        # Test fail mode
        with pytest.raises(ValueError, match="samples without coordinates"):
            locator.run_windows_holdouts(
                genotypes=genotypes,
                samples=samples,
                k=3,
                window_size=400_000,
                na_action='fail',
                return_df=True
            )
    
    def test_run_windows_holdouts_with_indices(self):
        """Test run_windows_holdouts with specific holdout indices."""
        genotypes, samples, sample_df, geno_df, positions = self.create_test_data_with_positions(
            n_samples=15, n_snps=100, n_known=15
        )
        
        locator = Locator({
            'sample_data': sample_df,
            'genotype_data': geno_df,
            'keras_verbose': 0,
            'max_epochs': 5,
            'out': 'test_windows_holdouts_indices'
        })
        
        # Specify exact holdout indices
        holdout_indices = [2, 5, 8, 11]
        
        result = locator.run_windows_holdouts(
            genotypes=genotypes,
            samples=samples,
            holdout_indices=holdout_indices,
            window_size=400_000,
            return_df=True,
            save_full_pred_matrix=False
        )
        
        assert result is not None
        assert len(result) == len(holdout_indices)
        
        # Check that the correct samples were held out
        expected_samples = samples[holdout_indices]
        assert set(result['sampleID'].values) == set(expected_samples)
    
    def test_run_windows_holdouts_window_parameters(self):
        """Test run_windows_holdouts with different window parameters."""
        genotypes, samples, sample_df, geno_df, positions = self.create_test_data_with_positions(
            n_samples=10, n_snps=150, n_known=10
        )
        
        locator = Locator({
            'sample_data': sample_df,
            'genotype_data': geno_df,
            'keras_verbose': 0,
            'max_epochs': 5,
            'out': 'test_windows_holdouts_params'
        })
        
        # Test with specific window start and stop
        result = locator.run_windows_holdouts(
            genotypes=genotypes,
            samples=samples,
            k=2,
            window_start=200_000,
            window_stop=700_000,
            window_size=250_000,
            return_df=True,
            save_full_pred_matrix=False
        )
        
        assert result is not None
        
        # Should have predictions for multiple windows
        x_cols = [col for col in result.columns if col.startswith('x_pos')]
        assert len(x_cols) >= 1
        assert len(x_cols) <= 2  # At most 2 windows in 500kb range with 250kb size
    
    def test_run_windows_holdouts_save_options(self):
        """Test save_full_pred_matrix option."""
        import os
        import tempfile
        
        genotypes, samples, sample_df, geno_df, positions = self.create_test_data_with_positions(
            n_samples=10, n_snps=100, n_known=10
        )
        
        # Use temporary directory for output
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = os.path.join(tmpdir, 'test_save')
            
            locator = Locator({
                'sample_data': sample_df,
                'genotype_data': geno_df,
                'keras_verbose': 0,
                'max_epochs': 5,
                'out': out_path
            })
            
            # Test with save_full_pred_matrix=True
            result = locator.run_windows_holdouts(
                genotypes=genotypes,
                samples=samples,
                k=3,
                window_size=400_000,
                return_df=True,
                save_full_pred_matrix=True
            )
            
            # Check that file was created
            expected_file = f"{out_path}_windows_holdouts_predlocs.csv"
            assert os.path.exists(expected_file)
            
            # Load and verify saved file
            saved_df = pd.read_csv(expected_file)
            assert len(saved_df) == 3  # k holdout samples
            assert 'sampleID' in saved_df.columns
    
    def test_run_windows_holdouts_empty_windows(self):
        """Test behavior when some windows have no SNPs."""
        # Create sparse SNP data with gaps
        n_samples = 10
        n_snps = 50
        samples = np.array([f'sample_{i}' for i in range(n_samples)])
        
        # Create positions with large gaps
        positions = np.concatenate([
            np.arange(0, 100_000, 2000),  # SNPs in first 100kb
            np.arange(800_000, 900_000, 2000)  # SNPs in 800-900kb range
        ])
        n_snps = len(positions)
        
        # Create genotype data
        genotype_array = np.random.randint(0, 2, size=(n_snps, n_samples, 2), dtype=np.int8)
        genotypes = allel.GenotypeArray(genotype_array)
        
        # Create sample data
        sample_df = pd.DataFrame({
            'sampleID': samples,
            'x': range(n_samples),
            'y': range(10, 10 + n_samples)
        })
        
        # Create genotype DataFrame
        ac = genotypes.to_allele_counts()[:, :, 1]
        geno_df = pd.DataFrame(
            ac.T,
            index=samples,
            columns=positions
        )
        
        locator = Locator({
            'sample_data': sample_df,
            'genotype_data': geno_df,
            'keras_verbose': 0,
            'max_epochs': 5,
            'out': 'test_empty_windows'
        })
        
        # Run with windows that will include empty regions
        result = locator.run_windows_holdouts(
            genotypes=genotypes,
            samples=samples,
            k=2,
            window_size=200_000,
            window_start=0,
            window_stop=1_000_000,
            return_df=True,
            save_full_pred_matrix=False
        )
        
        assert result is not None
        
        # Should have predictions only for windows with SNPs
        x_cols = [col for col in result.columns if col.startswith('x_pos')]
        # We expect predictions for windows containing SNPs
        assert len(x_cols) >= 2  # At least first window and last window