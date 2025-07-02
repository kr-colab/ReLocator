"""Tests for NA handling functionality in Locator."""

import pytest
import numpy as np
import pandas as pd
import allel
from locator import Locator
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to suppress plots
import matplotlib.pyplot as plt


class TestNAHandling:
    """Test suite for NA handling infrastructure."""
    
    def test_na_action_initialization(self):
        """Test that na_action parameter is properly initialized."""
        # Test default na_action
        locator = Locator()
        assert locator.na_action == 'separate'
        assert locator.config['na_action'] == 'separate'
        
        # Test custom na_action values
        for action in ['separate', 'exclude', 'fail']:
            locator = Locator({'na_action': action})
            assert locator.na_action == action
            assert locator.config['na_action'] == action
    
    def test_na_action_validation(self):
        """Test that invalid na_action values raise appropriate errors."""
        with pytest.raises(ValueError, match="Invalid na_action 'invalid'"):
            Locator({'na_action': 'invalid'})
        
        with pytest.raises(ValueError, match="Must be one of"):
            Locator({'na_action': 'unknown'})
    
    def test_get_sample_status_all_known(self):
        """Test get_sample_status with all samples having known coordinates."""
        # Create test data
        samples = np.array(['sample1', 'sample2', 'sample3'])
        sample_df = pd.DataFrame({
            'sampleID': samples,
            'x': [1.0, 2.0, 3.0],
            'y': [4.0, 5.0, 6.0]
        })
        
        locator = Locator({'sample_data': sample_df})
        status = locator.get_sample_status(samples)
        
        assert status['n_known'] == 3
        assert status['n_na'] == 0
        assert status['total'] == 3
        assert len(status['known_indices']) == 3
        assert len(status['na_indices']) == 0
        assert np.array_equal(status['known_samples'], samples)
        assert len(status['na_samples']) == 0
    
    def test_get_sample_status_all_na(self):
        """Test get_sample_status with all samples having NA coordinates."""
        samples = np.array(['sample1', 'sample2', 'sample3'])
        sample_df = pd.DataFrame({
            'sampleID': samples,
            'x': [np.nan, np.nan, np.nan],
            'y': [np.nan, np.nan, np.nan]
        })
        
        locator = Locator({'sample_data': sample_df})
        status = locator.get_sample_status(samples)
        
        assert status['n_known'] == 0
        assert status['n_na'] == 3
        assert status['total'] == 3
        assert len(status['known_indices']) == 0
        assert len(status['na_indices']) == 3
        assert len(status['known_samples']) == 0
        assert np.array_equal(status['na_samples'], samples)
    
    def test_get_sample_status_mixed(self):
        """Test get_sample_status with mix of known and NA coordinates."""
        samples = np.array(['sample1', 'sample2', 'sample3', 'sample4', 'sample5'])
        sample_df = pd.DataFrame({
            'sampleID': samples,
            'x': [1.0, np.nan, 3.0, np.nan, 5.0],
            'y': [6.0, np.nan, 8.0, 9.0, 10.0]  # Note: sample4 has x=nan but y=9.0
        })
        
        locator = Locator({'sample_data': sample_df})
        status = locator.get_sample_status(samples)
        
        # Samples are considered NA if either x or y is NaN
        assert status['n_known'] == 3  # samples 1, 3, 5
        assert status['n_na'] == 2      # samples 2, 4
        assert status['total'] == 5
        assert np.array_equal(status['known_indices'], [0, 2, 4])
        assert np.array_equal(status['na_indices'], [1, 3])
        assert np.array_equal(status['known_samples'], ['sample1', 'sample3', 'sample5'])
        assert np.array_equal(status['na_samples'], ['sample2', 'sample4'])
    
    def test_get_sample_status_with_provided_dataframe(self):
        """Test get_sample_status with externally provided DataFrame."""
        samples = np.array(['A', 'B', 'C'])
        external_df = pd.DataFrame({
            'sampleID': ['A', 'B', 'C'],
            'x': [1.0, 2.0, np.nan],
            'y': [3.0, 4.0, np.nan]
        })
        
        # Don't need sample_data in config
        locator = Locator()
        status = locator.get_sample_status(samples, sample_data=external_df)
        
        assert status['n_known'] == 2
        assert status['n_na'] == 1
        assert np.array_equal(status['known_samples'], ['A', 'B'])
        assert np.array_equal(status['na_samples'], ['C'])
    
    def test_get_sample_status_invalid_dataframe(self):
        """Test get_sample_status with invalid DataFrame."""
        samples = np.array(['A', 'B'])
        invalid_df = pd.DataFrame({
            'id': ['A', 'B'],  # Wrong column name
            'longitude': [1.0, 2.0],
            'latitude': [3.0, 4.0]
        })
        
        locator = Locator()
        with pytest.raises(ValueError, match="must contain columns"):
            locator.get_sample_status(samples, sample_data=invalid_df)
    
    def test_check_data_basic(self):
        """Test check_data method with basic functionality."""
        samples = np.array(['s1', 's2', 's3'])
        sample_df = pd.DataFrame({
            'sampleID': samples,
            'x': [1.0, np.nan, 3.0],
            'y': [4.0, np.nan, 6.0]
        })
        
        # Create mock genotype data
        genotypes = np.zeros((100, 3, 2))  # 100 SNPs, 3 samples, diploid
        
        locator = Locator({'sample_data': sample_df})
        
        # Test with verbose=False
        status = locator.check_data(genotypes, samples, verbose=False)
        assert status['n_known'] == 2
        assert status['n_na'] == 1
        
        # Test return value matches get_sample_status
        status2 = locator.get_sample_status(samples)
        # Compare dictionary values individually due to numpy arrays
        assert status['n_known'] == status2['n_known']
        assert status['n_na'] == status2['n_na']
        assert status['total'] == status2['total']
        assert np.array_equal(status['known_indices'], status2['known_indices'])
        assert np.array_equal(status['na_indices'], status2['na_indices'])
        assert np.array_equal(status['known_samples'], status2['known_samples'])
        assert np.array_equal(status['na_samples'], status2['na_samples'])
    
    def test_check_data_output_separate_mode(self, capsys):
        """Test check_data output in separate mode."""
        samples = np.array(['s1', 's2', 's3'])
        sample_df = pd.DataFrame({
            'sampleID': samples,
            'x': [1.0, np.nan, 3.0],
            'y': [4.0, np.nan, 6.0]
        })
        genotypes = np.zeros((50, 3, 2))
        
        locator = Locator({'sample_data': sample_df, 'na_action': 'separate'})
        locator.check_data(genotypes, samples, verbose=True)
        
        captured = capsys.readouterr()
        assert "Data Summary" in captured.out
        assert "Total samples: 3" in captured.out
        assert "Samples with coordinates: 2" in captured.out
        assert "Samples without coordinates: 1" in captured.out
        assert "Total SNPs: 50" in captured.out
        assert "Current NA handling mode: separate" in captured.out
        assert "Will train on samples with known locations" in captured.out
        assert "Can predict on samples without locations" in captured.out
        assert "s2" in captured.out  # The NA sample
    
    def test_check_data_output_exclude_mode(self, capsys):
        """Test check_data output in exclude mode."""
        samples = np.array(['s1', 's2'])
        sample_df = pd.DataFrame({
            'sampleID': samples,
            'x': [1.0, np.nan],
            'y': [2.0, np.nan]
        })
        genotypes = np.zeros((10, 2, 2))
        
        locator = Locator({'sample_data': sample_df, 'na_action': 'exclude'})
        locator.check_data(genotypes, samples, verbose=True)
        
        captured = capsys.readouterr()
        assert "Current NA handling mode: exclude" in captured.out
        assert "Will only use samples with known locations" in captured.out
        assert "Samples without locations will be excluded" in captured.out
    
    def test_check_data_output_fail_mode(self, capsys):
        """Test check_data output in fail mode with warning."""
        samples = np.array(['s1', 's2'])
        sample_df = pd.DataFrame({
            'sampleID': samples,
            'x': [1.0, np.nan],
            'y': [2.0, np.nan]
        })
        genotypes = np.zeros((10, 2, 2))
        
        locator = Locator({'sample_data': sample_df, 'na_action': 'fail'})
        locator.check_data(genotypes, samples, verbose=True)
        
        captured = capsys.readouterr()
        assert "Current NA handling mode: fail" in captured.out
        assert "Will raise an error if any samples lack coordinates" in captured.out
        assert "WARNING" in captured.out
        assert "na_action='fail' setting will cause" in captured.out
    
    def test_check_data_many_na_samples(self, capsys):
        """Test check_data with more than 10 NA samples."""
        # Create 15 samples where 12 have NA coordinates
        sample_ids = [f's{i}' for i in range(15)]
        samples = np.array(sample_ids)
        x_vals = [1.0, 2.0, 3.0] + [np.nan] * 12
        y_vals = [4.0, 5.0, 6.0] + [np.nan] * 12
        
        sample_df = pd.DataFrame({
            'sampleID': sample_ids,
            'x': x_vals,
            'y': y_vals
        })
        genotypes = np.zeros((100, 15, 2))
        
        locator = Locator({'sample_data': sample_df})
        locator.check_data(genotypes, samples, verbose=True)
        
        captured = capsys.readouterr()
        assert "Samples without coordinates: 12" in captured.out
        assert "Samples without coordinates (first 10):" in captured.out
        assert "... and 2 more" in captured.out
        # Check that exactly 10 sample IDs are shown
        na_sample_lines = [line for line in captured.out.split('\n') if line.strip().startswith('- s')]
        assert len(na_sample_lines) == 10


class TestPhase2NAHandling:
    """Test suite for Phase 2 - NA handling in analysis methods."""
    
    def create_test_data(self, n_samples=10, n_known=7):
        """Create test genotype and coordinate data."""
        # Create sample IDs
        samples = np.array([f'sample_{i}' for i in range(n_samples)])
        
        # Create genotype data that is guaranteed to be biallelic
        # 100 SNPs, n_samples, diploid
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
    
    def test_train_with_na_action_separate(self, capsys):
        """Test train() method with na_action='separate'."""
        genotypes, samples, sample_df = self.create_test_data(n_samples=10, n_known=7)
        
        locator = Locator({
            'sample_data': sample_df,
            'na_action': 'separate',
            'keras_verbose': 0,
            'max_epochs': 5,
            'out': 'test_separate'
        })
        
        # Should work fine with separate mode
        history = locator.train(genotypes=genotypes, samples=samples)
        
        captured = capsys.readouterr()
        assert "Training data: 7 samples with coordinates, 3 without" in captured.out
        assert "NA handling mode: separate" in captured.out
        assert history is not None
    
    def test_train_with_na_action_exclude(self, capsys):
        """Test train() method with na_action='exclude'."""
        genotypes, samples, sample_df = self.create_test_data(n_samples=10, n_known=7)
        
        locator = Locator({
            'sample_data': sample_df,
            'na_action': 'exclude',
            'keras_verbose': 0,
            'max_epochs': 5,
            'out': 'test_exclude'
        })
        
        history = locator.train(genotypes=genotypes, samples=samples)
        
        captured = capsys.readouterr()
        assert "Training data: 7 samples with coordinates, 3 without" in captured.out
        assert "NA handling mode: exclude" in captured.out
        assert "Excluding 3 samples without coordinates" in captured.out
        assert history is not None
    
    def test_train_with_na_action_fail(self):
        """Test train() method with na_action='fail' and NA samples."""
        genotypes, samples, sample_df = self.create_test_data(n_samples=10, n_known=7)
        
        locator = Locator({
            'sample_data': sample_df,
            'na_action': 'fail',
            'out': 'test_fail'
        })
        
        with pytest.raises(ValueError, match="Found 3 samples without coordinates"):
            locator.train(genotypes=genotypes, samples=samples)
    
    def test_train_method_override(self, capsys):
        """Test train() with method-level na_action override."""
        genotypes, samples, sample_df = self.create_test_data(n_samples=10, n_known=7)
        
        # Initialize with 'fail' but override with 'separate'
        locator = Locator({
            'sample_data': sample_df,
            'na_action': 'fail',
            'keras_verbose': 0,
            'max_epochs': 5,
            'out': 'test_override'
        })
        
        # Should work because we override with 'separate'
        history = locator.train(genotypes=genotypes, samples=samples, na_action='separate')
        
        captured = capsys.readouterr()
        assert "NA handling mode: separate" in captured.out
        assert history is not None
    
    def test_run_bootstraps_with_na_action(self, capsys):
        """Test run_bootstraps() with NA handling."""
        genotypes, samples, sample_df = self.create_test_data(n_samples=10, n_known=7)
        
        locator = Locator({
            'sample_data': sample_df,
            'na_action': 'separate',
            'keras_verbose': 0,
            'max_epochs': 5,
            'out': 'test_bootstrap'
        })
        
        # Run with just 2 bootstraps for speed
        result = locator.run_bootstraps(
            genotypes=genotypes, 
            samples=samples, 
            n_bootstraps=2,
            return_df=True
        )
        
        captured = capsys.readouterr()
        assert "Bootstrap analysis: 7 samples with coordinates, 3 without" in captured.out
        assert "NA handling mode: separate" in captured.out
        assert result is not None
        assert isinstance(result, pd.DataFrame)
    
    def test_run_bootstraps_fail_mode(self):
        """Test run_bootstraps() with fail mode and NA samples."""
        genotypes, samples, sample_df = self.create_test_data(n_samples=10, n_known=7)
        
        locator = Locator({
            'sample_data': sample_df,
            'na_action': 'fail',
            'out': 'test_bootstrap_fail'
        })
        
        with pytest.raises(ValueError, match="Found 3 samples without coordinates"):
            locator.run_bootstraps(genotypes=genotypes, samples=samples, n_bootstraps=2)
    
    def test_run_windows_with_na_action(self, capsys):
        """Test run_windows() with NA handling."""
        # Create genotype data with positions
        genotypes, samples, sample_df = self.create_test_data(n_samples=10, n_known=7)
        
        # Create a genotype DataFrame with positions as columns
        positions = np.arange(100) * 10000  # 100 SNPs spaced 10kb apart
        geno_df = pd.DataFrame(
            genotypes[:, :, 0].T,  # Just use one allele for simplicity
            index=samples,
            columns=positions
        )
        
        locator = Locator({
            'sample_data': sample_df,
            'genotype_data': geno_df,
            'na_action': 'separate',
            'keras_verbose': 0,
            'max_epochs': 5,
            'out': 'test_window'
        })
        
        # Run windows analysis
        result = locator.run_windows(
            genotypes=genotypes,
            samples=samples,
            window_size=3e5,  # 300kb windows
            return_df=True
        )
        
        captured = capsys.readouterr()
        assert "Window analysis: 7 samples with coordinates, 3 without" in captured.out
        assert "NA handling mode: separate" in captured.out
        assert result is not None


class TestPhase3NAHandling:
    """Test suite for Phase 3 - NA handling in holdout methods."""
    
    def setup_method(self):
        """Close any existing plots before each test."""
        plt.close('all')
    
    def teardown_method(self):
        """Close any plots created during test."""
        plt.close('all')
    
    def create_test_data(self, n_samples=10, n_known=7):
        """Create test genotype and coordinate data."""
        # Create sample IDs
        samples = np.array([f'sample_{i}' for i in range(n_samples)])
        
        # Create genotype data that is guaranteed to be biallelic
        # 100 SNPs, n_samples, diploid
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
    
    def test_run_holdouts_with_na_action(self, capsys):
        """Test run_holdouts() with NA handling."""
        genotypes, samples, sample_df = self.create_test_data(n_samples=10, n_known=7)
        
        locator = Locator({
            'sample_data': sample_df,
            'na_action': 'separate',
            'keras_verbose': 0,
            'max_epochs': 5,
            'out': 'test_holdouts'
        })
        
        # Run with just 2 replicates for speed
        result = locator.run_holdouts(
            genotypes=genotypes, 
            samples=samples, 
            k=2,  # Hold out 2 samples
            n_reps=2,
            return_df=True,
            save_full_pred_matrix=False  # Don't save to disk
        )
        
        captured = capsys.readouterr()
        assert "Holdout analysis: 7 samples with coordinates, 3 without" in captured.out
        assert "NA handling mode: separate" in captured.out
        assert "Note: Holdout analysis requires known locations" in captured.out
        assert result is not None
        assert isinstance(result, pd.DataFrame)
    
    def test_run_holdouts_fail_mode(self):
        """Test run_holdouts() with fail mode and NA samples."""
        genotypes, samples, sample_df = self.create_test_data(n_samples=10, n_known=7)
        
        locator = Locator({
            'sample_data': sample_df,
            'na_action': 'fail',
            'out': 'test_holdouts_fail'
        })
        
        with pytest.raises(ValueError, match="Found 3 samples without coordinates"):
            locator.run_holdouts(genotypes=genotypes, samples=samples, k=2, n_reps=1)
    
    def test_run_k_fold_holdouts_with_na_action(self, capsys):
        """Test run_k_fold_holdouts() with NA handling."""
        genotypes, samples, sample_df = self.create_test_data(n_samples=10, n_known=7)
        
        locator = Locator({
            'sample_data': sample_df,
            'na_action': 'separate',
            'keras_verbose': 0,
            'max_epochs': 5,
            'out': 'test_kfold'
        })
        
        # Run with just 2 folds for speed
        result = locator.run_k_fold_holdouts(
            genotypes=genotypes, 
            samples=samples, 
            k=2,
            return_df=True,
            verbose=True
        )
        
        captured = capsys.readouterr()
        assert "K-fold CV: 7 samples with coordinates, 3 without" in captured.out
        assert "NA handling mode: separate" in captured.out
        assert "Note: K-fold CV requires known locations" in captured.out
        assert result is not None
        assert isinstance(result, pd.DataFrame)
    
    def test_run_jacknife_with_na_action(self, capsys):
        """Test run_jacknife() with NA handling."""
        genotypes, samples, sample_df = self.create_test_data(n_samples=10, n_known=7)
        
        locator = Locator({
            'sample_data': sample_df,
            'na_action': 'separate',  # Changed from 'exclude' to test with samples to predict
            'keras_verbose': 0,
            'max_epochs': 5,
            'nboots': 2,  # Just 2 boots for speed
            'out': 'test_jacknife'
        })
        
        result = locator.run_jacknife(
            genotypes=genotypes, 
            samples=samples, 
            prop=0.1,
            return_df=True,
            save_full_pred_matrix=False  # Don't save to disk
        )
        
        captured = capsys.readouterr()
        assert "Jacknife analysis: 7 samples with coordinates, 3 without" in captured.out
        assert "NA handling mode: separate" in captured.out
        assert result is not None
        assert isinstance(result, pd.DataFrame)
    
    def test_run_jacknife_holdouts_with_na_action(self, capsys):
        """Test run_jacknife_holdouts() with NA handling."""
        genotypes, samples, sample_df = self.create_test_data(n_samples=10, n_known=7)
        
        locator = Locator({
            'sample_data': sample_df,
            'na_action': 'separate',
            'keras_verbose': 0,
            'max_epochs': 5,
            'out': 'test_jacknife_holdouts'
        })
        
        result = locator.run_jacknife_holdouts(
            genotypes=genotypes, 
            samples=samples,
            k=2,
            prop=0.1,
            n_boots=2,
            return_df=True
        )
        
        captured = capsys.readouterr()
        assert "Jacknife holdout analysis: 7 samples with coordinates, 3 without" in captured.out
        assert result is not None
        assert isinstance(result, pd.DataFrame)
    
    def test_run_windows_holdouts_with_na_action(self, capsys):
        """Test run_windows_holdouts() with NA handling."""
        genotypes, samples, sample_df = self.create_test_data(n_samples=10, n_known=7)
        
        # Create positions for the genotype data
        positions = np.arange(100) * 10000  # 100 SNPs spaced 10kb apart
        
        # We need to set up zarr or genotype DataFrame with positions
        # For simplicity, create a genotype DataFrame
        # Extract allele counts for first allele
        allele_counts = genotypes.to_allele_counts()[:, :, 1]  # Get alternate allele counts
        geno_df = pd.DataFrame(
            allele_counts.T,
            index=samples,
            columns=positions
        )
        
        locator = Locator({
            'sample_data': sample_df,
            'genotype_data': geno_df,
            'na_action': 'separate',
            'keras_verbose': 0,
            'max_epochs': 5,
            'out': 'test_windows_holdouts'
        })
        
        result = locator.run_windows_holdouts(
            genotypes=genotypes, 
            samples=samples,
            k=2,
            window_size=3e5,  # 300kb windows
            return_df=True
        )
        
        captured = capsys.readouterr()
        assert "Windows holdout analysis: 7 samples with coordinates, 3 without" in captured.out
        assert "Note: Holdout analysis requires known locations" in captured.out
        assert result is not None
    
    def test_holdout_method_override(self, capsys):
        """Test na_action override at method level for holdout methods."""
        genotypes, samples, sample_df = self.create_test_data(n_samples=10, n_known=7)
        
        # Initialize with 'fail' but override with 'exclude'
        locator = Locator({
            'sample_data': sample_df,
            'na_action': 'fail',
            'keras_verbose': 0,
            'max_epochs': 5,
            'out': 'test_override'
        })
        
        # Should work because we override with 'exclude'
        result = locator.run_holdouts(
            genotypes=genotypes, 
            samples=samples, 
            k=2,
            n_reps=1,
            na_action='exclude',
            return_df=True
        )
        
        captured = capsys.readouterr()
        assert "NA handling mode: exclude" in captured.out
        assert result is not None