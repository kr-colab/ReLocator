"""Tests for verbosity control features in Locator."""

import pytest
import numpy as np
import pandas as pd
import allel
from locator import Locator
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to suppress plots
import matplotlib.pyplot as plt
from io import StringIO
import sys


class TestVerbosityControl:
    """Test suite for verbosity control features."""
    
    def test_default_verbosity_settings(self):
        """Test that verbosity options default to False."""
        locator = Locator()
        assert locator.config.get('verbose_splits', False) == False
        assert locator.config.get('verbose_batch_size', False) == False
    
    def test_verbose_splits_training(self, genotype_data, sample_data_file, capsys, tmp_path):
        """Test verbose_splits output during training."""
        genotypes, samples, coords, n_samples, n_snps = genotype_data
        
        config = {
            'sample_data': str(sample_data_file),
            'out': str(tmp_path / 'test_verbose'),
            'verbose_splits': True,
            'max_epochs': 1,
            'keras_verbose': 0,
            'disable_gpu': True,
        }
        
        locator = Locator(config)
        locator.train(genotypes=genotypes, samples=samples)
        
        # Capture output
        captured = capsys.readouterr()
        
        # Check for expected output
        assert "Data split summary:" in captured.out
        assert "Training samples:" in captured.out
        assert "Validation samples:" in captured.out
        assert "Prediction samples (no coords):" in captured.out
        assert "Total samples:" in captured.out
        assert "Total SNPs:" in captured.out
        
        # Verify counts are correct
        assert f"Total samples: {n_samples}" in captured.out
        assert f"Total SNPs: {n_snps}" in captured.out
    
    def test_verbose_splits_holdout(self, genotype_data, sample_data_file, capsys, tmp_path):
        """Test verbose_splits output during holdout training."""
        genotypes, samples, coords, n_samples, n_snps = genotype_data
        
        config = {
            'sample_data': str(sample_data_file),
            'out': str(tmp_path / 'test_verbose_holdout'),
            'verbose_splits': True,
            'max_epochs': 1,
            'keras_verbose': 0,
            'disable_gpu': True,
        }
        
        locator = Locator(config)
        locator.train_holdout(genotypes=genotypes, samples=samples, k=10)
        
        # Capture output
        captured = capsys.readouterr()
        
        # Check for expected output
        assert "Holdout split summary:" in captured.out
        assert "Training samples:" in captured.out
        assert "Validation samples:" in captured.out
        assert "Holdout samples:" in captured.out
        assert "Total samples:" in captured.out
        assert "Total SNPs:" in captured.out
    
    def test_quiet_mode_splits(self, genotype_data, sample_data_file, capsys, tmp_path):
        """Test that split info is not printed when verbose_splits=False."""
        genotypes, samples, coords, n_samples, n_snps = genotype_data
        
        config = {
            'sample_data': str(sample_data_file),
            'out': str(tmp_path / 'test_quiet'),
            'verbose_splits': False,  # Explicitly set to False
            'max_epochs': 1,
            'keras_verbose': 0,
            'disable_gpu': True,
        }
        
        locator = Locator(config)
        locator.train(genotypes=genotypes, samples=samples)
        
        # Capture output
        captured = capsys.readouterr()
        
        # Should NOT contain split summary
        assert "Data split summary:" not in captured.out
        assert "Training samples:" not in captured.out or "Training data:" in captured.out  # Allow for other training messages
    
    def test_verbose_batch_size_cpu(self, genotype_data, sample_data_file, capsys, tmp_path):
        """Test verbose_batch_size with CPU (should not print GPU optimization info)."""
        genotypes, samples, coords, n_samples, n_snps = genotype_data
        
        config = {
            'sample_data': str(sample_data_file),
            'out': str(tmp_path / 'test_batch_cpu'),
            'verbose_batch_size': True,
            'batch_size': 32,
            'max_epochs': 1,
            'keras_verbose': 0,
            'disable_gpu': True,  # Force CPU
        }
        
        locator = Locator(config)
        locator.train(genotypes=genotypes, samples=samples)
        
        # With CPU and fixed batch size, no optimization messages should appear
        captured = capsys.readouterr()
        assert "Optimal batch size determined:" not in captured.out
        assert "Using optimized batch size:" not in captured.out
    
    def test_verbose_batch_size_auto(self, genotype_data, sample_data_file, monkeypatch, capsys, tmp_path):
        """Test verbose_batch_size with gpu_batch_size='auto'."""
        genotypes, samples, coords, n_samples, n_snps = genotype_data
        
        # Mock GPU availability
        def mock_list_physical_devices(device_type):
            if device_type == 'GPU':
                # Create a mock GPU device
                class MockDevice:
                    name = "NVIDIA GeForce RTX 3090"
                return [MockDevice()]
            return []
        
        monkeypatch.setattr('tensorflow.config.list_physical_devices', mock_list_physical_devices)
        
        config = {
            'sample_data': str(sample_data_file),
            'out': str(tmp_path / 'test_batch_auto'),
            'verbose_batch_size': True,
            'gpu_batch_size': 'auto',
            'max_epochs': 1,
            'keras_verbose': 0,
            'disable_gpu': False,
        }
        
        locator = Locator(config)
        
        # Since we're mocking, GPU optimization might fail, but verbose messages should attempt
        try:
            locator.train(genotypes=genotypes, samples=samples)
        except:
            pass  # OK if it fails, we're just testing verbosity
        
        captured = capsys.readouterr()
        # Should attempt to print optimization info (even if it fails)
        assert "Using optimized batch size:" in captured.out or "Failed to optimize batch size:" in captured.out
    
    def test_both_verbose_options(self, genotype_data, sample_data_file, capsys, tmp_path):
        """Test with both verbose options enabled."""
        genotypes, samples, coords, n_samples, n_snps = genotype_data
        
        config = {
            'sample_data': str(sample_data_file),
            'out': str(tmp_path / 'test_both_verbose'),
            'verbose_splits': True,
            'verbose_batch_size': True,
            'batch_size': 16,  # Fixed batch size
            'max_epochs': 1,
            'keras_verbose': 0,
            'disable_gpu': True,
        }
        
        locator = Locator(config)
        locator.train_holdout(genotypes=genotypes, samples=samples, k=5)
        
        # Capture output
        captured = capsys.readouterr()
        
        # Should see holdout split info
        assert "Holdout split summary:" in captured.out
        assert "Holdout samples: 5" in captured.out
        
        # With fixed batch size and CPU, no batch optimization messages
        assert "Optimal batch size determined:" not in captured.out
    
    def test_percentage_calculations(self, genotype_data, sample_data_file, capsys, tmp_path):
        """Test that percentage calculations in verbose output are correct."""
        genotypes, samples, coords, n_samples, n_snps = genotype_data
        
        config = {
            'sample_data': str(sample_data_file),
            'out': str(tmp_path / 'test_percentages'),
            'verbose_splits': True,
            'train_split': 0.8,  # 80% train
            'max_epochs': 1,
            'keras_verbose': 0,
            'disable_gpu': True,
        }
        
        locator = Locator(config)
        locator.train(genotypes=genotypes, samples=samples)
        
        # Capture output
        captured = capsys.readouterr()
        
        # Parse the output to check percentages
        lines = captured.out.split('\n')
        for line in lines:
            if "Training samples:" in line and "%" in line:
                # Extract percentage
                pct = float(line.split('(')[1].split('%')[0])
                # Should be approximately 80% * (45/50) since 5 samples have NA coords
                assert 65 <= pct <= 75  # Allow some variation due to rounding
            elif "Validation samples:" in line and "%" in line:
                pct = float(line.split('(')[1].split('%')[0])
                # Should be approximately 20% * (45/50)
                assert 15 <= pct <= 25