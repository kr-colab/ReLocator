"""Simple integration test for model metadata persistence."""

import pytest
import numpy as np
import pandas as pd
import tempfile
import os
import h5py
import json

# Import after sys.path modification
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from locator.core import Locator


def test_metadata_persistence_roundtrip():
    """Test that we can save and load model metadata correctly."""
    
    # Create simple test data that won't require complex mocking
    np.random.seed(42)
    n_samples = 20
    n_snps = 50
    
    # Create genotype DataFrame (bypasses allel complexity)
    positions = list(range(1000, 1000 + n_snps))
    geno_data = np.random.randint(0, 3, size=(n_samples, n_snps))
    geno_df = pd.DataFrame(
        geno_data,
        index=[f"sample_{i:03d}" for i in range(n_samples)],
        columns=positions
    )
    
    # Create location DataFrame
    sample_df = pd.DataFrame({
        'sampleID': [f"sample_{i:03d}" for i in range(n_samples)],
        'x': np.random.uniform(-180, 180, n_samples),
        'y': np.random.uniform(-90, 90, n_samples)
    })
    # Make last 5 samples have NA locations
    sample_df.loc[15:, ['x', 'y']] = np.nan
    
    with tempfile.TemporaryDirectory() as tmpdir:
        config = {
            'out': os.path.join(tmpdir, 'test_model'),
            'sample_data': sample_df,
            'genotype_data': geno_df,
            'max_epochs': 2,
            'patience': 1,
            'keras_verbose': 0,
            'min_mac': 3,
            'max_SNPs': 30,
            'impute_missing': True,
            'width': 128,
            'nlayers': 4
        }
        
        # Train model
        print("Training model...")
        loc1 = Locator(config)
        
        # Write genotype data to file and load it properly
        matrix_file = os.path.join(tmpdir, 'genotypes.txt')
        geno_df.to_csv(matrix_file, sep='\t')
        
        # Load using the proper method that creates GenotypeArray
        genotypes, samples = loc1.load_genotypes(matrix=matrix_file)
        
        loc1.train(genotypes=genotypes, samples=samples)
        
        # Check weights file exists and has metadata
        weights_path = f"{config['out']}.weights.h5"
        assert os.path.exists(weights_path)
        
        print("Checking saved metadata...")
        with h5py.File(weights_path, 'r') as f:
            # Verify all expected attributes exist
            assert 'coord_meanlong' in f.attrs
            assert 'coord_sdlong' in f.attrs
            assert 'coord_meanlat' in f.attrs
            assert 'coord_sdlat' in f.attrs
            assert f.attrs['min_mac'] == 3
            assert f.attrs['max_SNPs'] == 30
            assert f.attrs['impute_missing'] == True
            
            # Store values for comparison
            saved_meanlong = f.attrs['coord_meanlong']
            saved_sdlong = f.attrs['coord_sdlong']
            
        # Create new Locator and load the model
        print("Loading model in new instance...")
        loc2 = Locator(config)
        metadata = loc2.load_model(weights_path)
        
        # Verify metadata was loaded correctly
        assert abs(loc2.meanlong - saved_meanlong) < 1e-6
        assert abs(loc2.sdlong - saved_sdlong) < 1e-6
        assert metadata['preprocessing']['min_mac'] == 3
        assert metadata['preprocessing']['max_SNPs'] == 30
        
        print("Test passed!")


if __name__ == "__main__":
    test_metadata_persistence_roundtrip()