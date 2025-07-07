"""Tests for model metadata persistence and loading."""

import json
import os
import tempfile
from pathlib import Path

import allel
import h5py
import numpy as np
import pandas as pd
import pytest

from locator.core import Locator


class TestModelPersistence:
    """Test saving and loading model metadata in HDF5 files."""

    @pytest.fixture
    def sample_data(self):
        """Create sample genotype and location data."""
        np.random.seed(42)
        n_samples = 50
        n_snps = 100

        # Create genotype data as numpy array with proper biallelic SNPs
        # Make sure most SNPs are biallelic (0 or 1 alleles only)
        geno_array = np.zeros((n_snps, n_samples, 2), dtype=np.int8)
        for i in range(n_snps):
            # Create biallelic genotypes (0/0, 0/1, 1/1)
            for j in range(n_samples):
                allele_count = np.random.choice([0, 1, 2], p=[0.25, 0.5, 0.25])
                if allele_count == 0:
                    geno_array[i, j, :] = [0, 0]
                elif allele_count == 1:
                    geno_array[i, j, :] = [0, 1]
                else:
                    geno_array[i, j, :] = [1, 1]

        # Create real GenotypeArray from allel
        genotypes = allel.GenotypeArray(geno_array)

        # Create sample IDs
        samples = np.array([f"sample_{i:03d}" for i in range(n_samples)])

        # Create location data (some with NA)
        locs = np.random.uniform(-180, 180, size=(n_samples, 2))
        # Make last 10 samples have NA locations
        locs[-10:] = np.nan

        # Create sample data DataFrame
        sample_df = pd.DataFrame({"sampleID": samples, "x": locs[:, 0], "y": locs[:, 1]})

        return genotypes, samples, sample_df

    def test_save_model_metadata(self, sample_data):
        """Test that model metadata is saved correctly to HDF5."""
        genotypes, samples, sample_df = sample_data

        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                "out": os.path.join(tmpdir, "test_model"),
                "sample_data": sample_df,
                "max_epochs": 2,  # Quick training
                "patience": 1,
                "keras_verbose": 0,
                "min_mac": 3,
                "max_SNPs": 50,
                "impute_missing": True,
            }

            # Train model
            loc = Locator(config)
            loc.train(genotypes=genotypes, samples=samples)

            # Check that weights file exists
            weights_path = f"{config['out']}.weights.h5"
            assert os.path.exists(weights_path)

            # Check metadata in HDF5 file
            with h5py.File(weights_path, "r") as f:
                # Check normalization parameters
                assert "coord_meanlong" in f.attrs
                assert "coord_sdlong" in f.attrs
                assert "coord_meanlat" in f.attrs
                assert "coord_sdlat" in f.attrs

                # Check preprocessing parameters
                assert f.attrs["min_mac"] == 3
                assert f.attrs["max_SNPs"] == 50
                assert f.attrs["impute_missing"] == True

                # Check other metadata
                assert "n_samples" in f.attrs
                assert "n_snps" in f.attrs
                assert "metadata_version" in f.attrs
                assert "locator_version" in f.attrs
                assert "save_date" in f.attrs
                assert "config_json" in f.attrs

                # Validate config JSON
                config_loaded = json.loads(f.attrs["config_json"])
                assert config_loaded["min_mac"] == 3
                assert config_loaded["max_SNPs"] == 50

    def test_load_model_metadata(self, sample_data):
        """Test loading model metadata from HDF5."""
        genotypes, samples, sample_df = sample_data

        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                "out": os.path.join(tmpdir, "test_model"),
                "sample_data": sample_df,
                "max_epochs": 2,
                "patience": 1,
                "keras_verbose": 0,
            }

            # Train and save model
            loc1 = Locator(config)
            loc1.train(genotypes=genotypes, samples=samples)

            # Store normalization params for comparison
            orig_meanlong = loc1.meanlong
            orig_sdlong = loc1.sdlong
            orig_meanlat = loc1.meanlat
            orig_sdlat = loc1.sdlat

            # Create new Locator instance and load model
            loc2 = Locator(config)
            weights_path = f"{config['out']}.weights.h5"
            metadata = loc2.load_model(weights_path)

            # Check that normalization params were loaded correctly
            assert abs(loc2.meanlong - orig_meanlong) < 1e-6
            assert abs(loc2.sdlong - orig_sdlong) < 1e-6
            assert abs(loc2.meanlat - orig_meanlat) < 1e-6
            assert abs(loc2.sdlat - orig_sdlat) < 1e-6

            # Check metadata structure
            assert "normalization" in metadata
            assert "preprocessing" in metadata
            assert metadata["n_samples"] == 50  # Total samples, not just training
            assert metadata["n_snps"] > 0

    def test_predict_from_weights(self, sample_data):
        """Test making predictions from saved weights."""
        genotypes, samples, sample_df = sample_data

        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                "out": os.path.join(tmpdir, "test_model"),
                "sample_data": sample_df,
                "max_epochs": 2,
                "patience": 1,
                "keras_verbose": 0,
                "min_mac": 2,
                "max_SNPs": 80,
            }

            # Train and save model
            loc1 = Locator(config)
            loc1.train(genotypes=genotypes, samples=samples)

            # Create new instance and predict from weights
            loc2 = Locator(config)
            weights_path = f"{config['out']}.weights.h5"

            # Make predictions using saved weights
            predictions = loc2.predict_from_weights(
                weights_path=weights_path,
                genotypes=genotypes,
                samples=samples,
                return_df=True,
                save_preds_to_disk=False,
            )

            # Check predictions
            assert isinstance(predictions, pd.DataFrame)
            assert len(predictions) == 10  # Only NA samples
            assert "sampleID" in predictions.columns
            assert "x" in predictions.columns
            assert "y" in predictions.columns

            # Check that preprocessing was applied with same parameters
            assert loc2.predgen.shape[1] <= 80  # max_SNPs applied

    def test_bootstrap_metadata_persistence(self, sample_data):
        """Test metadata persistence for bootstrap models."""
        genotypes, samples, sample_df = sample_data

        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                "out": os.path.join(tmpdir, "test_model"),
                "sample_data": sample_df,
                "max_epochs": 2,
                "patience": 1,
                "keras_verbose": 0,
                "bootstrap": True,
            }

            # Train bootstrap model
            loc = Locator(config)
            loc.train(genotypes=genotypes, samples=samples, boot=0)

            # Check bootstrap weights file
            weights_path = f"{config['out']}_boot0.weights.h5"
            assert os.path.exists(weights_path)

            # Check metadata
            with h5py.File(weights_path, "r") as f:
                assert "coord_meanlong" in f.attrs
                assert "metadata_version" in f.attrs

    def test_backward_compatibility(self, sample_data):
        """Test loading models without metadata (backward compatibility)."""
        genotypes, samples, sample_df = sample_data

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a weights file with valid HDF5 but no metadata attrs
            weights_path = os.path.join(tmpdir, "old_model.weights.h5")

            # Create HDF5 file without metadata attributes
            with h5py.File(weights_path, "w") as f:
                # Add a dummy dataset to make it a valid HDF5 file
                f.create_dataset("dummy", data=np.array([1.0]))

            # Try to load it
            config = {"out": os.path.join(tmpdir, "test"), "sample_data": sample_df}
            loc = Locator(config)

            # Load the model - should succeed with defaults since attrs.get() provides defaults
            metadata = loc.load_model(weights_path)

            # Check that defaults were used
            assert loc.meanlong == 0.0
            assert loc.sdlong == 1.0
            assert loc.meanlat == 0.0
            assert loc.sdlat == 1.0

            # Check metadata structure exists (loaded with defaults)
            assert metadata is not None
            assert "normalization" in metadata
            assert metadata["normalization"]["meanlong"] == 0.0
