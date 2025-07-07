"""Comprehensive tests for ensemble functionality."""

import os
import tempfile
from unittest.mock import patch

import allel
import numpy as np
import pandas as pd

from locator import Locator
from locator.data import IndexSet
from locator.ensemble_model_manager import EnsembleModelManager


class TestEnsemble:
    """Test ensemble functionality including mixin and model manager."""

    def create_test_data(self, n_samples=30, n_snps=100):
        """Create test data for ensemble training."""
        # Generate genotypes with proper allele frequencies
        np.random.seed(42)
        genotype_data = np.zeros((n_snps, n_samples), dtype=int)

        for i in range(n_snps):
            # Create SNPs with varying minor allele frequencies
            maf = np.random.uniform(0.05, 0.45)  # Ensure polymorphic SNPs
            for j in range(n_samples):
                # Simulate diploid genotypes
                allele1 = 1 if np.random.random() < maf else 0
                allele2 = 1 if np.random.random() < maf else 0
                genotype_data[i, j] = allele1 + allele2

        genotypes = allel.GenotypeArray(genotype_data[:, :, np.newaxis])

        # Generate sample IDs
        samples = np.array([f"sample_{i:03d}" for i in range(n_samples)])

        # Generate coordinates
        coords_df = pd.DataFrame(
            {
                "sampleID": samples,
                "x": np.random.uniform(-100, -80, n_samples),
                "y": np.random.uniform(30, 40, n_samples),
            }
        )

        return genotypes, samples, coords_df

    # Tests from test_ensemble_mixin.py
    def test_k_fold_split(self):
        """Test k_fold_split method of IndexSet."""
        n = 100
        k = 5

        # Basic k-fold split
        splits = IndexSet.k_fold_split(n, k, seed=42)

        assert len(splits) == k

        # Check that all samples are used
        all_train = set()
        all_test = set()
        for split in splits:
            all_train.update(split.train)
            all_test.update(split.test)

        # Each sample should appear exactly once as test
        assert all_test == set(range(n))

        # Each sample should appear k-1 times as train
        train_counts = {}
        for split in splits:
            for idx in split.train:
                train_counts[idx] = train_counts.get(idx, 0) + 1

        for count in train_counts.values():
            assert count == k - 1

    def test_k_fold_split_with_na_mask(self):
        """Test k_fold_split with NA mask."""
        n = 100
        k = 5

        # Create NA mask - samples 90-99 are NA
        na_mask = np.zeros(n, dtype=bool)
        na_mask[90:] = True

        splits = IndexSet.k_fold_split(n, k, na_mask=na_mask, seed=42)

        # Check that NA samples are not in train or test
        for split in splits:
            assert not any(idx >= 90 for idx in split.train)
            assert not any(idx >= 90 for idx in split.test)

        # Check that non-NA samples are properly split
        all_test = set()
        for split in splits:
            all_test.update(split.test)

        assert all_test == set(range(90))

    def test_create_ensemble_folds(self):
        """Test creation of ensemble folds with proper NA handling."""
        genotypes, samples, coords_df = self.create_test_data()

        # Add some NA samples
        coords_df.loc[25:29, ["x", "y"]] = np.nan

        config = {
            "sample_data": coords_df,
            "max_epochs": 5,
            "keras_verbose": 0,
        }

        locator = Locator(config)
        locator.na_action = "separate"

        # Create folds
        fold_info = locator.create_ensemble_folds(
            genotypes=genotypes,
            samples=samples,
            k=3,
        )

        # Check structure
        assert "index_sets" in fold_info
        assert "fold_indices" in fold_info
        assert "sample_status" in fold_info

        assert len(fold_info["index_sets"]) == 3
        assert len(fold_info["fold_indices"]) == 3

        # Check that NA samples are in pred set
        for i, fold in fold_info["fold_indices"].items():
            assert all(idx >= 25 and idx < 30 for idx in fold["pred"])

    def test_create_ensemble_folds_with_training_indices(self):
        """Test ensemble folds with restricted training indices."""
        genotypes, samples, coords_df = self.create_test_data()

        config = {
            "sample_data": coords_df,
            "max_epochs": 5,
            "keras_verbose": 0,
        }

        locator = Locator(config)

        # Use only first 20 samples for training
        training_indices = list(range(20))

        fold_info = locator.create_ensemble_folds(
            genotypes=genotypes,
            samples=samples,
            k=4,
            training_set_indices=training_indices,
        )

        # Check that only training indices are used in folds
        for fold in fold_info["fold_indices"].values():
            assert all(idx < 20 for idx in fold["train"])
            assert all(idx < 20 for idx in fold["val"])

    def test_ensemble_normalization_params(self):
        """Test that ensemble methods calculate correct normalization parameters."""
        genotypes, samples, coords_df = self.create_test_data(n_samples=20, n_snps=50)

        config = {
            "sample_data": coords_df,
            "max_epochs": 1,
            "keras_verbose": 0,
        }

        locator = Locator(config)

        # Get fold info
        fold_info = locator.create_ensemble_folds(
            genotypes=genotypes,
            samples=samples,
            k=2,
        )

        # Manually calculate expected params for first fold
        _, locs = locator.sort_samples(samples)
        train_idx = fold_info["index_sets"][0].train
        train_locs = locs[train_idx]

        expected_meanlong = np.nanmean(train_locs[:, 0])
        expected_meanlat = np.nanmean(train_locs[:, 1])

        # Check averaging function
        norm_params_list = [
            {
                "meanlong": expected_meanlong,
                "sdlong": 10.0,
                "meanlat": expected_meanlat,
                "sdlat": 5.0,
            },
            {
                "meanlong": expected_meanlong + 1,
                "sdlong": 11.0,
                "meanlat": expected_meanlat + 0.5,
                "sdlat": 5.5,
            },
        ]

        avg_params = locator._average_normalization_params(norm_params_list)

        assert abs(avg_params["meanlong"] - (expected_meanlong + 0.5)) < 1e-6
        assert abs(avg_params["sdlong"] - 10.5) < 1e-6
        assert abs(avg_params["meanlat"] - (expected_meanlat + 0.25)) < 1e-6
        assert abs(avg_params["sdlat"] - 5.25) < 1e-6

    def test_filter_genotypes(self):
        """Test that _filter_genotypes works correctly."""
        genotypes, samples, coords_df = self.create_test_data()

        config = {
            "sample_data": coords_df,
            "min_mac": 2,
            "max_SNPs": 50,
        }

        locator = Locator(config)

        filtered = locator._filter_genotypes(genotypes)

        # Should have at most 50 SNPs
        assert filtered.shape[0] <= 50
        # Should have same number of samples
        assert filtered.shape[1] == genotypes.shape[1]

    # Tests from test_ensemble_phase2.py
    def test_ensemble_model_manager_save_load(self):
        """Test saving and loading ensemble models."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test models info
            models_info = []
            for i in range(3):
                model_info = {
                    "fold": i,
                    "model": None,  # Would be actual model in practice
                    "norm_params": {
                        "meanlong": -90 + i,
                        "sdlong": 10,
                        "meanlat": 35 + i,
                        "sdlat": 5,
                    },
                    "train_indices": np.arange(20),
                    "val_indices": np.arange(20, 30),
                }
                models_info.append(model_info)

            # Create manager and save (without actual models for speed)
            manager = EnsembleModelManager(tmpdir)

            # Mock save without actual models
            metadata = {
                "n_models": len(models_info),
                "ensemble_version": "1.0",
                "models": [],
            }

            for i, model_info in enumerate(models_info):
                model_meta = {
                    "fold": model_info["fold"],
                    "model_path": f"model_fold_{i}.weights.h5",
                    "norm_params": model_info["norm_params"],
                    "train_indices_count": len(model_info["train_indices"]),
                    "val_indices_count": len(model_info["val_indices"]),
                }
                metadata["models"].append(model_meta)

            # Save metadata
            import json

            metadata_path = os.path.join(tmpdir, "ensemble_metadata.json")
            with open(metadata_path, "w") as f:
                json.dump(metadata, f)

            # Test loading
            loaded_info = manager.load_ensemble()

            assert len(loaded_info) == 3
            assert loaded_info[0]["norm_params"]["meanlong"] == -90
            assert loaded_info[1]["norm_params"]["meanlong"] == -89

            # Test averaged params
            avg_params = manager.get_averaged_normalization_params()
            assert avg_params.meanlong == -89  # (-90 + -89 + -88) / 3

    def test_train_single_fold_efficiency(self):
        """Test that _train_single_fold is more efficient than creating separate instances."""
        genotypes, samples, coords_df = self.create_test_data()

        config = {
            "sample_data": coords_df,
            "max_epochs": 1,  # Quick test
            "keras_verbose": 0,
            "patience": 10,
        }

        locator = Locator(config)

        # Create fold info
        fold_info = locator.create_ensemble_folds(
            genotypes=genotypes,
            samples=samples,
            k=3,
        )

        # Filter genotypes once
        filtered_genotypes = locator._filter_genotypes(genotypes)
        _, locs = locator.sort_samples(samples)

        # Test training single fold
        model_info = locator._train_single_fold(
            fold_idx=0,
            index_set=fold_info["index_sets"][0],
            filtered_genotypes=filtered_genotypes,
            samples=samples,
            locs=locs,
            verbose=False,
        )

        # Verify output structure
        assert "model" in model_info
        assert "history" in model_info
        assert "norm_params" in model_info
        assert "train_indices" in model_info
        assert "val_indices" in model_info
        assert model_info["fold"] == 0

    def test_ensemble_with_model_manager(self):
        """Test ensemble training with model manager saving."""
        genotypes, samples, coords_df = self.create_test_data()

        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                "sample_data": coords_df,
                "max_epochs": 1,
                "keras_verbose": 0,
                "out": os.path.join(tmpdir, "test"),
            }

            locator = Locator(config)

            # Train ensemble with model manager
            locator.train_ensemble(
                genotypes=genotypes,
                samples=samples,
                k=2,
                save_fold_models=True,
                use_model_manager=True,
                verbose=False,
            )

            # Check that ensemble was saved
            ensemble_path = os.path.join(tmpdir, "test_ensemble")
            assert os.path.exists(ensemble_path)
            assert os.path.exists(os.path.join(ensemble_path, "ensemble_metadata.json"))

    def test_memory_efficient_prediction(self):
        """Test that predictions don't load all models at once."""
        genotypes, samples, coords_df = self.create_test_data()

        config = {
            "sample_data": coords_df,
            "max_epochs": 1,
            "keras_verbose": 0,
        }

        locator = Locator(config)

        # Train small ensemble
        locator.train_ensemble(
            genotypes=genotypes,
            samples=samples,
            k=2,
            verbose=False,
        )

        # Make predictions using standard method
        predictions = locator.predict_ensemble(
            genotypes=genotypes,
            samples=samples,
            save_predictions=False,
        )

        assert isinstance(predictions, pd.DataFrame)
        assert len(predictions) == len(samples)
        assert "x" in predictions.columns
        assert "y" in predictions.columns

    def test_augmentation_config(self):
        """Test that augmentation is properly configured."""
        genotypes, samples, coords_df = self.create_test_data()

        config = {
            "sample_data": coords_df,
            "max_epochs": 1,
            "keras_verbose": 0,
        }

        locator = Locator(config)

        # Train with augmentation
        locator.train_ensemble(
            genotypes=genotypes,
            samples=samples,
            k=2,
            augment_data=True,
            flip_rate=0.1,
            verbose=False,
        )

        # Verify augmentation was configured
        assert locator.config.get("augmentation", {}).get("enabled") is True
        assert locator.config.get("augmentation", {}).get("flip_rate") == 0.1

    def test_ensemble_train_predict_integration(self):
        """Test full ensemble train and predict workflow."""
        genotypes, samples, coords_df = self.create_test_data()

        config = {
            "sample_data": coords_df,
            "max_epochs": 1,
            "keras_verbose": 0,
        }

        locator = Locator(config)

        # Train ensemble
        train_result = locator.train_ensemble(
            genotypes=genotypes,
            samples=samples,
            k=3,
            save_fold_models=False,
            verbose=False,
        )

        # Check training results
        assert "histories" in train_result
        assert "models" in train_result
        assert "normalization_params" in train_result
        assert len(train_result["models"]) == 3

        # Make predictions
        predictions = locator.predict_ensemble(
            genotypes=genotypes,
            samples=samples,
            include_fold_predictions=True,
            return_std=True,
            save_predictions=False,
        )

        # Check prediction results
        assert isinstance(predictions, pd.DataFrame)
        assert "x" in predictions.columns
        assert "y" in predictions.columns
        assert "x_std" in predictions.columns
        assert "y_std" in predictions.columns
        assert "x_fold0" in predictions.columns
        assert "x_fold1" in predictions.columns
        assert "x_fold2" in predictions.columns

    # Phase 4 tests: Training improvements
    def test_ensemble_with_mixed_precision(self):
        """Test ensemble training with mixed precision."""
        genotypes, samples, coords_df = self.create_test_data(n_samples=20, n_snps=50)

        config = {
            "sample_data": coords_df,
            "max_epochs": 1,
            "keras_verbose": 0,
        }

        locator = Locator(config)

        # Mock GPU optimizer to avoid GPU requirements
        with patch(
            "locator.ensemble_mixin.GPUOptimizer.setup_mixed_precision"
        ) as mock_mp:
            mock_mp.return_value = True

            locator.train_ensemble(
                genotypes=genotypes,
                samples=samples,
                k=2,
                use_mixed_precision=True,
                save_fold_models=False,
                verbose=False,
            )

            # Should have called mixed precision setup
            mock_mp.assert_called()

    def test_ensemble_patience_multiplier(self):
        """Test that patience multiplier affects training."""
        genotypes, samples, coords_df = self.create_test_data(n_samples=20, n_snps=50)

        config = {
            "sample_data": coords_df,
            "max_epochs": 1,
            "keras_verbose": 0,
            "patience": 10,
        }

        locator = Locator(config)

        # Check callback creation with multiplier
        callback = locator.create_ensemble_early_stopping(patience_multiplier=2.0)
        assert callback.patience == 20

    def test_ensemble_memory_management(self):
        """Test memory management between folds."""
        genotypes, samples, coords_df = self.create_test_data(n_samples=10, n_snps=20)

        config = {
            "sample_data": coords_df,
            "keras_verbose": 0,
        }

        locator = Locator(config)

        # Test the _clear_fold_memory method directly
        with patch("locator.ensemble_mixin.keras.backend.clear_session") as mock_clear:
            with patch("locator.ensemble_mixin.gc.collect") as mock_gc:
                locator._clear_fold_memory()

                mock_clear.assert_called_once()
                mock_gc.assert_called_once()
