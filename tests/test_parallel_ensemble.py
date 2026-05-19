"""Tests for parallel ensemble training functionality."""

import inspect
import os
import tempfile
from unittest.mock import MagicMock, patch

import allel
import numpy as np
import pandas as pd
import pytest

from locator import Locator

# Check if Ray is available
try:
    import ray

    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False


# Skip all tests in this module if Ray is not available
pytestmark = pytest.mark.skipif(not RAY_AVAILABLE, reason="Ray not installed")


class TestParallelEnsemble:
    """Test parallel ensemble training functionality."""

    def create_test_data(self, n_samples=30, n_snps=100):
        """Create test data for ensemble training."""
        np.random.seed(42)
        # Vectorized: per-SNP varying MAF, two independent allele draws
        mafs = np.random.uniform(0.05, 0.45, size=(n_snps, 1))
        allele1 = (np.random.random((n_snps, n_samples)) < mafs).astype(int)
        allele2 = (np.random.random((n_snps, n_samples)) < mafs).astype(int)
        genotype_data = allele1 + allele2

        genotypes = allel.GenotypeArray(genotype_data[:, :, np.newaxis])

        samples = np.array([f"sample_{i:03d}" for i in range(n_samples)])

        coords_df = pd.DataFrame(
            {
                "sampleID": samples,
                "x": np.random.uniform(-100, -80, n_samples),
                "y": np.random.uniform(30, 40, n_samples),
            }
        )

        return genotypes, samples, coords_df

    def test_parallel_train_ensemble_cpu_smoke(self):
        """End-to-end CPU smoke test for the actor-based ensemble path."""
        genotypes, samples, coords_df = self.create_test_data(n_samples=20, n_snps=50)
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                "sample_data": coords_df,
                "max_epochs": 1,
                "patience": 1,
                "width": 8,
                "nlayers": 2,
                "keras_verbose": 0,
                "out": os.path.join(tmpdir, "ens"),
            }
            locator = Locator(config)
            from locator.parallel import parallel_train_ensemble

            result = parallel_train_ensemble(
                locator=locator,
                genotypes=genotypes,
                samples=samples,
                k=2,
                gpu_ids=[],
                gpu_fraction=0.0,
                save_fold_models=False,
                use_model_manager=False,
                verbose=False,
            )
        assert set(result.keys()) >= {
            "histories",
            "models",
            "normalization_params",
            "fold_info",
            "training_time",
            "parallel",
        }
        assert len(result["models"]) == 2
        assert {"meanlong", "meanlat", "sdlong", "sdlat"} <= set(
            result["normalization_params"].keys()
        )

    def test_parallel_train_ensemble_cpu_with_na_samples(self):
        """Ensemble with NA-coord samples runs (excluded from training)."""
        genotypes, samples, coords_df = self.create_test_data(n_samples=20, n_snps=50)
        coords_df = coords_df.copy()
        coords_df.loc[16:, ["x", "y"]] = np.nan
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                "sample_data": coords_df,
                "max_epochs": 1,
                "patience": 1,
                "width": 8,
                "nlayers": 2,
                "keras_verbose": 0,
                "out": os.path.join(tmpdir, "ens_na"),
                "na_action": "exclude",
            }
            locator = Locator(config)
            from locator.parallel import parallel_train_ensemble

            result = parallel_train_ensemble(
                locator=locator,
                genotypes=genotypes,
                samples=samples,
                k=2,
                gpu_ids=[],
                gpu_fraction=0.0,
                save_fold_models=False,
                use_model_manager=False,
                verbose=False,
            )
        assert len(result["models"]) == 2

    # Integration tests (from test_parallel_ensemble_integration.py)

    def test_parallel_ensemble_import(self):
        """Test that parallel_train_ensemble can be imported."""
        from locator.parallel import parallel_train_ensemble

        assert parallel_train_ensemble is not None
        # If it's a stub, that's OK - Ray is not installed
        if parallel_train_ensemble.__name__ == "_not_available":
            pytest.skip("Using stub function - Ray not installed")

    def test_parallel_ensemble_function_exists(self):
        """Test that the parallel ensemble function has correct signature."""
        # Skip this test if we're using the stub function
        from locator.parallel import parallel_train_ensemble

        # If this is the stub function, skip the signature check
        if (
            hasattr(parallel_train_ensemble, "__name__")
            and parallel_train_ensemble.__name__ == "_not_available"
        ):
            pytest.skip("Using stub function - Ray not installed")

        sig = inspect.signature(parallel_train_ensemble)
        params = list(sig.parameters.keys())

        # Check key parameters exist
        assert "locator" in params
        assert "genotypes" in params
        assert "samples" in params
        assert "k" in params
        assert "gpu_ids" in params
        assert "gpu_fraction" in params

    def test_ensemble_mixin_has_required_methods(self):
        """Test that EnsembleMixin has all required methods for parallel training."""
        _, _, coords_df = self.create_test_data(n_samples=20, n_snps=50)

        config = {
            "sample_data": coords_df,
            "max_epochs": 1,
            "keras_verbose": 0,
        }

        locator = Locator(config)

        # Check that all required methods exist
        assert hasattr(locator, "create_ensemble_folds")
        assert hasattr(locator, "_train_single_fold")
        assert hasattr(locator, "_filter_genotypes")
        assert hasattr(locator, "_average_normalization_params")
        assert hasattr(locator, "setup_ensemble_gpu_optimization")
        assert hasattr(locator, "get_ensemble_batch_size")
        assert hasattr(locator, "create_ensemble_early_stopping")
        assert hasattr(locator, "create_ensemble_lr_scheduler")
        assert hasattr(locator, "_clear_fold_memory")

    def test_train_single_fold_works(self):
        """Test that _train_single_fold method works correctly."""
        genotypes, samples, coords_df = self.create_test_data(n_samples=20, n_snps=50)

        config = {
            "sample_data": coords_df,
            "max_epochs": 1,
            "keras_verbose": 0,
        }

        locator = Locator(config)

        # Create fold info
        fold_info = locator.create_ensemble_folds(genotypes, samples, k=2)

        # Filter genotypes
        filtered_genotypes = locator._filter_genotypes(genotypes)

        # Get locations
        _, locs = locator.sort_samples(samples)

        # Train single fold
        model_info = locator._train_single_fold(
            fold_idx=0,
            index_set=fold_info["index_sets"][0],
            filtered_genotypes=filtered_genotypes,
            samples=samples,
            locs=locs,
            verbose=False,
        )

        # Check result structure
        assert "fold" in model_info
        assert "model" in model_info
        assert "history" in model_info
        assert "norm_params" in model_info
        assert "train_indices" in model_info
        assert "val_indices" in model_info

        assert model_info["fold"] == 0
        assert model_info["norm_params"]["meanlong"] is not None
        assert model_info["norm_params"]["meanlat"] is not None

    def test_parallel_ensemble_module_structure(self):
        """Test that parallel module is properly structured."""
        # Check that parallel module exists
        import locator.parallel

        # Check __all__ exports
        assert hasattr(locator.parallel, "__all__")
        exports = locator.parallel.__all__

        assert "parallel_train_ensemble" in exports
        assert "parallel_k_fold_holdouts" in exports
        assert "parallel_holdouts" in exports
        assert "parallel_windows_holdouts" in exports

        # If functions are stubs, that's OK
        if hasattr(locator.parallel.parallel_train_ensemble, "__name__"):
            if locator.parallel.parallel_train_ensemble.__name__ == "_not_available":
                pytest.skip("Using stub functions - Ray not installed")
