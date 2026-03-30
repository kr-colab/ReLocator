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

    def test_parallel_train_ensemble_basic(self):
        """Test basic parallel ensemble training functionality."""
        genotypes, samples, coords_df = self.create_test_data(n_samples=20, n_snps=50)

        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                "sample_data": coords_df,
                "max_epochs": 1,
                "keras_verbose": 0,
                "out": os.path.join(tmpdir, "test"),
            }

            locator = Locator(config)

            # Mock Ray to avoid actual parallel execution in tests
            with (
                patch("locator.parallel.ensemble.ray") as mock_ray,
                patch("locator.parallel._helpers.ray", mock_ray),
            ):
                # Mock Ray initialization check
                mock_ray.is_initialized.return_value = False
                mock_ray.init.return_value = None

                # Mock Ray worker submission and results
                mock_future = MagicMock()
                mock_ray.remote.return_value = mock_future

                # Create mock results for each fold
                mock_results = []
                for fold_idx in range(3):
                    mock_result = {
                        "fold": fold_idx,
                        "gpu_id": fold_idx % 2,
                        "train_time": 10.0 + fold_idx,
                        "model_info": {
                            "fold": fold_idx,
                            "weights_file": f"{config['out']}_fold{fold_idx}.weights.h5",
                            "norm_params": {
                                "meanlong": -90.0 + fold_idx,
                                "sdlong": 10.0,
                                "meanlat": 35.0 + fold_idx,
                                "sdlat": 5.0,
                            },
                            "train_indices": list(range(10)),
                            "val_indices": list(range(10, 15)),
                        },
                        "history": {
                            "loss": [0.5, 0.4, 0.3],
                            "val_loss": [0.6, 0.5, 0.4],
                        },
                        "final_loss": 0.3,
                        "final_val_loss": 0.4,
                    }
                    mock_results.append(mock_result)

                mock_ray.get.return_value = mock_results
                mock_ray.wait.return_value = ([MagicMock()], [])

                # Mock the worker creation
                with patch(
                    "locator.parallel.ensemble._create_ray_ensemble_worker"
                ) as mock_worker:
                    mock_worker.return_value.remote.return_value = mock_future

                    # Import and call parallel_train_ensemble
                    from locator.parallel import parallel_train_ensemble

                    # Mock time to make the test faster
                    with patch("time.time", side_effect=[0, 10]):
                        result = parallel_train_ensemble(
                            locator=locator,
                            genotypes=genotypes,
                            samples=samples,
                            k=3,
                            gpu_ids=[0, 1],
                            save_fold_models=False,
                            use_model_manager=False,
                            verbose=False,
                        )

                    # Check results
                    assert result is not None
                    assert "histories" in result
                    assert "models" in result
                    assert "normalization_params" in result
                    assert len(result["models"]) == 3

                    # Check that averaged normalization parameters are calculated
                    avg_params = result["normalization_params"]
                    assert "meanlong" in avg_params
                    assert "meanlat" in avg_params

    def test_parallel_vs_sequential_consistency(self):
        """Test that parallel ensemble gives similar results to sequential."""
        genotypes, samples, coords_df = self.create_test_data(n_samples=20, n_snps=50)

        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                "sample_data": coords_df,
                "max_epochs": 1,
                "keras_verbose": 0,
                "out": os.path.join(tmpdir, "test"),
                "seed": 42,
            }

            # Train sequential ensemble
            locator_seq = Locator(config.copy())
            seq_result = locator_seq.train_ensemble(
                genotypes=genotypes,
                samples=samples,
                k=2,
                save_fold_models=False,
                verbose=False,
            )

            # For parallel, we'll mock to return similar structure
            locator_par = Locator(config.copy())

            with (
                patch("locator.parallel.ensemble.ray") as mock_ray,
                patch("locator.parallel._helpers.ray", mock_ray),
            ):
                mock_ray.is_initialized.return_value = False

                # Use sequential results to create parallel mock results
                mock_results = []
                for i in range(2):
                    mock_result = {
                        "fold": i,
                        "gpu_id": i,
                        "train_time": 5.0,
                        "model_info": {
                            "fold": i,
                            "weights_file": None,
                            "norm_params": seq_result["models"][i]["norm_params"],
                            "train_indices": seq_result["models"][i][
                                "train_indices"
                            ].tolist(),
                            "val_indices": seq_result["models"][i][
                                "val_indices"
                            ].tolist(),
                        },
                        "history": {
                            "loss": seq_result["histories"][i].history.get("loss", []),
                            "val_loss": seq_result["histories"][i].history.get(
                                "val_loss", []
                            ),
                        },
                        "final_loss": 0.3,
                        "final_val_loss": 0.4,
                    }
                    mock_results.append(mock_result)

                mock_ray.get.return_value = mock_results

                from locator.parallel import parallel_train_ensemble

                with patch(
                    "locator.parallel.ensemble._create_ray_ensemble_worker"
                ) as mock_worker:
                    mock_future = MagicMock()
                    mock_worker.return_value.remote.return_value = mock_future

                    par_result = parallel_train_ensemble(
                        locator=locator_par,
                        genotypes=genotypes,
                        samples=samples,
                        k=2,
                        gpu_ids=[0, 1],
                        save_fold_models=False,
                        use_model_manager=False,
                        verbose=False,
                    )

                # Compare normalization parameters
                seq_norm = seq_result["normalization_params"]
                par_norm = par_result["normalization_params"]

                assert abs(seq_norm["meanlong"] - par_norm["meanlong"]) < 1e-6
                assert abs(seq_norm["meanlat"] - par_norm["meanlat"]) < 1e-6

    def test_parallel_ensemble_with_na_samples(self):
        """Test parallel ensemble with NA samples."""
        genotypes, samples, coords_df = self.create_test_data(n_samples=20, n_snps=50)

        # Add some NA samples
        coords_df.loc[15:19, ["x", "y"]] = np.nan

        config = {
            "sample_data": coords_df,
            "max_epochs": 1,
            "keras_verbose": 0,
        }

        locator = Locator(config)
        locator.na_action = "separate"

        # Mock Ray
        with (
            patch("locator.parallel.ensemble.ray") as mock_ray,
            patch("locator.parallel._helpers.ray", mock_ray),
        ):
            mock_ray.is_initialized.return_value = False

            # Create minimal mock results
            mock_results = []
            for fold_idx in range(3):
                mock_result = {
                    "fold": fold_idx,
                    "gpu_id": fold_idx % 2,
                    "train_time": 5.0,
                    "model_info": {
                        "fold": fold_idx,
                        "weights_file": None,
                        "norm_params": {
                            "meanlong": -90.0,
                            "sdlong": 10.0,
                            "meanlat": 35.0,
                            "sdlat": 5.0,
                        },
                        "train_indices": list(range(10)),
                        "val_indices": list(range(10, 15)),
                    },
                    "history": {"loss": [0.5], "val_loss": [0.6]},
                    "final_loss": 0.5,
                    "final_val_loss": 0.6,
                }
                mock_results.append(mock_result)

            mock_ray.get.return_value = mock_results

            from locator.parallel import parallel_train_ensemble

            with patch(
                "locator.parallel.ensemble._create_ray_ensemble_worker"
            ) as mock_worker:
                mock_future = MagicMock()
                mock_worker.return_value.remote.return_value = mock_future

                result = parallel_train_ensemble(
                    locator=locator,
                    genotypes=genotypes,
                    samples=samples,
                    k=3,
                    gpu_ids=[0, 1],
                    na_action="separate",
                    save_fold_models=False,
                    use_model_manager=False,
                    verbose=False,
                )

                assert result is not None
                assert "fold_info" in result
                # Should have identified NA samples
                assert result["fold_info"]["sample_status"]["n_na"] == 5

    def test_parallel_ensemble_gpu_assignment(self):
        """Test that GPU assignment works correctly in parallel ensemble."""
        genotypes, samples, coords_df = self.create_test_data(n_samples=20, n_snps=50)

        config = {
            "sample_data": coords_df,
            "max_epochs": 1,
            "keras_verbose": 0,
        }

        locator = Locator(config)

        # Track GPU assignments
        gpu_assignments = []

        def track_gpu_assignment(fold_idx, gpu_id, data):
            gpu_assignments.append((fold_idx, gpu_id))
            # data parameter is required by interface but not used here
            _ = data
            mock_future = MagicMock()
            return mock_future

        with (
            patch("locator.parallel.ensemble.ray") as mock_ray,
            patch("locator.parallel._helpers.ray", mock_ray),
        ):
            mock_ray.is_initialized.return_value = False

            # Create mock results
            mock_results = []
            for fold_idx in range(5):
                mock_result = {
                    "fold": fold_idx,
                    "gpu_id": fold_idx % 3,  # Should match our GPU assignment
                    "train_time": 5.0,
                    "model_info": {
                        "fold": fold_idx,
                        "weights_file": None,
                        "norm_params": {
                            "meanlong": -90.0,
                            "sdlong": 10.0,
                            "meanlat": 35.0,
                            "sdlat": 5.0,
                        },
                        "train_indices": list(range(10)),
                        "val_indices": list(range(10, 15)),
                    },
                    "history": {"loss": [0.5], "val_loss": [0.6]},
                    "final_loss": 0.5,
                    "final_val_loss": 0.6,
                }
                mock_results.append(mock_result)

            mock_ray.get.return_value = mock_results

            from locator.parallel import parallel_train_ensemble

            with patch(
                "locator.parallel.ensemble._create_ray_ensemble_worker"
            ) as mock_worker:
                mock_worker.return_value.remote.side_effect = track_gpu_assignment

                _ = parallel_train_ensemble(
                    locator=locator,
                    genotypes=genotypes,
                    samples=samples,
                    k=5,
                    gpu_ids=[0, 1, 2],  # 3 GPUs
                    save_fold_models=False,
                    use_model_manager=False,
                    verbose=False,
                )

                # Check GPU assignments
                expected_assignments = [
                    (0, 0),  # Fold 0 -> GPU 0
                    (1, 1),  # Fold 1 -> GPU 1
                    (2, 2),  # Fold 2 -> GPU 2
                    (3, 0),  # Fold 3 -> GPU 0 (round-robin)
                    (4, 1),  # Fold 4 -> GPU 1
                ]

                assert gpu_assignments == expected_assignments

    @pytest.mark.skipif(
        "CI" in os.environ,
        reason="Skip model manager test in CI to avoid file I/O issues",
    )
    def test_parallel_ensemble_with_model_manager(self):
        """Test parallel ensemble with model manager saving."""
        genotypes, samples, coords_df = self.create_test_data(n_samples=20, n_snps=50)

        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                "sample_data": coords_df,
                "max_epochs": 1,
                "keras_verbose": 0,
                "out": os.path.join(tmpdir, "test"),
            }

            locator = Locator(config)

            # Mock Ray and model creation
            with (
                patch("locator.parallel.ensemble.ray") as mock_ray,
                patch("locator.parallel._helpers.ray", mock_ray),
            ):
                mock_ray.is_initialized.return_value = False

                # Create mock results
                mock_results = []
                for fold_idx in range(2):
                    mock_result = {
                        "fold": fold_idx,
                        "gpu_id": fold_idx,
                        "train_time": 5.0,
                        "model_info": {
                            "fold": fold_idx,
                            "weights_file": f"{config['out']}_fold{fold_idx}.weights.h5",
                            "norm_params": {
                                "meanlong": -90.0,
                                "sdlong": 10.0,
                                "meanlat": 35.0,
                                "sdlat": 5.0,
                            },
                            "train_indices": list(range(10)),
                            "val_indices": list(range(10, 15)),
                        },
                        "history": {"loss": [0.5], "val_loss": [0.6]},
                        "final_loss": 0.5,
                        "final_val_loss": 0.6,
                    }
                    mock_results.append(mock_result)

                mock_ray.get.return_value = mock_results

                from locator.parallel import parallel_train_ensemble

                with patch(
                    "locator.parallel.ensemble._create_ray_ensemble_worker"
                ) as mock_worker:
                    mock_future = MagicMock()
                    mock_worker.return_value.remote.return_value = mock_future

                    # Mock model creation and loading
                    with patch.object(locator, "_create_model") as mock_create_model:
                        mock_model = MagicMock()
                        mock_create_model.return_value = mock_model

                        # Mock os.path.exists so fake weight files pass the check
                        with patch(
                            "locator.parallel.ensemble.os.path.exists",
                            return_value=True,
                        ):
                            # Mock EnsembleModelManager
                            with patch(
                                "locator.ensemble_model_manager.EnsembleModelManager"
                            ) as mock_manager_class:
                                mock_manager = MagicMock()
                                mock_manager_class.return_value = mock_manager

                                _ = parallel_train_ensemble(
                                    locator=locator,
                                    genotypes=genotypes,
                                    samples=samples,
                                    k=2,
                                    gpu_ids=[0, 1],
                                    save_fold_models=True,
                                    use_model_manager=True,
                                    verbose=False,
                                )

                                # Verify model manager was used
                                mock_manager_class.assert_called_once()
                                mock_manager.save_ensemble.assert_called_once()

                                # Check metadata includes parallel training info
                                call_args = mock_manager.save_ensemble.call_args
                                metadata = call_args[0][1]
                                assert metadata["parallel_training"] is True
                                assert metadata["gpu_ids"] == [0, 1]

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
