"""Ensemble training parallel methods."""

import os
import time
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import ray

from ._helpers import (
    _collect_ray_results,
    _create_worker_locator,
    _ensure_ray_initialized,
    _precalculate_bandwidth,
    _restore_bandwidth,
    _setup_worker_env,
)


def _create_ray_ensemble_worker(gpu_fraction: float = 1.0):
    """
    Factory function to create a Ray worker for ensemble training.

    Args:
        gpu_fraction: Fraction of GPU to allocate per worker (value between 0.0 to 1.0)

    Returns
    -------
        Ray remote function configured with specified GPU fraction
    """

    @ray.remote(num_gpus=gpu_fraction)
    def _ray_ensemble_worker(fold_idx: int, gpu_id: int, data: dict) -> Dict[str, Any]:
        """
        Ray worker function that trains a single ensemble fold on a specific GPU.

        Args:
            fold_idx: Fold index
            gpu_id: GPU ID to use
            data: Shared data dict (resolved from Ray object store)

        Returns
        -------
            Dictionary with model information and metadata
        """
        _setup_worker_env(gpu_id)

        import tensorflow as tf

        tf.get_logger().setLevel("ERROR")

        print(f"Worker training ensemble fold {fold_idx} on GPU {gpu_id}")

        filtered_genotypes = data["filtered_genotypes"]

        locator = _create_worker_locator(data, f"fold{fold_idx}")

        # Train single fold using existing method
        start_time = time.time()
        model_info = locator._train_single_fold(
            fold_idx=fold_idx,
            index_set=data["fold_info"]["index_sets"][fold_idx],
            filtered_genotypes=filtered_genotypes,
            samples=data["samples"],
            locs=data["locs"],
            augment_config=data.get("augment_config"),
            save_fold_models=data["save_fold_models"],
            patience_multiplier=data["patience_multiplier"],
            verbose=False,
        )
        train_time = time.time() - start_time

        # Add weights file path if saving
        if data["save_fold_models"]:
            model_info["weights_file"] = f"{locator.config['out']}.weights.h5"
        else:
            model_info["weights_file"] = None

        result = {
            "fold": fold_idx,
            "gpu_id": gpu_id,
            "train_time": train_time,
            "model_info": {
                "fold": model_info["fold"],
                "weights_file": model_info["weights_file"],
                "norm_params": model_info["norm_params"],
                "train_indices": (model_info["train_indices"].tolist()),
                "val_indices": (model_info["val_indices"].tolist()),
            },
            "history": {
                "loss": model_info["history"].history.get("loss", []),
                "val_loss": model_info["history"].history.get("val_loss", []),
            },
            "final_loss": float(model_info["history"].history["loss"][-1]),
            "final_val_loss": float(model_info["history"].history["val_loss"][-1]),
        }

        tf.keras.backend.clear_session()

        return result

    return _ray_ensemble_worker


def parallel_train_ensemble(  # noqa: C901
    locator,
    genotypes,
    samples,
    k: int = 5,
    gpu_ids: List[int] = [0, 1],
    gpu_fraction: float = 1.0,
    training_set_indices: Optional[List[int]] = None,
    na_action: Optional[str] = None,
    augment_data: bool = False,
    flip_rate: float = 0.05,
    save_fold_models: bool = True,
    use_model_manager: bool = True,
    use_mixed_precision: Optional[bool] = None,
    patience_multiplier: float = 1.0,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Train an ensemble of k models in parallel across multiple GPUs using Ray.

    This is a parallel version of EnsembleMixin.train_ensemble() that distributes
    fold training across available GPUs.

    Args:
        locator: Locator instance (for configuration and methods)
        genotypes: GenotypeArray containing genetic data
        samples: Array of sample IDs
        k: Number of folds/models in ensemble (default: 5)
        gpu_ids: List of GPU IDs to use (default: [0, 1])
        gpu_fraction: Fraction of GPU to allocate per worker (default 1.0)
            - 1.0: One full GPU per worker (safest, no GPU sharing)
            - 0.5: Two workers can share one GPU
            - 0.25: Four workers can share one GPU
            - 0.0: CPU only execution
        training_set_indices: Optional array of indices to restrict training
        na_action: How to handle NA samples ('separate', 'exclude', 'fail')
        augment_data: Whether to apply data augmentation (default: False)
        flip_rate: Rate for genotype flipping augmentation (default: 0.05)
        save_fold_models: Whether to save individual fold models (default: True)
        use_model_manager: Whether to use model manager for saving (default: True)
        use_mixed_precision: Whether to use mixed precision training (default: None, auto-detect)
        patience_multiplier: Multiply patience for ensemble training (default: 1.0)
        verbose: Whether to show training progress (default: True)

    Returns
    -------
        dict: Dictionary containing:
            - 'histories': List of training histories for each fold
            - 'models': List of trained model configurations
            - 'normalization_params': Averaged normalization parameters
            - 'fold_info': Information about fold splits
    """
    _ensure_ray_initialized()

    # Setup GPU optimizations for ensemble training in main process
    if verbose:
        mixed_precision_enabled = locator.setup_ensemble_gpu_optimization(
            use_mixed_precision
        )
        if mixed_precision_enabled:
            print("Mixed precision training enabled for ensemble")
    else:
        locator.setup_ensemble_gpu_optimization(use_mixed_precision)

    # Store samples for later use
    locator.samples = samples

    # Create folds using IndexSet
    fold_info = locator.create_ensemble_folds(
        genotypes, samples, k, training_set_indices, na_action
    )

    # Filter SNPs once before training
    filtered_genotypes = locator._filter_genotypes(genotypes)

    _, locs = locator._resolve_locations(samples)

    # Configure augmentation if requested
    augment_config = None
    if augment_data:
        augment_config = {
            "enabled": True,
            "flip_rate": flip_rate,
        }
        locator.config["augmentation"] = augment_config

    # Get sample data for serialization
    sample_data = None
    if hasattr(locator, "_sample_data_df"):
        sample_data = locator._sample_data_df

    # Pre-calculate KDE bandwidth if needed
    na_mask = np.isnan(locs[:, 0]) | np.isnan(locs[:, 1])
    bw_locs = locs[~na_mask]
    bw_calculated, bw_original = _precalculate_bandwidth(
        locator,
        bw_locs,
        f"ensemble_k{k}_n{len(bw_locs)}",
        verbose,
    )

    # Share only filtered array via Ray object store
    data_ref = ray.put(
        {
            "filtered_genotypes": filtered_genotypes,
            "samples": samples,
            "sample_data": sample_data,
            "locs": locs,
            "config": locator.config,
            "fold_info": fold_info,
            "augment_config": augment_config,
            "save_fold_models": save_fold_models,
            "patience_multiplier": patience_multiplier,
        }
    )

    if verbose:
        print(f"Training {k}-fold ensemble across GPUs {gpu_ids} using Ray...")

    start_time = time.time()

    _ray_ensemble_worker = _create_ray_ensemble_worker(gpu_fraction)

    # Submit all folds to Ray
    futures = []
    for fold_idx in range(k):
        if len(gpu_ids) == 0:
            gpu_id = -1
        else:
            gpu_id = gpu_ids[fold_idx % len(gpu_ids)]

        future = _ray_ensemble_worker.remote(
            fold_idx=fold_idx, gpu_id=gpu_id, data=data_ref
        )
        futures.append(future)
        if verbose:
            device_str = "CPU" if gpu_id == -1 else f"GPU {gpu_id}"
            print(f"Submitted fold {fold_idx} to {device_str}")

    # Wait for all folds to complete
    results = _collect_ray_results(
        futures,
        desc="Folds completed",
        postfix_fn=lambda r: (
            f"Last: Fold {r['fold']}, "
            f"GPU {r['gpu_id']}, "
            f"Final loss: {r['final_loss']:.4f}"
        ),
        verbose=verbose,
    )

    total_time = time.time() - start_time

    if verbose:
        print(
            f"\nCompleted ensemble training in "
            f"{total_time:.1f}s "
            f"({total_time / k:.1f}s per fold)"
        )

        # Show speedup vs sequential
        if len(gpu_ids) > 0:
            num_gpus = len(set(gpu_ids))
            estimated_speedup = k / num_gpus
            print(
                f"Estimated speedup: {estimated_speedup:.1f}x "
                f"(using {num_gpus} "
                f"GPU{'s' if num_gpus > 1 else ''})"
            )
        else:
            print("CPU mode - no GPU speedup available")

    _restore_bandwidth(locator, bw_calculated, bw_original)

    # Aggregate results (sort by fold index)
    results_sorted = sorted(results, key=lambda x: x["fold"])

    # Store results on locator instance
    locator._ensemble_genotypes = genotypes
    locator._ensemble_fold_info = fold_info
    locator._ensemble_models = []
    locator._ensemble_histories = []
    locator._ensemble_norm_params = []

    for result in results_sorted:
        model_info = result["model_info"]
        model_info["model"] = None
        model_info["train_indices"] = np.array(model_info["train_indices"])
        model_info["val_indices"] = np.array(model_info["val_indices"])

        class HistoryStub:
            def __init__(self, history_dict):
                self.history = history_dict

        model_info["history"] = HistoryStub(result["history"])

        locator._ensemble_models.append(model_info)
        locator._ensemble_histories.append(model_info["history"])
        locator._ensemble_norm_params.append(model_info["norm_params"])

    # Calculate averaged normalization parameters
    avg_norm_params = locator._average_normalization_params(
        locator._ensemble_norm_params
    )

    # Store averaged parameters on instance
    locator.meanlong = avg_norm_params["meanlong"]
    locator.sdlong = avg_norm_params["sdlong"]
    locator.meanlat = avg_norm_params["meanlat"]
    locator.sdlat = avg_norm_params["sdlat"]

    # Save ensemble using model manager if requested
    if use_model_manager and save_fold_models:
        from locator.ensemble_model_manager import (
            EnsembleModelManager,
        )

        model_manager = EnsembleModelManager(f"{locator.config['out']}_ensemble")

        serializable_config = {
            k_: v for k_, v in locator.config.items() if not isinstance(v, pd.DataFrame)
        }

        ensemble_metadata = {
            "k_folds": k,
            "na_action": na_action or locator.na_action,
            "augment_data": augment_data,
            "config": serializable_config,
            "parallel_training": True,
            "gpu_ids": gpu_ids,
        }

        models_loaded = False
        if verbose:
            print("Checking for saved model weights...")

        for i, m_info in enumerate(locator._ensemble_models):
            if m_info["weights_file"] and os.path.exists(m_info["weights_file"]):
                if not models_loaded and verbose:
                    print("Loading models for ensemble manager...")
                models_loaded = True
                model = locator._create_model(input_shape=filtered_genotypes.shape[0])
                model.load_weights(m_info["weights_file"])
                m_info["model"] = model
            else:
                if verbose and m_info["weights_file"]:
                    print(
                        "Warning: Expected weights file "
                        f"not found: {m_info['weights_file']}"
                    )

        if models_loaded:
            model_manager.save_ensemble(locator._ensemble_models, ensemble_metadata)
            if verbose:
                print(f"Saved ensemble to {model_manager.ensemble_dir}")
        else:
            if verbose:
                print(
                    "Models were saved individually by workers,"
                    " skipping ensemble manager."
                )

    return {
        "histories": locator._ensemble_histories,
        "models": locator._ensemble_models,
        "normalization_params": avg_norm_params,
        "fold_info": fold_info,
        "training_time": total_time,
        "parallel": True,
    }
