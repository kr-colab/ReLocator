"""K-fold cross-validation parallel methods."""

import time
from typing import Any, Dict, List, Optional, Union

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


def _create_ray_kfold_worker(gpu_fraction: float = 1.0):
    """
    Factory function to create a Ray worker with specified GPU fraction.

    Args:
        gpu_fraction: Fraction of GPU to allocate per worker (value between 0.0 to 1.0)
                     1.0 = one full GPU per worker (default)
                     0.5 = two workers can share one GPU
                     0.25 = four workers can share one GPU
                     ...
                     0.0 = CPU only

    Returns
    -------
        Ray remote function configured with specified GPU fraction
    """

    @ray.remote(num_gpus=gpu_fraction)
    def _ray_kfold_worker(fold_idx: int, gpu_id: int, data: dict) -> Dict[str, Any]:
        """
        Ray worker function that runs a single k-fold on a specific GPU.

        Args:
            fold_idx: Fold index
            gpu_id: GPU ID to use
            data: Shared data dict (resolved from Ray object store)

        Returns
        -------
            Dictionary with predictions and metadata
        """
        _setup_worker_env(gpu_id)

        import allel
        import tensorflow as tf

        tf.get_logger().setLevel("ERROR")

        print(f"Worker processing fold {fold_idx} on GPU {gpu_id}")

        locator = _create_worker_locator(data, f"fold{fold_idx}")

        # Get fold's IndexSet
        index_set = data["fold_index_sets"][fold_idx]
        holdout_indices = index_set.test

        # Use pre-filtered allele counts if available, avoiding the
        # expensive per-worker copy of the full genotype array.
        filtered = data.get("filtered_genotypes")
        if filtered is not None:
            genotypes = None
        elif "genotypes_array" in data:
            genotypes = allel.GenotypeArray(data["genotypes_array"])
        else:
            raise ValueError("Worker received neither filtered nor raw genotypes")

        # Train with holdout
        start_time = time.time()
        history = locator.train_holdout(
            genotypes=genotypes,
            samples=data["samples"],
            holdout_indices=holdout_indices,
            filtered_genotypes=filtered,
        )
        train_time = time.time() - start_time

        # Make predictions
        predictions = locator.predict_holdout(
            verbose=False,
            return_df=True,
            save_preds_to_disk=False,
            plot_summary=False,
            plot_map=False,
        )

        # Verify sample IDs match expected holdout samples
        expected_samples = [data["samples"][i] for i in holdout_indices]
        actual_samples = predictions["sampleID"].tolist()

        if set(expected_samples) != set(actual_samples):
            print(f"WARNING: Sample mismatch in fold {fold_idx}!")
            print(f"Expected {len(expected_samples)} samples, got {len(actual_samples)}")
            print(f"First 5 expected: {expected_samples[:5]}")
            print(f"First 5 actual: {actual_samples[:5]}")

        tf.keras.backend.clear_session()

        return {
            "fold": fold_idx,
            "gpu_id": gpu_id,
            "train_time": train_time,
            "predictions": predictions.to_dict("records"),
            "holdout_indices": holdout_indices.tolist(),
            "final_loss": (
                float(history.history["loss"][-1])
                if history and "loss" in history.history
                else None
            ),
        }

    return _ray_kfold_worker


def parallel_k_fold_holdouts(  # noqa: C901
    locator,
    genotypes,
    samples,
    k: int = 10,
    gpu_ids: List[int] = [0, 1],
    gpu_fraction: float = 1.0,
    return_df: bool = True,
    save_full_pred_matrix: bool = True,
    verbose: bool = True,
    na_action: Optional[str] = None,
) -> Union[pd.DataFrame, None]:
    """
    Run true k-fold cross-validation in parallel across multiple GPUs using Ray.

    This is a parallel version of AnalysisMixin.run_k_fold_holdouts() that distributes
    folds across available GPUs.

    Args:
        locator: Locator instance (for configuration and methods)
        genotypes: GenotypeArray
        samples: List of sample IDs
        k: Number of folds (holdout sets)
        gpu_ids: List of GPU IDs to use
        gpu_fraction: Fraction of GPU to allocate per worker (default 1.0)
            - 1.0: One full GPU per worker (safest, no GPU sharing)
            - 0.5: Two workers can share one GPU
            - 0.25: Four workers can share one GPU
            - 0.0: CPU only execution
        return_df: Whether to return DataFrame with all predictions
        save_full_pred_matrix: Whether to save full prediction matrix to disk
        verbose: Whether to show training progress and intermediate output
        na_action: How to handle NA samples ('separate', 'exclude', 'fail').
            If None, uses locator.na_action

    Returns
    -------
        pandas.DataFrame or None: If return_df=True, returns DataFrame with one prediction
            per held-out sample containing columns:
            - sampleID: Sample identifier
            - x_pred: Predicted longitude
            - y_pred: Predicted latitude
            - fold: Fold number (0 to k-1)

            Note: True locations are not included. To calculate prediction errors, merge
            the returned DataFrame with your sample metadata using the sampleID column.
    """
    _ensure_ray_initialized()

    na_action, status = locator._validate_na_action(samples, na_action, "K-fold CV")

    sample_data, locs = locator._resolve_locations(samples)

    # Create NA mask
    na_mask = np.isnan(locs[:, 0])
    n_total_samples = len(locs)
    n_samples_with_coords = np.sum(~na_mask)

    if k > n_samples_with_coords:
        raise ValueError(
            f"k ({k}) must be less than or equal to number of "
            f"samples with known locations "
            f"({n_samples_with_coords})"
        )

    # Import IndexSet
    from locator.data.indexset import IndexSet

    # Create list to store IndexSets for each fold
    # Use a fixed seed based on config seed or numpy's current state
    if "seed" in locator.config and locator.config["seed"] is not None:
        kfold_seed = locator.config["seed"]
    else:
        kfold_seed = np.random.randint(0, 2**31)

    fold_index_sets = []
    for fold_idx in range(k):
        index_set = IndexSet.from_k_fold(
            n=n_total_samples,
            k=k,
            fold=fold_idx,
            seed=kfold_seed,
            na_mask=na_mask,
        )
        fold_index_sets.append(index_set)

    # Pre-calculate KDE bandwidth if needed
    bw_locs = locs[~na_mask]
    bw_calculated, bw_original = _precalculate_bandwidth(
        locator,
        bw_locs,
        f"kfold_k{k}_n{len(bw_locs)}",
        verbose,
    )

    # Pre-filter once so workers share a small array instead of the full genotypes
    filtered_genotypes = locator._filter_genotypes(genotypes)
    if verbose:
        print(
            f"Pre-filtered genotypes: {genotypes.shape[0]:,} → "
            f"{filtered_genotypes.shape[0]:,} SNPs"
        )

    data_ref = ray.put(
        {
            "filtered_genotypes": filtered_genotypes,
            "samples": samples,
            "sample_data": sample_data,
            "locs": locs,
            "config": locator.config,
            "fold_index_sets": fold_index_sets,
            "na_mask": na_mask,
        }
    )

    if verbose:
        print(
            f"Running true {k}-fold cross-validation across GPUs {gpu_ids} using Ray..."
        )

    start_time = time.time()

    # Create the Ray worker with specified GPU fraction
    _ray_kfold_worker = _create_ray_kfold_worker(gpu_fraction)

    # Submit all folds to Ray
    futures = []
    for fold_idx in range(k):
        if len(gpu_ids) == 0:
            gpu_id = -1
        else:
            gpu_id = gpu_ids[fold_idx % len(gpu_ids)]

        future = _ray_kfold_worker.remote(
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
        postfix_fn=lambda r: f"Last: Fold {r['fold']}, GPU {r['gpu_id']}",
        verbose=verbose,
    )

    total_time = time.time() - start_time

    if verbose:
        print(
            f"\nCompleted {k}-fold CV in {total_time:.1f}s "
            f"({total_time / k:.1f}s per fold)"
        )

    _restore_bandwidth(locator, bw_calculated, bw_original)

    if return_df:
        # Build predictions DataFrame
        pred_rows = []
        for result in results:
            for pred in result["predictions"]:
                pred_rows.append(
                    {
                        "sampleID": pred["sampleID"],
                        "x_pred": pred["x_pred"],
                        "y_pred": pred["y_pred"],
                        "fold": result["fold"],
                    }
                )

        all_predictions = pd.DataFrame(pred_rows)

        # Verify we have predictions for all expected samples
        expected_samples = set(samples[i] for i in range(len(samples)) if not na_mask[i])
        actual_samples = set(all_predictions["sampleID"].unique())

        if expected_samples != actual_samples:
            print("WARNING: Sample mismatch in final results!")
            print(f"Expected {len(expected_samples)} unique samples")
            print(f"Got {len(actual_samples)} unique samples")
            missing = expected_samples - actual_samples
            extra = actual_samples - expected_samples
            if missing:
                print("Missing samples: {list(missing)[:10]}...")
            if extra:
                print(f"Extra samples: {list(extra)[:10]}...")

        if save_full_pred_matrix:
            all_predictions.to_csv(
                f"{locator.config['out']}_kfold_holdouts_predlocs.csv",
                index=False,
            )

        return all_predictions

    return None


def parallel_leave_one_out(
    locator,
    genotypes,
    samples,
    gpu_ids: List[int] = [0, 1],
    gpu_fraction: float = 1.0,
    return_df: bool = True,
    save_full_pred_matrix: bool = True,
    na_action: Optional[str] = None,
) -> Union[pd.DataFrame, None]:
    """
    Perform leave-one-out cross-validation in parallel across multiple GPUs.

    This is a parallel version of AnalysisMixin.run_leave_one_out() that uses
    Ray to distribute the computation. It's a convenience wrapper around
    parallel_k_fold_holdouts with k equal to the number of samples with known locations.

    Args:
        locator: Locator instance (for configuration and methods)
        genotypes: Array of genotype data
        samples: Sample IDs corresponding to genotypes
        gpu_ids: List of GPU IDs to use
        gpu_fraction: Fraction of GPU to allocate per worker (default 1.0)
        return_df: Whether to return DataFrame with all predictions
        save_full_pred_matrix: Whether to save full prediction matrix to disk
        na_action: How to handle NA samples ('separate', 'exclude', 'fail').
            If None, uses locator.na_action

    Returns
    -------
        pandas.DataFrame or None: DataFrame with predictions for each left-out sample
    """
    # Get sample status to determine k
    status = locator.get_sample_status(samples)
    n_known = status["n_known"]

    if n_known == 0:
        raise ValueError("No samples with known coordinates for leave-one-out CV")

    print(
        f"Running leave-one-out cross-validation for "
        f"{n_known} samples across GPUs {gpu_ids}"
    )

    result = parallel_k_fold_holdouts(
        locator=locator,
        genotypes=genotypes,
        samples=samples,
        k=n_known,
        gpu_ids=gpu_ids,
        gpu_fraction=gpu_fraction,
        return_df=return_df,
        save_full_pred_matrix=False,
        verbose=False,
        na_action=na_action,
    )

    if result is not None and save_full_pred_matrix:
        result.to_csv(
            f"{locator.config['out']}_leave_one_out_predlocs.csv",
            index=False,
        )

    return result
