"""
Parallel analysis methods using Ray for multi-GPU execution.

This module provides Ray-based parallel implementations of analysis methods
from locator.analysis, enabling efficient multi-GPU utilization.

Key Features:
- Configurable GPU resource allocation via gpu_fraction parameter
- Support for multiple workers per GPU to maximize throughput
- Automatic load balancing across available GPUs
- Memory-efficient data sharing via Ray's object store

GPU Fraction Settings:
- gpu_fraction=1.0: One worker per GPU (default, safest)
- gpu_fraction=0.5: Two workers per GPU (moderate sharing)
- gpu_fraction=0.25: Four workers per GPU (moar parallelism)
- gpu_fraction=0.0: CPU only execution

Default Logging Settings:
- TensorFlow logging reduced to warnings/errors only (TF_CPP_MIN_LOG_LEVEL=2)
- Ray log deduplication enabled (RAY_DEDUP_LOGS=1)
- Ray spill logs disabled (RAY_verbose_spill_logs=0)
- Override by setting environment variables before importing
"""

import os
import time
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

# Set default logging levels for cleaner output
# Users can override these by setting env vars before importing
if "TF_CPP_MIN_LOG_LEVEL" not in os.environ:
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Only show warnings and errors
if "RAY_DEDUP_LOGS" not in os.environ:
    os.environ["RAY_DEDUP_LOGS"] = "1"  # Deduplicate Ray logs
if "RAY_verbose_spill_logs" not in os.environ:
    os.environ["RAY_verbose_spill_logs"] = "0"  # Don't show spill logs

# Ray imports
import ray

# -------------------------------------------------------------------
# Helper functions shared by workers and main functions
# -------------------------------------------------------------------


def _setup_worker_env(gpu_id):
    """Configure GPU and threading env vars for a Ray worker.

    Must be called before importing TensorFlow.
    """
    if gpu_id == -1:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ["TF_NUM_INTEROP_THREADS"] = "1"
    os.environ["TF_NUM_INTRAOP_THREADS"] = "4"
    os.environ["TF_DATA_EXPERIMENTAL_SLACK"] = "false"


def _create_worker_locator(data, suffix):
    """Create a Locator instance inside a Ray worker.

    Parameters
    ----------
    data : dict
        Shared data dict containing ``config``, ``samples``, and
        optionally ``sample_data``.
    suffix : str
        Suffix appended to the output path (e.g. ``fold0``).

    Returns
    -------
    Locator
    """
    from locator import Locator

    config = data["config"].copy()
    config["out"] = f"{config['out']}_{suffix}"
    config["disable_gpu"] = False
    config["gpu_number"] = 0
    config["keras_verbose"] = 0

    if "_sample_data_df" not in config:
        config["_sample_data_df"] = data.get("sample_data")

    locator = Locator(config)
    locator.samples = data["samples"]
    return locator


def _ensure_ray_initialized():
    """Initialize Ray if not already running."""
    if not ray.is_initialized():
        ray.init(
            log_to_driver=False,
            logging_level="ERROR",
            include_dashboard=False,
        )


def _collect_ray_results(futures, desc, postfix_fn, verbose):
    """Collect results from Ray futures with optional progress bar.

    Parameters
    ----------
    futures : list
        Ray ObjectRefs to collect.
    desc : str
        Progress bar description.
    postfix_fn : callable
        ``result -> str`` for the progress bar postfix.
    verbose : bool
        Whether to show a tqdm progress bar.

    Returns
    -------
    list
        Collected result dicts.
    """
    if verbose:
        from tqdm import tqdm

        total = len(futures)
        results = []
        remaining = list(futures)
        with tqdm(total=total, desc=desc) as pbar:
            while remaining:
                ready, remaining = ray.wait(remaining, num_returns=1)
                result = ray.get(ready[0])
                results.append(result)
                pbar.set_postfix_str(postfix_fn(result))
                pbar.update(1)
    else:
        results = ray.get(futures)
    return results


def _precalculate_bandwidth(locator, locs, cache_key, verbose):
    """Pre-calculate KDE bandwidth if sample weighting is enabled.

    Parameters
    ----------
    locator : Locator
        Locator instance whose config may request KDE weighting.
    locs : np.ndarray
        Training locations for bandwidth estimation.
    cache_key : str
        Cache key for the bandwidth optimizer.
    verbose : bool
        Whether to print progress.

    Returns
    -------
    tuple[bool, object]
        ``(calculated, original_bandwidth)``.
    """
    ws = locator.config.get("weight_samples", {})
    if not (ws.get("enabled", False) and ws.get("method") == "KD"):
        return False, None

    existing = ws.get("bandwidth")
    if existing is not None:
        return False, None

    if len(locs) <= 1:
        return False, None

    if verbose:
        print("Pre-calculating optimal KDE bandwidth...")

    from locator.sample_weights import (
        get_global_bandwidth_optimizer,
    )

    optimizer = get_global_bandwidth_optimizer()
    optimal = optimizer.get_bandwidth(
        locs,
        cache_key=cache_key,
        n_bandwidths=ws.get("n_bandwidths", 100),
        verbose=verbose,
    )

    original = existing  # None
    locator.config["weight_samples"]["bandwidth"] = optimal

    if verbose:
        print(f"Using bandwidth: {optimal:.3f}")

    return True, original


def _restore_bandwidth(locator, calculated, original):
    """Restore the original bandwidth setting after a parallel run.

    Parameters
    ----------
    locator : Locator
        Locator instance.
    calculated : bool
        Whether bandwidth was set by ``_precalculate_bandwidth``.
    original : object
        The original bandwidth value to restore.
    """
    if not calculated:
        return
    if original is None:
        locator.config.get("weight_samples", {}).pop("bandwidth", None)
    else:
        locator.config["weight_samples"]["bandwidth"] = original


# -------------------------------------------------------------------
# K-fold cross-validation
# -------------------------------------------------------------------


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

        genotypes = allel.GenotypeArray(data["genotypes_array"])
        locator = _create_worker_locator(data, f"fold{fold_idx}")

        # Get fold's IndexSet
        index_set = data["fold_index_sets"][fold_idx]
        holdout_indices = index_set.test

        # Train with holdout
        start_time = time.time()
        history = locator.train_holdout(
            genotypes=genotypes,
            samples=data["samples"],
            holdout_indices=holdout_indices,
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

    # Use instance default if na_action not specified
    if na_action is None:
        na_action = locator.na_action

    # Get sample status
    status = locator.get_sample_status(samples)

    # Report status
    if verbose:
        print(
            f"K-fold CV: {status['n_known']} samples with coordinates, "
            f"{status['n_na']} without"
        )
        if status["n_na"] > 0:
            print(f"NA handling mode: {na_action}")
            if na_action == "separate":
                print(
                    "Note: K-fold CV requires known locations; "
                    "'separate' behaves like 'exclude'"
                )

    # Apply NA action
    if na_action == "fail" and status["n_na"] > 0:
        raise ValueError(
            f"Found {status['n_na']} samples without coordinates. "
            f"Set na_action='separate' or 'exclude' to proceed."
        )

    # Get sample data and locations
    # CRITICAL: Use the same method as non-parallel version
    if hasattr(locator, "_sample_data_df"):
        sample_data, locs = locator.sort_samples(samples)
    else:
        sample_data_path = locator.config.get("sample_data")
        if not sample_data_path:
            raise ValueError("sample_data file path must be provided in config")
        sample_data, locs = locator.sort_samples(samples, sample_data_path)

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

    # Share data via Ray object store (zero-copy for numpy arrays)
    data_ref = ray.put(
        {
            "genotypes_array": genotypes.values,
            "genotypes_shape": genotypes.shape,
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


# -------------------------------------------------------------------
# Holdout replicates
# -------------------------------------------------------------------


def _create_ray_holdout_worker(gpu_fraction: float = 1.0):
    """
    Factory function to create a Ray worker for holdout analysis.

    Args:
        gpu_fraction: Fraction of GPU to allocate per worker

    Returns
    -------
        Ray remote function configured with specified GPU fraction
    """

    @ray.remote(num_gpus=gpu_fraction)
    def _ray_holdout_worker(
        rep_idx: int,
        gpu_id: int,
        data: dict,
        holdout_indices: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Ray worker function that runs a single holdout replicate on a specific GPU.

        Args:
            rep_idx: Replicate index
            gpu_id: GPU ID to use
            data: Shared data dict (resolved from Ray object store)
            holdout_indices: Indices to hold out for this replicate

        Returns
        -------
            Dictionary with predictions and metadata
        """
        _setup_worker_env(gpu_id)

        import allel
        import tensorflow as tf

        tf.get_logger().setLevel("ERROR")

        print(f"Worker processing replicate {rep_idx} on GPU {gpu_id}")

        genotypes = allel.GenotypeArray(data["genotypes_array"])
        locator = _create_worker_locator(data, f"rep{rep_idx}")

        # Train with holdout
        start_time = time.time()
        history = locator.train_holdout(
            genotypes=genotypes,
            samples=data["samples"],
            holdout_indices=holdout_indices,
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

        tf.keras.backend.clear_session()

        return {
            "rep": rep_idx,
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

    return _ray_holdout_worker


def parallel_holdouts(  # noqa: C901
    locator,
    genotypes,
    samples,
    k: int = 10,
    n_reps: int = 10,
    holdout_indices: Optional[List[List[int]]] = None,
    holdout_sample_ids: Optional[Union[List[str], List[List[str]]]] = None,
    gpu_ids: List[int] = [0, 1],
    gpu_fraction: float = 1.0,
    return_df: bool = True,
    save_full_pred_matrix: bool = True,
    verbose: bool = True,
    na_action: Optional[str] = None,
) -> Union[pd.DataFrame, None]:
    """
    Run multiple holdout replicates in parallel across multiple GPUs using Ray.

    This is a parallel version of AnalysisMixin.run_holdouts() that distributes
    replicates across available GPUs.

    Args:
        locator: Locator instance (for configuration and methods)
        genotypes: GenotypeArray
        samples: List of sample IDs
        k: Number of samples to hold out in each replicate
        n_reps: Number of holdout replicates to run
        holdout_indices: Optional list of lists, each containing indices to hold out
        holdout_sample_ids: Optional list of sample IDs to hold out. If provided,
            these specific samples will be held out (overrides k and holdout_indices).
            Can be a single list (used for all replicates) or list of lists
            (different samples per replicate).
        gpu_ids: List of GPU IDs to use
        gpu_fraction: Fraction of GPU to allocate per worker (default 1.0)
        return_df: Whether to return DataFrame with all predictions
        save_full_pred_matrix: Whether to save full prediction matrix to disk
        verbose: Whether to show training progress and intermediate output
        na_action: How to handle NA samples ('separate', 'exclude', 'fail').
            If None, uses locator.na_action

    Returns
    -------
        pandas.DataFrame or None: If return_df=True, returns DataFrame with predictions
            for each holdout replicate containing columns:
            - sampleID: Sample identifier
            - x_rep0, y_rep0: Predictions from replicate 0
            - x_rep1, y_rep1: Predictions from replicate 1
            - ... and so on for all replicates

            Note: True locations are not included. Merge with sample metadata to calculate errors.
    """
    _ensure_ray_initialized()

    # Use instance default if na_action not specified
    if na_action is None:
        na_action = locator.na_action

    # Get sample status
    status = locator.get_sample_status(samples)

    # Report status
    if verbose:
        print(
            f"Holdout analysis: {status['n_known']} samples "
            f"with coordinates, {status['n_na']} without"
        )
        if status["n_na"] > 0:
            print(f"NA handling mode: {na_action}")
            if na_action == "separate":
                print(
                    "Note: Holdout analysis requires known "
                    "locations; 'separate' behaves like 'exclude'"
                )

    # Apply NA action
    if na_action == "fail" and status["n_na"] > 0:
        raise ValueError(
            f"Found {status['n_na']} samples without coordinates. "
            f"Set na_action='separate' or 'exclude' to proceed."
        )

    # Get sample data and locations
    if hasattr(locator, "_sample_data_df"):
        sample_data, locs = locator.sort_samples(samples)
    else:
        sample_data_path = locator.config.get("sample_data")
        if not sample_data_path:
            raise ValueError("sample_data file path must be provided in config")
        sample_data, locs = locator.sort_samples(samples, sample_data_path)

    # Get indices of samples with known locations (optimized)
    known_mask = ~np.isnan(locs[:, 0])
    known_idx = np.where(known_mask)[0]

    if k >= len(known_idx):
        raise ValueError(
            f"k ({k}) must be less than number of samples with "
            f"known locations ({len(known_idx)})"
        )

    # Pre-calculate KDE bandwidth if needed
    bw_locs = locs[known_idx]
    bw_calculated, bw_original = _precalculate_bandwidth(
        locator,
        bw_locs,
        f"holdouts_k{k}_n{len(bw_locs)}",
        verbose,
    )

    # Handle holdout_sample_ids if provided
    if holdout_sample_ids is not None:
        if hasattr(samples, "tolist"):
            samples_list = samples.tolist()
        else:
            samples_list = list(samples)

        if isinstance(holdout_sample_ids[0], str):
            try:
                holdout_indices = [
                    [samples_list.index(sid) for sid in holdout_sample_ids]
                ]
            except ValueError:
                missing = [sid for sid in holdout_sample_ids if sid not in samples_list]
                raise ValueError(f"Sample IDs not found in samples list: {missing}")
            holdout_indices = holdout_indices * n_reps
            k = len(holdout_sample_ids)
        else:
            holdout_indices = []
            for rep_ids in holdout_sample_ids:
                try:
                    rep_indices = [samples_list.index(sid) for sid in rep_ids]
                except ValueError:
                    missing = [sid for sid in rep_ids if sid not in samples_list]
                    raise ValueError(f"Sample IDs not found in samples list: {missing}")
                holdout_indices.append(rep_indices)
            n_reps = len(holdout_indices)

    # Generate holdout indices for all replicates
    all_holdout_indices = []
    for rep in range(n_reps):
        if holdout_indices is not None and rep < len(holdout_indices):
            rep_holdout_idx = holdout_indices[rep]
        else:
            rep_holdout_idx = np.random.choice(known_idx, k, replace=False)
        all_holdout_indices.append(rep_holdout_idx)

    # Share data via Ray object store
    data_ref = ray.put(
        {
            "genotypes_array": genotypes.values,
            "genotypes_shape": genotypes.shape,
            "samples": samples,
            "sample_data": sample_data,
            "locs": locs,
            "config": locator.config,
            "known_idx": known_idx,
        }
    )

    if verbose:
        print(f"Running {n_reps} holdout replicates across GPUs {gpu_ids} using Ray...")

    start_time = time.time()

    _ray_holdout_worker = _create_ray_holdout_worker(gpu_fraction)

    # Submit all replicates to Ray
    futures = []
    for rep_idx in range(n_reps):
        if len(gpu_ids) == 0:
            gpu_id = -1
        else:
            gpu_id = gpu_ids[rep_idx % len(gpu_ids)]

        future = _ray_holdout_worker.remote(
            rep_idx=rep_idx,
            gpu_id=gpu_id,
            data=data_ref,
            holdout_indices=all_holdout_indices[rep_idx],
        )
        futures.append(future)
        if verbose:
            device_str = "CPU" if gpu_id == -1 else f"GPU {gpu_id}"
            print(f"Submitted replicate {rep_idx} to {device_str}")

    # Wait for all replicates to complete
    results = _collect_ray_results(
        futures,
        desc="Replicates completed",
        postfix_fn=lambda r: f"Last: Rep {r['rep']}, GPU {r['gpu_id']}",
        verbose=verbose,
    )

    total_time = time.time() - start_time

    if verbose:
        print(
            f"\nCompleted {n_reps} replicates in "
            f"{total_time:.1f}s "
            f"({total_time / n_reps:.1f}s per replicate)"
        )

    _restore_bandwidth(locator, bw_calculated, bw_original)

    if return_df:
        pred_dfs = []

        for result in results:
            rep_idx = result["rep"]
            predictions = pd.DataFrame(result["predictions"])

            holdout_preds = predictions[["x_pred", "y_pred"]].copy()
            holdout_preds.columns = [
                f"x_rep{rep_idx}",
                f"y_rep{rep_idx}",
            ]
            holdout_preds["sampleID"] = predictions["sampleID"]
            pred_dfs.append(holdout_preds)

        all_predictions = pred_dfs[0]
        for df in pred_dfs[1:]:
            all_predictions = pd.merge(all_predictions, df, on="sampleID", how="outer")

        if save_full_pred_matrix:
            all_predictions.to_csv(
                f"{locator.config['out']}_holdouts_predlocs.csv",
                index=False,
            )

        return all_predictions

    return None


# -------------------------------------------------------------------
# Windowed holdout analysis
# -------------------------------------------------------------------


def _create_ray_windows_worker(gpu_fraction: float = 1.0):
    """
    Factory function to create a Ray worker for windowed holdout analysis.

    Args:
        gpu_fraction: Fraction of GPU to allocate per worker

    Returns
    -------
        Ray remote function configured with specified GPU fraction
    """

    @ray.remote(num_gpus=gpu_fraction)
    def _ray_windows_worker(
        window_idx: int,
        window_start: int,
        window_stop: int,
        gpu_id: int,
        data: dict,
    ) -> Dict[str, Any]:
        """
        Ray worker function that runs holdout analysis for a single genomic window.

        Args:
            window_idx: Window index
            window_start: Start position of window
            window_stop: Stop position of window
            gpu_id: GPU ID to use
            data: Shared data dict (resolved from Ray object store)

        Returns
        -------
            Dictionary with predictions and metadata
        """
        _setup_worker_env(gpu_id)

        import allel
        import tensorflow as tf

        from locator.data.filters import normalize_locs  # noqa: F401
        from locator.data.indexset import IndexSet  # noqa: F401

        tf.get_logger().setLevel("ERROR")

        print(
            f"Worker processing window {window_idx} "
            f"({window_start}-{window_stop}) on GPU {gpu_id}"
        )

        genotypes = allel.GenotypeArray(data["genotypes_array"])

        # Get window specification
        windows = data.get("windows", [])
        if window_idx < len(windows):
            window_spec = windows[window_idx]
            snp_indices = np.where(window_spec["indices"])[0]
            window_label = window_spec["label"]
            window_chromosome = window_spec.get("chromosome")
        else:
            positions = data["positions"]
            snp_mask = (positions >= window_start) & (positions < window_stop)
            snp_indices = np.where(snp_mask)[0]
            window_label = f"pos{window_start}"
            window_chromosome = None

        if len(snp_indices) == 0:
            print(f"No SNPs in window {window_start}-{window_stop}")
            return {
                "window_idx": window_idx,
                "window_start": window_start,
                "window_stop": window_stop,
                "window_label": window_label,
                "window_chromosome": window_chromosome,
                "predictions": None,
                "n_snps": 0,
            }

        locator = _create_worker_locator(data, f"win{window_idx}")
        locator.genotypes = genotypes
        locator.index_set = data["index_set"]

        # Set normalization parameters
        locator.meanlong = data["meanlong"]
        locator.sdlong = data["sdlong"]
        locator.meanlat = data["meanlat"]
        locator.sdlat = data["sdlat"]
        locator.unnormedlocs = data["unnormedlocs"]

        # Train on window
        start_time = time.time()
        locator.train_window(
            genotypes=genotypes,
            samples=data["samples"],
            window_snp_indices=snp_indices,
            index_set=data["index_set"],
            normalized_locs=data["normalized_locs"],
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

        tf.keras.backend.clear_session()

        return {
            "window_idx": window_idx,
            "window_start": window_start,
            "window_stop": window_stop,
            "window_label": window_label,
            "window_chromosome": window_chromosome,
            "gpu_id": gpu_id,
            "train_time": train_time,
            "predictions": (
                predictions.to_dict("records") if predictions is not None else None
            ),
            "n_snps": len(snp_indices),
        }

    return _ray_windows_worker


def parallel_windows_holdouts(  # noqa: C901
    locator,
    genotypes,
    samples,
    k: int = 10,
    window_start: int = 0,
    window_size: int = int(5e5),
    window_stop: Optional[int] = None,
    respect_chromosomes: bool = True,
    holdout_indices: Optional[List[int]] = None,
    holdout_sample_ids: Optional[List[str]] = None,
    gpu_ids: List[int] = [0, 1],
    gpu_fraction: float = 1.0,
    return_df: bool = True,
    save_full_pred_matrix: bool = True,
    verbose: bool = True,
    na_action: Optional[str] = None,
) -> Union[pd.DataFrame, None]:
    """
    Run windowed analysis on holdout samples in parallel across multiple GPUs using Ray.

    This is a parallel version of AnalysisMixin.run_windows_holdouts() that distributes
    windows across available GPUs.

    Args:
        locator: Locator instance (for configuration and methods)
        genotypes: GenotypeArray
        samples: List of sample IDs
        k: Number of samples to hold out
        window_start: Start position for windows
        window_size: Size of windows in base pairs
        window_stop: Stop position for windows (if None, uses max position)
        respect_chromosomes: Whether to respect chromosome boundaries when creating
            windows (default: True). If True, windows will not span chromosome
            boundaries. Requires chromosome information from VCF/Zarr input.
        holdout_indices: Optional specific indices to hold out
        holdout_sample_ids: Optional list of sample IDs to hold out. If provided,
            these specific samples will be held out (overrides k and holdout_indices).
        gpu_ids: List of GPU IDs to use
        gpu_fraction: Fraction of GPU to allocate per worker (default 1.0)
        return_df: Whether to return DataFrame with all predictions
        save_full_pred_matrix: Whether to save full prediction matrix to disk
        verbose: Whether to show training progress and intermediate output
        na_action: How to handle NA samples ('separate', 'exclude', 'fail').
            If None, uses locator.na_action

    Returns
    -------
        pandas.DataFrame or None: If return_df=True, returns DataFrame with predictions
            for each window containing columns:
            - sampleID: Sample identifier
            - x_pos0, y_pos0: Predictions from window starting at position 0
            - x_pos500000, y_pos500000: Predictions from window starting at position 500000
            - ... and so on for all windows

            Note: True locations are not included. Merge with sample metadata to calculate errors.

    Warning:
        When respect_chromosomes=False, window analysis treats all SNP positions as
        continuous along a single coordinate axis. If your data contains multiple
        chromosomes, windows may span across chromosome boundaries. Use
        respect_chromosomes=True (default) for biologically meaningful windows.
    """
    _ensure_ray_initialized()

    # Use instance default if na_action not specified
    if na_action is None:
        na_action = locator.na_action

    # Store samples and genotypes
    locator.samples = samples
    locator.genotypes = genotypes

    # Get sample status and create NA mask
    status = locator.get_sample_status(samples)
    na_mask = None
    if status["n_na"] > 0:
        if isinstance(samples, pd.DataFrame):
            na_mask = samples["x"].isna() | samples["y"].isna()
        else:
            if hasattr(locator, "_sample_data_df"):
                sample_data_df = locator._sample_data_df
            else:
                sample_data_path = locator.config.get("sample_data")
                if sample_data_path:
                    sample_data_df = pd.read_csv(sample_data_path, sep="\t")
                else:
                    raise ValueError("No sample data available")
            merged = pd.DataFrame({"sampleID": samples})
            merged = merged.merge(sample_data_df, on="sampleID", how="left")
            na_mask = merged["x"].isna() | merged["y"].isna()
        na_mask = na_mask.values

    # Report status
    if verbose:
        print(
            f"Windows holdout analysis: "
            f"{status['n_known']} samples with coordinates, "
            f"{status['n_na']} without"
        )
        if status["n_na"] > 0:
            print(f"NA handling mode: {na_action}")
            if na_action == "separate":
                print(
                    "Note: Holdout analysis requires known "
                    "locations; 'separate' behaves like 'exclude'"
                )

    # Apply NA action
    if na_action == "fail" and status["n_na"] > 0:
        raise ValueError(
            f"Found {status['n_na']} samples without coordinates. "
            f"Set na_action='separate' or 'exclude' to proceed."
        )

    # Get positions
    if not hasattr(locator, "positions") or locator.positions is None:
        if hasattr(locator, "_genotype_df"):
            locator.positions = np.array(locator._genotype_df.columns, dtype=int)
        elif locator.config.get("zarr"):
            import zarr

            callset = zarr.open_group(locator.config["zarr"], mode="r")
            locator.positions = callset["variants/POS"][:]
        elif locator.config.get("vcf"):
            if verbose:
                print("Loading SNP positions from VCF...")
            import allel

            vcf = allel.read_vcf(
                locator.config["vcf"],
                fields=["POS", "CHROM"],
            )
            if vcf is not None and "variants/POS" in vcf:
                locator.positions = vcf["variants/POS"]
                if "variants/CHROM" in vcf:
                    locator.chromosomes = vcf["variants/CHROM"]
                if verbose:
                    print(f"Loaded {len(locator.positions)} SNP positions")
            else:
                raise ValueError(
                    f"Could not load positions from VCF: {locator.config['vcf']}"
                )
        else:
            raise ValueError(
                "SNP positions required for windowed analysis. "
                "Use VCF, zarr input or genotype DataFrame "
                "with position-labeled columns."
            )

    # Handle holdout_sample_ids if provided
    if holdout_sample_ids is not None:
        if hasattr(samples, "tolist"):
            samples_list = samples.tolist()
        else:
            samples_list = list(samples)

        try:
            holdout_indices = [samples_list.index(sid) for sid in holdout_sample_ids]
        except ValueError:
            missing = [sid for sid in holdout_sample_ids if sid not in samples_list]
            raise ValueError(f"Sample IDs not found in samples list: {missing}")
        k = len(holdout_indices)

    # Create IndexSet for holdout splitting
    from locator.data.indexset import IndexSet

    n_samples = len(samples)

    if holdout_indices is not None:
        holdout_idx = np.array(holdout_indices)
        train_mask = np.ones(n_samples, dtype=bool)
        train_mask[holdout_idx] = False
        train_idx = np.where(train_mask)[0]

        if na_mask is not None and (na_action == "exclude" or na_action == "separate"):
            valid_mask = ~na_mask
            holdout_idx = holdout_idx[valid_mask[holdout_idx]]
            train_idx = train_idx[valid_mask[train_idx]]

        index_set = IndexSet(
            indices={"train": train_idx, "test": holdout_idx},
            total_samples=n_samples,
            na_mask=na_mask,
        )
    else:
        index_set = IndexSet.random_split(
            n=n_samples,
            splits={
                "train": 1.0 - k / n_samples,
                "test": k / n_samples,
            },
            seed=locator.config.get("seed", 42),
            na_mask=na_mask,
            na_action=(na_action if na_action != "separate" else "exclude"),
        )

    if window_stop is None:
        window_stop = max(locator.positions)

    # Generate windows
    from locator.data.windows import generate_genomic_windows

    chromosomes = getattr(locator, "chromosomes", None)
    windows = generate_genomic_windows(
        positions=locator.positions,
        chromosomes=chromosomes,
        window_start=window_start,
        window_size=window_size,
        window_stop=window_stop,
        respect_chromosomes=respect_chromosomes,
        min_snps_per_window=locator.config.get("min_snps_per_window", 1),
        verbose=verbose,
    )

    # Get sample data and locations (single call, used for both
    # bandwidth calculation and location normalization)
    if hasattr(locator, "_sample_data_df"):
        sample_data, locs = locator.sort_samples(samples)
    else:
        sample_data_path = locator.config.get("sample_data")
        if not sample_data_path:
            raise ValueError("sample_data file path must be provided in config")
        sample_data, locs = locator.sort_samples(samples, sample_data_path)

    # Pre-calculate KDE bandwidth if needed
    bw_train_mask = np.ones(len(samples), dtype=bool)
    bw_train_mask[index_set.test] = False
    bw_train_mask &= ~np.isnan(locs[:, 0])
    bw_locs = locs[bw_train_mask]
    bw_calculated, bw_original = _precalculate_bandwidth(
        locator,
        bw_locs,
        f"windows_holdouts_n{len(bw_locs)}",
        verbose,
    )

    # Normalize locations once
    from locator.data.filters import normalize_locs

    (
        meanlong,
        sdlong,
        meanlat,
        sdlat,
        unnormedlocs,
        normalized_locs,
    ) = normalize_locs(locs)

    # Share data via Ray object store
    data_ref = ray.put(
        {
            "genotypes_array": genotypes.values,
            "genotypes_shape": genotypes.shape,
            "samples": samples,
            "sample_data": sample_data,
            "config": locator.config,
            "positions": locator.positions,
            "windows": windows,
            "index_set": index_set,
            "meanlong": meanlong,
            "sdlong": sdlong,
            "meanlat": meanlat,
            "sdlat": sdlat,
            "unnormedlocs": unnormedlocs,
            "normalized_locs": normalized_locs,
        }
    )

    if verbose:
        print(
            f"Running windowed analysis for {len(windows)} "
            f"windows across GPUs {gpu_ids} using Ray..."
        )

    start_time = time.time()

    _ray_windows_worker = _create_ray_windows_worker(gpu_fraction)

    # Submit all windows to Ray
    futures = []
    for window_idx, window in enumerate(windows):
        if len(gpu_ids) == 0:
            gpu_id = -1
        else:
            gpu_id = gpu_ids[window_idx % len(gpu_ids)]

        future = _ray_windows_worker.remote(
            window_idx=window_idx,
            window_start=window["start"],
            window_stop=window["stop"],
            gpu_id=gpu_id,
            data=data_ref,
        )
        futures.append(future)
        if verbose and window_idx < 10:
            chrom_str = f" (chr{window['chromosome']})" if window["chromosome"] else ""
            device_str = "CPU" if gpu_id == -1 else f"GPU {gpu_id}"
            print(
                f"Submitted window {window_idx}{chrom_str} "
                f"({window['start']}-{window['stop']}) "
                f"to {device_str}"
            )

    if verbose and len(windows) > 10:
        print(f"... and {len(windows) - 10} more windows")

    # Wait for all windows to complete
    results = _collect_ray_results(
        futures,
        desc="Windows completed",
        postfix_fn=lambda r: (
            f"Last: Window {r['window_idx']}"
            + (f" (chr{r['window_chromosome']})" if r["window_chromosome"] else "")
            + f", GPU {r['gpu_id']}"
        ),
        verbose=verbose,
    )

    total_time = time.time() - start_time

    if verbose:
        print(
            f"\nCompleted {len(windows)} windows in "
            f"{total_time:.1f}s "
            f"({total_time / len(windows):.1f}s per window)"
        )

        # Show GPU utilization summary
        gpu_counts = {}
        for result in results:
            gid = result["gpu_id"]
            gpu_counts[gid] = gpu_counts.get(gid, 0) + 1

        print("\nGPU utilization:")
        for gid in sorted(gpu_counts.keys()):
            pct = gpu_counts[gid] / len(windows) * 100
            print(f"  GPU {gid}: {gpu_counts[gid]} windows ({pct:.1f}%)")

    _restore_bandwidth(locator, bw_calculated, bw_original)

    if return_df:
        pred_dfs = []

        for result in results:
            if result["predictions"] is not None:
                window_label = result.get(
                    "window_label",
                    f"pos{result['window_start']}",
                )
                predictions = pd.DataFrame(result["predictions"])

                window_preds = predictions[["x_pred", "y_pred"]].copy()
                window_preds.columns = [
                    f"x_{window_label}",
                    f"y_{window_label}",
                ]
                window_preds["sampleID"] = predictions["sampleID"]
                pred_dfs.append(window_preds)

        if not pred_dfs:
            print("Warning: No windows contained SNPs. No predictions generated.")
            return None

        all_predictions = pred_dfs[0]
        for df in pred_dfs[1:]:
            all_predictions = pd.merge(all_predictions, df, on="sampleID")

        if save_full_pred_matrix:
            all_predictions.to_csv(
                f"{locator.config['out']}_windows_holdouts_predlocs.csv",
                index=False,
            )

        return all_predictions

    return None


# -------------------------------------------------------------------
# Ensemble training
# -------------------------------------------------------------------


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

        import allel
        import tensorflow as tf

        tf.get_logger().setLevel("ERROR")

        print(f"Worker training ensemble fold {fold_idx} on GPU {gpu_id}")

        # Reconstruct GenotypeArray (for consistency check)
        _ = allel.GenotypeArray(  # noqa: F841
            data["genotypes_array"]
        )
        filtered_genotypes = data["filtered_genotypes_array"]

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

    # Get locations once
    _, locs = locator.sort_samples(samples)

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

    # Share data via Ray object store
    data_ref = ray.put(
        {
            "genotypes_array": genotypes.values,
            "filtered_genotypes_array": filtered_genotypes,
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


# Additional parallel methods that could be implemented:
# - parallel_jacknife_holdouts() - for run_jacknife_holdouts()
