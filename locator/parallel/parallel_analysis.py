"""
Parallel analysis methods using Ray for multi-GPU execution.

This module provides Ray-based parallel implementations of analysis methods
from locator.analysis, enabling efficient multi-GPU utilization.

Key Features:
- Configurable GPU resource allocation via gpu_fraction parameter
- Support for multiple workers per GPU to maximize throughput
- Automatic load balancing across available GPUs
- Memory-efficient data serialization

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
import pickle
import tempfile
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

    Returns:
        Ray remote function configured with specified GPU fraction
    """

    @ray.remote(num_gpus=gpu_fraction)
    def _ray_kfold_worker(fold_idx: int, gpu_id: int, data_file: str) -> Dict[str, Any]:
        """
        Ray worker function that runs a single k-fold on a specific GPU.

        Args:
            fold_idx: Fold index
            gpu_id: GPU ID to use
            data_file: Path to pickled data file

        Returns:
            Dictionary with predictions and metadata
        """
        # Set GPU before importing TensorFlow
        if gpu_id == -1:
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable GPU
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        # Set TensorFlow threading environment variables BEFORE import
        # This ensures the tf.data pipeline doesn't fork excessively
        os.environ["TF_NUM_INTEROP_THREADS"] = "1"
        os.environ["TF_NUM_INTRAOP_THREADS"] = "4"
        os.environ["TF_DATA_EXPERIMENTAL_SLACK"] = "false"

        # Import inside worker to ensure proper GPU setup
        import allel
        import tensorflow as tf

        from locator import Locator

        # Suppress TF warnings
        tf.get_logger().setLevel("ERROR")

        print(f"Worker processing fold {fold_idx} on GPU {gpu_id}")

        # Load data from pickle file
        with open(data_file, "rb") as f:
            data = pickle.load(f)

        # Reconstruct GenotypeArray
        gt_array = data["genotypes_array"]
        # shape = data["genotypes_shape"]  # noqa: F841
        # FIXED: Simply reconstruct from the raw values
        genotypes = allel.GenotypeArray(gt_array)

        # Get fold's IndexSet
        index_set = data["fold_index_sets"][fold_idx]
        holdout_indices = index_set.test

        # Create Locator instance
        locator_config = data["config"].copy()
        locator_config["out"] = f"{locator_config['out']}_fold{fold_idx}"
        locator_config["disable_gpu"] = False
        locator_config["gpu_number"] = 0  # Use first visible GPU
        locator_config["keras_verbose"] = 0  # Suppress keras output

        # CRITICAL FIX: Store the sample data DataFrame in the config
        # This ensures sort_samples works correctly
        if "_sample_data_df" not in locator_config:
            locator_config["_sample_data_df"] = data["sample_data"]

        locator = Locator(locator_config)  # Pass as dictionary

        # This must match the exact order used when creating the IndexSets
        locator.samples = data["samples"]

        # Train with holdout
        start_time = time.time()
        history = locator.train_holdout(
            genotypes=genotypes,
            samples=data["samples"],  # Pass the same samples list
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

        # Clear keras session
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

    Returns:
        pandas.DataFrame or None: If return_df=True, returns DataFrame with one prediction
            per held-out sample containing columns:
            - sampleID: Sample identifier
            - x_pred: Predicted longitude
            - y_pred: Predicted latitude
            - fold: Fold number (0 to k-1)

            Note: True locations are not included. To calculate prediction errors, merge
            the returned DataFrame with your sample metadata using the sampleID column.
    """
    # Initialize Ray if not already initialized
    if not ray.is_initialized():
        ray.init(
            log_to_driver=False,  # Don't log worker output to driver
            logging_level="ERROR",  # Only show errors
            include_dashboard=False,  # Don't start Ray dashboard
        )

    # Use instance default if na_action not specified
    if na_action is None:
        na_action = locator.na_action

    # Get sample status
    status = locator.get_sample_status(samples)

    # Report status
    if verbose:
        print(
            f"K-fold CV: {status['n_known']} samples with coordinates, {status['n_na']} without"
        )
        if status["n_na"] > 0:
            print(f"NA handling mode: {na_action}")
            if na_action == "separate":
                print(
                    "Note: K-fold CV requires known locations; 'separate' behaves like 'exclude'"
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

    # Update samples array to match filtered data (after exclusions)
    # sort_samples may have excluded samples, so we need to use the filtered sample IDs
    filtered_samples = sample_data["sampleID"].values

    # If samples were excluded, we need to filter genotypes to match
    if len(filtered_samples) < len(samples):
        # Find indices of samples that remain after exclusion
        samples_list = samples.tolist() if hasattr(samples, "tolist") else list(samples)
        keep_indices = [
            i
            for i, s in enumerate(samples_list)
            if str(s) in set(str(fs) for fs in filtered_samples)
        ]
        genotypes = genotypes[:, keep_indices]

    samples = np.array(filtered_samples)

    # Create NA mask
    na_mask = np.isnan(locs[:, 0])
    n_total_samples = len(locs)
    n_samples_with_coords = np.sum(~na_mask)

    if k > n_samples_with_coords:
        raise ValueError(
            f"k ({k}) must be less than or equal to number of samples with known locations ({n_samples_with_coords})"
        )

    # Import IndexSet
    from locator.data.indexset import IndexSet

    # Create list to store IndexSets for each fold
    # Use a fixed seed based on config seed or numpy's current state
    if "seed" in locator.config and locator.config["seed"] is not None:
        kfold_seed = locator.config["seed"]
    else:
        # Generate a seed from current numpy state to ensure consistency
        kfold_seed = np.random.randint(0, 2**31)

    fold_index_sets = []
    for fold_idx in range(k):
        index_set = IndexSet.from_k_fold(
            n=n_total_samples,
            k=k,
            fold=fold_idx,
            seed=kfold_seed,  # Use consistent seed for all folds
            na_mask=na_mask,
        )
        fold_index_sets.append(index_set)

    # Pre-calculate KDE bandwidth if needed
    bandwidth_calculated = False
    original_bandwidth = None

    if (
        locator.config.get("weight_samples", {}).get("enabled", False)
        and locator.config.get("weight_samples", {}).get("method") == "KD"
    ):

        existing_bandwidth = locator.config.get("weight_samples", {}).get("bandwidth")

        if existing_bandwidth is None:
            # Get all samples with coordinates for bandwidth calculation
            coords_mask = ~na_mask
            all_train_locs = locs[coords_mask]

            if len(all_train_locs) > 1:
                if verbose:
                    print("Pre-calculating optimal KDE bandwidth for k-fold CV...")

                from locator.sample_weights import get_global_bandwidth_optimizer

                optimizer = get_global_bandwidth_optimizer()

                optimal_bandwidth = optimizer.get_bandwidth(
                    all_train_locs,
                    cache_key=f"kfold_k{k}_n{len(all_train_locs)}",
                    n_bandwidths=locator.config.get("weight_samples", {}).get(
                        "n_bandwidths", 100
                    ),
                    verbose=verbose,
                )

                # Store original value
                original_bandwidth = existing_bandwidth
                # Set in config
                locator.config["weight_samples"]["bandwidth"] = optimal_bandwidth
                bandwidth_calculated = True

                if verbose:
                    print(f"Using bandwidth: {optimal_bandwidth:.3f}")

    # Save data to temporary file
    with tempfile.NamedTemporaryFile(mode="wb", delete=False, suffix=".pkl") as f:
        data = {
            "genotypes_array": genotypes.values,  # FIXED: Save raw values, not to_n_alt()
            "genotypes_shape": genotypes.shape,
            "samples": samples,  # CRITICAL: Pass the original samples list
            "sample_data": sample_data,  # Pass the sorted sample data
            "locs": locs,
            "config": locator.config,
            "fold_index_sets": fold_index_sets,
            "na_mask": na_mask,
        }
        pickle.dump(data, f)
        data_file = f.name

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
        # Handle empty gpu_ids (CPU only mode)
        if len(gpu_ids) == 0:
            gpu_id = -1  # Use CPU
        else:
            gpu_id = gpu_ids[fold_idx % len(gpu_ids)]

        future = _ray_kfold_worker.remote(
            fold_idx=fold_idx, gpu_id=gpu_id, data_file=data_file
        )
        futures.append(future)
        if verbose:
            device_str = "CPU" if gpu_id == -1 else f"GPU {gpu_id}"
            print(f"Submitted fold {fold_idx} to {device_str}")

    # Wait for all folds to complete with progress bar
    if verbose:
        print("\nProcessing folds across GPUs...")
        from tqdm import tqdm

        # Process results with progress bar
        results = []
        with tqdm(total=k, desc="Folds completed") as pbar:
            while futures:
                # Wait for any task to complete
                ready, futures = ray.wait(futures, num_returns=1)
                result = ray.get(ready[0])
                results.append(result)

                # Update progress bar
                pbar.set_postfix_str(
                    f"Last: Fold {result['fold']}, GPU {result['gpu_id']}"
                )
                pbar.update(1)
    else:
        # No progress bar if not verbose
        results = ray.get(futures)

    total_time = time.time() - start_time

    # Clean up
    os.unlink(data_file)

    if verbose:
        print(
            f"\nCompleted {k}-fold CV in {total_time:.1f}s ({total_time/k:.1f}s per fold)"
        )

    # Restore original bandwidth setting if we changed it
    if bandwidth_calculated:
        if original_bandwidth is None:
            # Remove the key if it wasn't there originally
            locator.config.get("weight_samples", {}).pop("bandwidth", None)
        else:
            locator.config["weight_samples"]["bandwidth"] = original_bandwidth

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
                f"{locator.config['out']}_kfold_holdouts_predlocs.csv", index=False
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

    Returns:
        pandas.DataFrame or None: DataFrame with predictions for each left-out sample
    """
    # Get sample status to determine k
    status = locator.get_sample_status(samples)
    n_known = status["n_known"]

    if n_known == 0:
        raise ValueError("No samples with known coordinates for leave-one-out CV")

    print(
        f"Running leave-one-out cross-validation for {n_known} samples across GPUs {gpu_ids}"
    )

    # Run k-fold with k equal to number of known samples
    # This will create folds with exactly 1 sample each
    result = parallel_k_fold_holdouts(
        locator=locator,
        genotypes=genotypes,
        samples=samples,
        k=n_known,
        gpu_ids=gpu_ids,
        gpu_fraction=gpu_fraction,
        return_df=return_df,
        save_full_pred_matrix=False,  # We'll save with our own name
        verbose=False,  # We already printed our message
        na_action=na_action,
    )

    # Save with leave-one-out specific filename if requested
    if result is not None and save_full_pred_matrix:
        result.to_csv(f"{locator.config['out']}_leave_one_out_predlocs.csv", index=False)

    return result


def _create_ray_holdout_worker(gpu_fraction: float = 1.0):
    """
    Factory function to create a Ray worker for holdout analysis.

    Args:
        gpu_fraction: Fraction of GPU to allocate per worker

    Returns:
        Ray remote function configured with specified GPU fraction
    """

    @ray.remote(num_gpus=gpu_fraction)
    def _ray_holdout_worker(
        rep_idx: int, gpu_id: int, data_file: str, holdout_indices: np.ndarray
    ) -> Dict[str, Any]:
        """
        Ray worker function that runs a single holdout replicate on a specific GPU.

        Args:
            rep_idx: Replicate index
            gpu_id: GPU ID to use
            data_file: Path to pickled data file
            holdout_indices: Indices to hold out for this replicate

        Returns:
            Dictionary with predictions and metadata
        """
        # Set GPU before importing TensorFlow
        if gpu_id == -1:
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable GPU
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        # Set TensorFlow threading environment variables BEFORE import
        # This ensures the tf.data pipeline doesn't fork excessively
        os.environ["TF_NUM_INTEROP_THREADS"] = "1"
        os.environ["TF_NUM_INTRAOP_THREADS"] = "4"
        os.environ["TF_DATA_EXPERIMENTAL_SLACK"] = "false"

        # Import inside worker to ensure proper GPU setup
        import allel
        import tensorflow as tf

        from locator import Locator

        # Suppress TF warnings
        tf.get_logger().setLevel("ERROR")

        print(f"Worker processing replicate {rep_idx} on GPU {gpu_id}")

        # Load data from pickle file
        with open(data_file, "rb") as f:
            data = pickle.load(f)

        # Reconstruct GenotypeArray
        gt_array = data["genotypes_array"]
        genotypes = allel.GenotypeArray(gt_array)

        # Create Locator instance
        locator_config = data["config"].copy()
        locator_config["out"] = f"{locator_config['out']}_rep{rep_idx}"
        locator_config["disable_gpu"] = False
        locator_config["gpu_number"] = 0  # Use first visible GPU
        locator_config["keras_verbose"] = 0  # Suppress keras output

        # Store the sample data DataFrame in the config
        if "_sample_data_df" not in locator_config:
            locator_config["_sample_data_df"] = data["sample_data"]

        locator = Locator(locator_config)

        # Ensure samples are set correctly
        locator.samples = data["samples"]

        # Train with holdout
        start_time = time.time()
        history = locator.train_holdout(
            genotypes=genotypes, samples=data["samples"], holdout_indices=holdout_indices
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

        # Clear keras session
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

    Returns:
        pandas.DataFrame or None: If return_df=True, returns DataFrame with predictions
            for each holdout replicate containing columns:
            - sampleID: Sample identifier
            - x_rep0, y_rep0: Predictions from replicate 0
            - x_rep1, y_rep1: Predictions from replicate 1
            - ... and so on for all replicates

            Note: True locations are not included. Merge with sample metadata to calculate errors.
    """
    # Initialize Ray if not already initialized
    if not ray.is_initialized():
        ray.init(
            log_to_driver=False,  # Don't log worker output to driver
            logging_level="ERROR",  # Only show errors
            include_dashboard=False,  # Don't start Ray dashboard
        )

    # Use instance default if na_action not specified
    if na_action is None:
        na_action = locator.na_action

    # Get sample status
    status = locator.get_sample_status(samples)

    # Report status
    if verbose:
        print(
            f"Holdout analysis: {status['n_known']} samples with coordinates, {status['n_na']} without"
        )
        if status["n_na"] > 0:
            print(f"NA handling mode: {na_action}")
            if na_action == "separate":
                print(
                    "Note: Holdout analysis requires known locations; 'separate' behaves like 'exclude'"
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

    # Update samples array to match filtered data (after exclusions)
    # sort_samples may have excluded samples, so we need to use the filtered sample IDs
    filtered_samples = sample_data["sampleID"].values

    # If samples were excluded, we need to filter genotypes to match
    if len(filtered_samples) < len(samples):
        # Find indices of samples that remain after exclusion
        samples_list = samples.tolist() if hasattr(samples, "tolist") else list(samples)
        keep_indices = [
            i
            for i, s in enumerate(samples_list)
            if str(s) in set(str(fs) for fs in filtered_samples)
        ]
        genotypes = genotypes[:, keep_indices]

    samples = np.array(filtered_samples)

    # Get indices of samples with known locations (optimized)
    # Use boolean indexing instead of argwhere for efficiency
    known_mask = ~np.isnan(locs[:, 0])
    known_idx = np.where(known_mask)[0]

    if k >= len(known_idx):
        raise ValueError(
            f"k ({k}) must be less than number of samples with known locations ({len(known_idx)})"
        )

    # Pre-calculate KDE bandwidth if needed
    bandwidth_calculated = False
    original_bandwidth = None

    if (
        locator.config.get("weight_samples", {}).get("enabled", False)
        and locator.config.get("weight_samples", {}).get("method") == "KD"
    ):

        existing_bandwidth = locator.config.get("weight_samples", {}).get("bandwidth")

        if existing_bandwidth is None:
            # Get all samples with coordinates for bandwidth calculation
            all_train_locs = locs[known_idx]

            if len(all_train_locs) > 1:
                if verbose:
                    print(
                        "Pre-calculating optimal KDE bandwidth for holdout analysis..."
                    )

                from locator.sample_weights import get_global_bandwidth_optimizer

                optimizer = get_global_bandwidth_optimizer()

                optimal_bandwidth = optimizer.get_bandwidth(
                    all_train_locs,
                    cache_key=f"holdouts_k{k}_n{len(all_train_locs)}",
                    n_bandwidths=locator.config.get("weight_samples", {}).get(
                        "n_bandwidths", 100
                    ),
                    verbose=verbose,
                )

                # Store original value
                original_bandwidth = existing_bandwidth
                # Set in config
                locator.config["weight_samples"]["bandwidth"] = optimal_bandwidth
                bandwidth_calculated = True

                if verbose:
                    print(f"Using bandwidth: {optimal_bandwidth:.3f}")

    # Handle holdout_sample_ids if provided
    if holdout_sample_ids is not None:
        # Convert samples to list if it's a numpy array
        if hasattr(samples, "tolist"):
            samples_list = samples.tolist()
        else:
            samples_list = list(samples)

        # Convert sample IDs to indices
        if isinstance(holdout_sample_ids[0], str):
            # Single list of sample IDs for all replicates
            try:
                holdout_indices = [
                    [samples_list.index(sid) for sid in holdout_sample_ids]
                ]
            except ValueError:
                missing = [sid for sid in holdout_sample_ids if sid not in samples_list]
                raise ValueError(f"Sample IDs not found in samples list: {missing}")
            # Replicate for all n_reps if needed
            holdout_indices = holdout_indices * n_reps
            k = len(holdout_sample_ids)  # Update k to match
        else:
            # List of lists - different sample IDs per replicate
            holdout_indices = []
            for rep_ids in holdout_sample_ids:
                try:
                    rep_indices = [samples_list.index(sid) for sid in rep_ids]
                except ValueError:
                    missing = [sid for sid in rep_ids if sid not in samples_list]
                    raise ValueError(f"Sample IDs not found in samples list: {missing}")
                holdout_indices.append(rep_indices)
            n_reps = len(holdout_indices)  # Update n_reps to match

    # Generate holdout indices for all replicates
    all_holdout_indices = []
    for rep in range(n_reps):
        if holdout_indices is not None and rep < len(holdout_indices):
            rep_holdout_idx = holdout_indices[rep]
        else:
            # Random selection
            rep_holdout_idx = np.random.choice(known_idx, k, replace=False)
        all_holdout_indices.append(rep_holdout_idx)

    # Save data to temporary file
    with tempfile.NamedTemporaryFile(mode="wb", delete=False, suffix=".pkl") as f:
        data = {
            "genotypes_array": genotypes.values,
            "genotypes_shape": genotypes.shape,
            "samples": samples,
            "sample_data": sample_data,
            "locs": locs,
            "config": locator.config,
            "known_idx": known_idx,
        }
        pickle.dump(data, f)
        data_file = f.name

    if verbose:
        print(f"Running {n_reps} holdout replicates across GPUs {gpu_ids} using Ray...")

    start_time = time.time()

    # Create the Ray worker with specified GPU fraction
    _ray_holdout_worker = _create_ray_holdout_worker(gpu_fraction)

    # Submit all replicates to Ray
    futures = []
    for rep_idx in range(n_reps):
        # Handle empty gpu_ids (CPU only mode)
        if len(gpu_ids) == 0:
            gpu_id = -1  # Use CPU
        else:
            gpu_id = gpu_ids[rep_idx % len(gpu_ids)]

        future = _ray_holdout_worker.remote(
            rep_idx=rep_idx,
            gpu_id=gpu_id,
            data_file=data_file,
            holdout_indices=all_holdout_indices[rep_idx],
        )
        futures.append(future)
        if verbose:
            device_str = "CPU" if gpu_id == -1 else f"GPU {gpu_id}"
            print(f"Submitted replicate {rep_idx} to {device_str}")

    # Wait for all replicates to complete with progress bar
    if verbose:
        print("\nProcessing replicates across GPUs...")
        from tqdm import tqdm

        # Process results with progress bar
        results = []
        with tqdm(total=n_reps, desc="Replicates completed") as pbar:
            while futures:
                # Wait for any task to complete
                ready, futures = ray.wait(futures, num_returns=1)
                result = ray.get(ready[0])
                results.append(result)

                # Update progress bar
                pbar.set_postfix_str(
                    f"Last: Rep {result['rep']}, GPU {result['gpu_id']}"
                )
                pbar.update(1)
    else:
        # No progress bar if not verbose
        results = ray.get(futures)

    total_time = time.time() - start_time

    # Clean up
    os.unlink(data_file)

    if verbose:
        print(
            f"\nCompleted {n_reps} replicates in {total_time:.1f}s ({total_time/n_reps:.1f}s per replicate)"
        )

    # Restore original bandwidth setting if we changed it
    if bandwidth_calculated:
        if original_bandwidth is None:
            # Remove the key if it wasn't there originally
            locator.config.get("weight_samples", {}).pop("bandwidth", None)
        else:
            locator.config["weight_samples"]["bandwidth"] = original_bandwidth

    if return_df:
        # Build predictions DataFrame in the same format as sequential version
        pred_dfs = []

        for result in results:
            rep_idx = result["rep"]
            predictions = pd.DataFrame(result["predictions"])

            # Rename columns to include replicate number
            holdout_preds = predictions[["x_pred", "y_pred"]].copy()
            holdout_preds.columns = [f"x_rep{rep_idx}", f"y_rep{rep_idx}"]
            holdout_preds["sampleID"] = predictions["sampleID"]
            pred_dfs.append(holdout_preds)

        # Merge all predictions
        all_predictions = pred_dfs[0]
        for df in pred_dfs[1:]:
            all_predictions = pd.merge(all_predictions, df, on="sampleID", how="outer")

        if save_full_pred_matrix:
            all_predictions.to_csv(
                f"{locator.config['out']}_holdouts_predlocs.csv", index=False
            )

        return all_predictions

    return None


def _create_ray_windows_worker(gpu_fraction: float = 1.0):
    """
    Factory function to create a Ray worker for windowed holdout analysis.

    Args:
        gpu_fraction: Fraction of GPU to allocate per worker

    Returns:
        Ray remote function configured with specified GPU fraction
    """

    @ray.remote(num_gpus=gpu_fraction)
    def _ray_windows_worker(
        window_idx: int, window_start: int, window_stop: int, gpu_id: int, data_file: str
    ) -> Dict[str, Any]:
        """
        Ray worker function that runs holdout analysis for a single genomic window.

        Args:
            window_idx: Window index
            window_start: Start position of window
            window_stop: Stop position of window
            gpu_id: GPU ID to use
            data_file: Path to pickled data file

        Returns:
            Dictionary with predictions and metadata
        """
        # Set GPU before importing TensorFlow
        if gpu_id == -1:
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable GPU
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        # Set TensorFlow threading environment variables BEFORE import
        # This ensures the tf.data pipeline doesn't fork excessively
        os.environ["TF_NUM_INTEROP_THREADS"] = "1"
        os.environ["TF_NUM_INTRAOP_THREADS"] = "4"
        os.environ["TF_DATA_EXPERIMENTAL_SLACK"] = "false"

        # Import inside worker to ensure proper GPU setup
        import allel
        import tensorflow as tf

        from locator import Locator
        from locator.data.filters import normalize_locs  # noqa: F401
        from locator.data.indexset import IndexSet  # noqa: F401

        # Suppress TF warnings
        tf.get_logger().setLevel("ERROR")

        print(
            f"Worker processing window {window_idx} ({window_start}-{window_stop}) on GPU {gpu_id}"
        )

        # Load data from pickle file
        with open(data_file, "rb") as f:
            data = pickle.load(f)

        # Reconstruct GenotypeArray
        gt_array = data["genotypes_array"]
        genotypes = allel.GenotypeArray(gt_array)

        # Get window specification
        windows = data.get("windows", [])
        if window_idx < len(windows):
            # Use pre-computed window indices
            window_spec = windows[window_idx]
            snp_indices = np.where(window_spec["indices"])[0]
            window_label = window_spec["label"]
            window_chromosome = window_spec.get("chromosome")
        else:
            # Fallback to position-based calculation
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

        # Create Locator instance
        locator_config = data["config"].copy()
        locator_config["out"] = f"{locator_config['out']}_win{window_idx}"
        locator_config["disable_gpu"] = False
        locator_config["gpu_number"] = 0  # Use first visible GPU
        locator_config["keras_verbose"] = 0  # Suppress keras output

        # Store the sample data DataFrame in the config
        if "_sample_data_df" not in locator_config:
            locator_config["_sample_data_df"] = data["sample_data"]

        locator = Locator(locator_config)

        # Ensure samples are set correctly
        locator.samples = data["samples"]
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

        # Clear keras session
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

    Returns:
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
    # Initialize Ray if not already initialized
    if not ray.is_initialized():
        ray.init(
            log_to_driver=False,  # Don't log worker output to driver
            logging_level="ERROR",  # Only show errors
            include_dashboard=False,  # Don't start Ray dashboard
        )

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
        # Create boolean mask for NA samples
        if isinstance(samples, pd.DataFrame):
            na_mask = samples["x"].isna() | samples["y"].isna()
        else:
            # Use stored sample data or load from config
            if hasattr(locator, "_sample_data_df"):
                sample_data = locator._sample_data_df
            else:
                sample_data_path = locator.config.get("sample_data")
                if sample_data_path:
                    sample_data = pd.read_csv(sample_data_path, sep="\t")
                else:
                    raise ValueError("No sample data available")

            merged = pd.DataFrame({"sampleID": samples})
            merged = merged.merge(sample_data, on="sampleID", how="left")
            na_mask = merged["x"].isna() | merged["y"].isna()
        na_mask = na_mask.values

    # Report status
    if verbose:
        print(
            f"Windows holdout analysis: {status['n_known']} samples with coordinates, {status['n_na']} without"
        )
        if status["n_na"] > 0:
            print(f"NA handling mode: {na_action}")
            if na_action == "separate":
                print(
                    "Note: Holdout analysis requires known locations; 'separate' behaves like 'exclude'"
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
            # Re-read VCF to get positions and chromosomes
            if verbose:
                print("Loading SNP positions from VCF...")
            import allel

            vcf = allel.read_vcf(locator.config["vcf"], fields=["POS", "CHROM"])
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
                "SNP positions required for windowed analysis. Use VCF, zarr input or "
                "genotype DataFrame with position-labeled columns."
            )

    # Handle holdout_sample_ids if provided
    if holdout_sample_ids is not None:
        # Convert sample IDs to indices
        # Handle both list and numpy array cases
        if hasattr(samples, "tolist"):
            samples_list = samples.tolist()
        else:
            samples_list = list(samples)

        try:
            holdout_indices = [samples_list.index(sid) for sid in holdout_sample_ids]
        except ValueError:
            missing = [sid for sid in holdout_sample_ids if sid not in samples_list]
            raise ValueError(f"Sample IDs not found in samples list: {missing}")
        k = len(holdout_indices)  # Update k to match

    # Create IndexSet for holdout splitting
    from locator.data.indexset import IndexSet

    n_samples = len(samples)

    if holdout_indices is not None:
        # Use provided holdout indices
        holdout_idx = np.array(holdout_indices)
        # More efficient than setdiff1d for this use case
        train_mask = np.ones(n_samples, dtype=bool)
        train_mask[holdout_idx] = False
        train_idx = np.where(train_mask)[0]

        # Apply NA mask if needed
        if na_mask is not None and (na_action == "exclude" or na_action == "separate"):
            # Only keep samples with known coordinates
            valid_mask = ~na_mask
            holdout_idx = holdout_idx[valid_mask[holdout_idx]]
            train_idx = train_idx[valid_mask[train_idx]]

        index_set = IndexSet(
            indices={"train": train_idx, "test": holdout_idx},
            total_samples=n_samples,
            na_mask=na_mask,
        )
    else:
        # Random holdout selection using IndexSet
        index_set = IndexSet.random_split(
            n=n_samples,
            splits={"train": 1.0 - k / n_samples, "test": k / n_samples},
            seed=locator.config.get("seed", 42),
            na_mask=na_mask,
            na_action=na_action if na_action != "separate" else "exclude",
        )

    if window_stop is None:
        window_stop = max(locator.positions)

    # Generate windows using the new helper function
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

    # Pre-calculate KDE bandwidth if needed
    bandwidth_calculated = False
    original_bandwidth = None

    if (
        locator.config.get("weight_samples", {}).get("enabled", False)
        and locator.config.get("weight_samples", {}).get("method") == "KD"
    ):

        existing_bandwidth = locator.config.get("weight_samples", {}).get("bandwidth")

        if existing_bandwidth is None:
            # Get sample data and locations
            if hasattr(locator, "_sample_data_df"):
                sample_data, locs = locator.sort_samples(samples)
            else:
                sample_data_path = locator.config.get("sample_data")
                if not sample_data_path:
                    raise ValueError("sample_data file path must be provided in config")
                sample_data, locs = locator.sort_samples(samples, sample_data_path)

            # Update samples array to match filtered data (after exclusions)
            # sort_samples may have excluded samples, so we need to use the filtered sample IDs
            filtered_samples = sample_data["sampleID"].values

            # If samples were excluded, we need to filter genotypes to match
            if len(filtered_samples) < len(samples):
                # Find indices of samples that remain after exclusion
                samples_list = (
                    samples.tolist() if hasattr(samples, "tolist") else list(samples)
                )
                keep_indices = [
                    i
                    for i, s in enumerate(samples_list)
                    if str(s) in set(str(fs) for fs in filtered_samples)
                ]
                genotypes = genotypes[:, keep_indices]

            samples = np.array(filtered_samples)

            # Get training locations (exclude holdout samples) - optimized
            # Avoid creating intermediate arrays
            train_mask = np.ones(len(samples), dtype=bool)
            train_mask[index_set.test] = False
            # Combine with location mask in-place
            train_mask &= ~np.isnan(locs[:, 0])
            train_locs = locs[train_mask]

            if len(train_locs) > 1:
                if verbose:
                    print(
                        "Pre-calculating optimal KDE bandwidth for windows holdout analysis..."
                    )

                from locator.sample_weights import get_global_bandwidth_optimizer

                optimizer = get_global_bandwidth_optimizer()

                optimal_bandwidth = optimizer.get_bandwidth(
                    train_locs,
                    cache_key=f"windows_holdouts_n{len(train_locs)}",
                    n_bandwidths=locator.config.get("weight_samples", {}).get(
                        "n_bandwidths", 100
                    ),
                    verbose=verbose,
                )

                # Store original value
                original_bandwidth = existing_bandwidth
                # Set in config
                locator.config["weight_samples"]["bandwidth"] = optimal_bandwidth
                bandwidth_calculated = True

                if verbose:
                    print(f"Using bandwidth: {optimal_bandwidth:.3f}")

    # Pre-normalize locations for efficiency
    if hasattr(locator, "_sample_data_df"):
        sample_data, locs = locator.sort_samples(samples)
    else:
        sample_data_path = locator.config.get("sample_data")
        if not sample_data_path:
            raise ValueError("sample_data file path must be provided in config")
        sample_data, locs = locator.sort_samples(samples, sample_data_path)

    # Update samples array to match filtered data (after exclusions)
    # sort_samples may have excluded samples, so we need to use the filtered sample IDs
    filtered_samples = sample_data["sampleID"].values

    # If samples were excluded, we need to filter genotypes to match
    if len(filtered_samples) < len(samples):
        # Find indices of samples that remain after exclusion
        samples_list = samples.tolist() if hasattr(samples, "tolist") else list(samples)
        keep_indices = [
            i
            for i, s in enumerate(samples_list)
            if str(s) in set(str(fs) for fs in filtered_samples)
        ]
        genotypes = genotypes[:, keep_indices]

    samples = np.array(filtered_samples)

    # Normalize locations once
    from locator.data.filters import normalize_locs

    meanlong, sdlong, meanlat, sdlat, unnormedlocs, normalized_locs = normalize_locs(
        locs
    )

    # Save data to temporary file
    with tempfile.NamedTemporaryFile(mode="wb", delete=False, suffix=".pkl") as f:
        data = {
            "genotypes_array": genotypes.values,
            "genotypes_shape": genotypes.shape,
            "samples": samples,
            "sample_data": sample_data,
            "config": locator.config,
            "positions": locator.positions,
            "windows": windows,  # Include window specifications
            "index_set": index_set,
            "meanlong": meanlong,
            "sdlong": sdlong,
            "meanlat": meanlat,
            "sdlat": sdlat,
            "unnormedlocs": unnormedlocs,
            "normalized_locs": normalized_locs,
        }
        pickle.dump(data, f)
        data_file = f.name

    if verbose:
        print(
            f"Running windowed analysis for {len(windows)} windows across GPUs {gpu_ids} using Ray..."
        )

    start_time = time.time()

    # Create the Ray worker with specified GPU fraction
    _ray_windows_worker = _create_ray_windows_worker(gpu_fraction)

    # Submit all windows to Ray
    futures = []
    for window_idx, window in enumerate(windows):
        # Handle empty gpu_ids (CPU only mode)
        if len(gpu_ids) == 0:
            gpu_id = -1  # Use CPU
        else:
            gpu_id = gpu_ids[window_idx % len(gpu_ids)]

        future = _ray_windows_worker.remote(
            window_idx=window_idx,
            window_start=window["start"],
            window_stop=window["stop"],
            gpu_id=gpu_id,
            data_file=data_file,
        )
        futures.append(future)
        if verbose and window_idx < 10:  # Only print first few for brevity
            chrom_str = f" (chr{window['chromosome']})" if window["chromosome"] else ""
            device_str = "CPU" if gpu_id == -1 else f"GPU {gpu_id}"
            print(
                f"Submitted window {window_idx}{chrom_str} ({window['start']}-{window['stop']}) to {device_str}"
            )

    if verbose and len(windows) > 10:
        print(f"... and {len(windows)-10} more windows")

    # Wait for all windows to complete with progress bar
    if verbose:
        print("\nProcessing windows across GPUs...")
        from tqdm import tqdm

        # Process results with progress bar
        results = []
        completed = 0
        with tqdm(total=len(futures), desc="Windows completed") as pbar:
            while futures:
                # Wait for any task to complete
                ready, futures = ray.wait(futures, num_returns=1)
                result = ray.get(ready[0])
                results.append(result)

                # Update progress bar with window info
                window_info = f"Window {result['window_idx']}"
                if result["window_chromosome"]:
                    window_info += f" (chr{result['window_chromosome']})"
                pbar.set_postfix_str(f"Last: {window_info}, GPU {result['gpu_id']}")
                pbar.update(1)
                completed += 1
    else:
        # No progress bar if not verbose
        results = ray.get(futures)

    total_time = time.time() - start_time

    # Clean up
    os.unlink(data_file)

    if verbose:
        print(
            f"\nCompleted {len(windows)} windows in {total_time:.1f}s ({total_time/len(windows):.1f}s per window)"
        )

        # Show GPU utilization summary
        gpu_counts = {}
        for result in results:
            gpu_id = result["gpu_id"]
            gpu_counts[gpu_id] = gpu_counts.get(gpu_id, 0) + 1

        print("\nGPU utilization:")
        for gpu_id in sorted(gpu_counts.keys()):
            print(
                f"  GPU {gpu_id}: {gpu_counts[gpu_id]} windows ({gpu_counts[gpu_id]/len(windows)*100:.1f}%)"
            )

    # Restore original bandwidth setting if we changed it
    if bandwidth_calculated:
        if original_bandwidth is None:
            # Remove the key if it wasn't there originally
            locator.config.get("weight_samples", {}).pop("bandwidth", None)
        else:
            locator.config["weight_samples"]["bandwidth"] = original_bandwidth

    if return_df:
        # Build predictions DataFrame in the same format as sequential version
        pred_dfs = []

        for result in results:
            if result["predictions"] is not None:
                window_label = result.get("window_label", f"pos{result['window_start']}")
                predictions = pd.DataFrame(result["predictions"])

                # Rename columns to include window label
                window_preds = predictions[["x_pred", "y_pred"]].copy()
                window_preds.columns = [f"x_{window_label}", f"y_{window_label}"]
                window_preds["sampleID"] = predictions["sampleID"]
                pred_dfs.append(window_preds)

        # Check if any windows had predictions
        if not pred_dfs:
            print("Warning: No windows contained SNPs. No predictions generated.")
            return None

        # Merge all predictions
        all_predictions = pred_dfs[0]
        for df in pred_dfs[1:]:
            all_predictions = pd.merge(all_predictions, df, on="sampleID")

        if save_full_pred_matrix:
            all_predictions.to_csv(
                f"{locator.config['out']}_windows_holdouts_predlocs.csv", index=False
            )

        return all_predictions

    return None


def _create_ray_ensemble_worker(gpu_fraction: float = 1.0):
    """
    Factory function to create a Ray worker for ensemble training.

    Args:
        gpu_fraction: Fraction of GPU to allocate per worker (value between 0.0 to 1.0)

    Returns:
        Ray remote function configured with specified GPU fraction
    """

    @ray.remote(num_gpus=gpu_fraction)
    def _ray_ensemble_worker(
        fold_idx: int, gpu_id: int, data_file: str
    ) -> Dict[str, Any]:
        """
        Ray worker function that trains a single ensemble fold on a specific GPU.

        Args:
            fold_idx: Fold index
            gpu_id: GPU ID to use
            data_file: Path to pickled data file

        Returns:
            Dictionary with model information and metadata
        """
        # Set GPU before importing TensorFlow
        if gpu_id == -1:
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable GPU
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        # Set TensorFlow threading environment variables BEFORE import
        # This ensures the tf.data pipeline doesn't fork excessively
        os.environ["TF_NUM_INTEROP_THREADS"] = "1"
        os.environ["TF_NUM_INTRAOP_THREADS"] = "4"
        os.environ["TF_DATA_EXPERIMENTAL_SLACK"] = "false"

        # Import inside worker to ensure proper GPU setup
        import allel
        import tensorflow as tf

        from locator import Locator

        # Suppress TF warnings
        tf.get_logger().setLevel("ERROR")

        print(f"Worker training ensemble fold {fold_idx} on GPU {gpu_id}")

        # Load data from pickle file
        with open(data_file, "rb") as f:
            data = pickle.load(f)

        # Reconstruct GenotypeArrays (genotypes not used but reconstructed for consistency)
        _ = allel.GenotypeArray(data["genotypes_array"])  # noqa: F841
        filtered_genotypes = data["filtered_genotypes_array"]  # Already a numpy array

        # Create Locator instance
        locator_config = data["config"].copy()
        locator_config["out"] = f"{locator_config['out']}_fold{fold_idx}"
        locator_config["disable_gpu"] = False
        locator_config["gpu_number"] = 0  # Use first visible GPU
        locator_config["keras_verbose"] = 0  # Suppress keras output

        # Store the sample data DataFrame in the config
        if "_sample_data_df" not in locator_config:
            locator_config["_sample_data_df"] = data.get("sample_data")

        locator = Locator(locator_config)

        # Set samples to ensure consistency
        locator.samples = data["samples"]

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
            verbose=False,  # Suppress individual fold output
        )
        train_time = time.time() - start_time

        # Add weights file path if saving
        if data["save_fold_models"]:
            model_info["weights_file"] = f"{locator_config['out']}.weights.h5"
        else:
            model_info["weights_file"] = None

        # Don't include the actual model in the result to avoid serialization issues
        # We'll load it from disk if needed
        result = {
            "fold": fold_idx,
            "gpu_id": gpu_id,
            "train_time": train_time,
            "model_info": {
                "fold": model_info["fold"],
                "weights_file": model_info["weights_file"],
                "norm_params": model_info["norm_params"],
                "train_indices": model_info["train_indices"].tolist(),
                "val_indices": model_info["val_indices"].tolist(),
            },
            "history": {
                "loss": model_info["history"].history.get("loss", []),
                "val_loss": model_info["history"].history.get("val_loss", []),
            },
            "final_loss": float(model_info["history"].history["loss"][-1]),
            "final_val_loss": float(model_info["history"].history["val_loss"][-1]),
        }

        # Clear keras session
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

    Returns:
        dict: Dictionary containing:
            - 'histories': List of training histories for each fold
            - 'models': List of trained model configurations
            - 'normalization_params': Averaged normalization parameters
            - 'fold_info': Information about fold splits
    """
    # Initialize Ray if not already initialized
    if not ray.is_initialized():
        ray.init(
            log_to_driver=False,  # Don't log worker output to driver
            logging_level="ERROR",  # Only show errors
            include_dashboard=False,  # Don't start Ray dashboard
        )

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
    sample_data, locs = locator.sort_samples(samples)

    # Update samples array to match filtered data (after exclusions)
    # sort_samples may have excluded samples, so we need to use the filtered sample IDs
    filtered_samples = sample_data["sampleID"].values

    # If samples were excluded, we need to filter genotypes to match
    if len(filtered_samples) < len(samples):
        # Find indices of samples that remain after exclusion
        samples_list = samples.tolist() if hasattr(samples, "tolist") else list(samples)
        keep_indices = [
            i
            for i, s in enumerate(samples_list)
            if str(s) in set(str(fs) for fs in filtered_samples)
        ]
        genotypes = genotypes[:, keep_indices]
        # Also need to update filtered_genotypes
        filtered_genotypes = filtered_genotypes[:, keep_indices]

    samples = np.array(filtered_samples)

    # Configure augmentation if requested
    augment_config = None
    if augment_data:
        augment_config = {
            "enabled": True,
            "flip_rate": flip_rate,
        }
        # Also set in config for consistency
        locator.config["augmentation"] = augment_config

    # Get sample data for serialization
    sample_data = None
    if hasattr(locator, "_sample_data_df"):
        sample_data = locator._sample_data_df

    # Pre-calculate KDE bandwidth if needed (same pattern as k-fold)
    bandwidth_calculated = False
    original_bandwidth = None

    if (
        locator.config.get("weight_samples", {}).get("enabled", False)
        and locator.config.get("weight_samples", {}).get("method") == "KD"
    ):

        existing_bandwidth = locator.config.get("weight_samples", {}).get("bandwidth")

        if existing_bandwidth is None:
            # Get all samples with coordinates for bandwidth calculation
            na_mask = np.isnan(locs[:, 0]) | np.isnan(locs[:, 1])
            coords_mask = ~na_mask
            all_train_locs = locs[coords_mask]

            if len(all_train_locs) > 1:
                if verbose:
                    print(
                        "Pre-calculating optimal KDE bandwidth for ensemble training..."
                    )

                from locator.sample_weights import get_global_bandwidth_optimizer

                optimizer = get_global_bandwidth_optimizer()

                optimal_bandwidth = optimizer.get_bandwidth(
                    all_train_locs,
                    cache_key=f"ensemble_k{k}_n{len(all_train_locs)}",
                    n_bandwidths=locator.config.get("weight_samples", {}).get(
                        "n_bandwidths", 100
                    ),
                    verbose=verbose,
                )

                # Store original value
                original_bandwidth = existing_bandwidth
                # Set in config
                locator.config["weight_samples"]["bandwidth"] = optimal_bandwidth
                bandwidth_calculated = True

                if verbose:
                    print(f"Using bandwidth: {optimal_bandwidth:.3f}")

    # Save data to temporary file
    with tempfile.NamedTemporaryFile(mode="wb", delete=False, suffix=".pkl") as f:
        data = {
            "genotypes_array": genotypes.values,
            "filtered_genotypes_array": filtered_genotypes,  # Already a numpy array
            "samples": samples,
            "sample_data": sample_data,
            "locs": locs,
            "config": locator.config,
            "fold_info": fold_info,
            "augment_config": augment_config,
            "save_fold_models": save_fold_models,
            "patience_multiplier": patience_multiplier,
        }
        pickle.dump(data, f)
        data_file = f.name

    if verbose:
        print(f"Training {k}-fold ensemble across GPUs {gpu_ids} using Ray...")

    start_time = time.time()

    # Create the Ray worker with specified GPU fraction
    _ray_ensemble_worker = _create_ray_ensemble_worker(gpu_fraction)

    # Submit all folds to Ray
    futures = []
    for fold_idx in range(k):
        # Handle empty gpu_ids (CPU only mode)
        if len(gpu_ids) == 0:
            gpu_id = -1  # Use CPU
        else:
            gpu_id = gpu_ids[fold_idx % len(gpu_ids)]

        future = _ray_ensemble_worker.remote(
            fold_idx=fold_idx, gpu_id=gpu_id, data_file=data_file
        )
        futures.append(future)
        if verbose:
            device_str = "CPU" if gpu_id == -1 else f"GPU {gpu_id}"
            print(f"Submitted fold {fold_idx} to {device_str}")

    # Wait for all folds to complete with progress bar
    if verbose:
        print("\nTraining ensemble folds across GPUs...")
        from tqdm import tqdm

        # Process results with progress bar
        results = []
        with tqdm(total=k, desc="Folds completed") as pbar:
            while futures:
                # Wait for any task to complete
                ready, futures = ray.wait(futures, num_returns=1)
                result = ray.get(ready[0])
                results.append(result)

                # Update progress bar
                pbar.set_postfix_str(
                    f"Last: Fold {result['fold']}, GPU {result['gpu_id']}, "
                    f"Final loss: {result['final_loss']:.4f}"
                )
                pbar.update(1)
    else:
        # No progress bar if not verbose
        results = ray.get(futures)

    total_time = time.time() - start_time

    # Clean up
    os.unlink(data_file)

    if verbose:
        print(
            f"\nCompleted ensemble training in {total_time:.1f}s ({total_time/k:.1f}s per fold)"
        )

        # Show speedup vs sequential
        if len(gpu_ids) > 0:
            num_gpus = len(set(gpu_ids))
            estimated_speedup = k / num_gpus
            print(
                f"Estimated speedup: {estimated_speedup:.1f}x (using {num_gpus} GPU{'s' if num_gpus > 1 else ''})"
            )
        else:
            print("CPU mode - no GPU speedup available")

    # Restore original bandwidth setting if we changed it
    if bandwidth_calculated:
        if original_bandwidth is None:
            # Remove the key if it wasn't there originally
            locator.config.get("weight_samples", {}).pop("bandwidth", None)
        else:
            locator.config["weight_samples"]["bandwidth"] = original_bandwidth

    # Aggregate results (sort by fold index to ensure correct order)
    results_sorted = sorted(results, key=lambda x: x["fold"])

    # Store results on locator instance (mimicking sequential version)
    locator._ensemble_genotypes = genotypes
    locator._ensemble_fold_info = fold_info
    locator._ensemble_models = []
    locator._ensemble_histories = []
    locator._ensemble_norm_params = []

    for result in results_sorted:
        # Reconstruct model info
        model_info = result["model_info"]
        model_info["model"] = None  # Model will be loaded from disk when needed
        model_info["train_indices"] = np.array(model_info["train_indices"])
        model_info["val_indices"] = np.array(model_info["val_indices"])

        # Create history object for compatibility
        class HistoryStub:
            def __init__(self, history_dict):
                self.history = history_dict

        model_info["history"] = HistoryStub(result["history"])

        # Store results
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
        from locator.ensemble_model_manager import EnsembleModelManager

        model_manager = EnsembleModelManager(f"{locator.config['out']}_ensemble")

        # Create a serializable version of config (excluding DataFrames)
        serializable_config = {
            k: v for k, v in locator.config.items() if not isinstance(v, pd.DataFrame)
        }

        ensemble_metadata = {
            "k_folds": k,
            "na_action": na_action or locator.na_action,
            "augment_data": augment_data,
            "config": serializable_config,
            "parallel_training": True,
            "gpu_ids": gpu_ids,
        }

        # Check if we can load models for the model manager
        models_loaded = False
        if verbose:
            print("Checking for saved model weights...")

        for i, model_info in enumerate(locator._ensemble_models):
            if model_info["weights_file"] and os.path.exists(model_info["weights_file"]):
                if not models_loaded and verbose:
                    print("Loading models for ensemble manager...")
                models_loaded = True
                # Create model and load weights
                model = locator._create_model(input_shape=filtered_genotypes.shape[0])
                model.load_weights(model_info["weights_file"])
                model_info["model"] = model
            else:
                if verbose and model_info["weights_file"]:
                    print(
                        f"Warning: Expected weights file not found: {model_info['weights_file']}"
                    )

        if models_loaded:
            model_manager.save_ensemble(locator._ensemble_models, ensemble_metadata)
            if verbose:
                print(f"Saved ensemble to {model_manager.ensemble_dir}")
        else:
            if verbose:
                print(
                    "Models were saved individually by workers, skipping ensemble manager."
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
