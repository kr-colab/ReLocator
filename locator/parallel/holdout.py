"""Holdout replicate parallel methods."""

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

        locator = _create_worker_locator(data, f"rep{rep_idx}")

        # Use pre-filtered allele counts if available
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

    na_action, status = locator._validate_na_action(
        samples, na_action, "Holdout analysis"
    )

    sample_data, locs = locator._resolve_locations(samples)

    # Get indices of samples with known locations
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
