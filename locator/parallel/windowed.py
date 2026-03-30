"""Windowed holdout analysis parallel methods."""

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

    # Store samples and genotypes
    locator.samples = samples
    locator.genotypes = genotypes

    na_action, status = locator._validate_na_action(
        samples, na_action, "Windows holdout analysis"
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

    sample_data, locs = locator._resolve_locations(samples)

    # Derive NA mask from resolved locations
    na_mask = np.isnan(locs[:, 0])
    if not na_mask.any():
        na_mask = None

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

    # Share data via Ray object store.
    # NOTE: Windowed analysis requires the full genotype array because each
    # window slices different SNP indices. Pre-filtering is not possible here
    # unlike k-fold/holdout dispatchers.
    data_ref = ray.put(
        {
            "genotypes_array": genotypes.values,
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
