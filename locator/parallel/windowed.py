"""Windowed holdout-analysis parallel methods.

Each window trains a model on a different slice of SNP indices but shares
the same train/holdout sample split. The full genotype array is materialised
once on each ``WindowActor`` at startup; per-window calls only ship the SNP
index list. Predictions are streamed to a long-format checkpoint as windows
complete and pivoted to the wide-format output (``x_<label>``, ``y_<label>``)
at the end. ``resume=True`` (default) skips windows already present in the
checkpoint.
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import ray
from ray.util import ActorPool

from locator.data.filters import is_dosage_matrix

from ._actor import DEFAULT_GPU_MEM_MB, make_window_actors
from ._helpers import (
    _ensure_ray_initialized,
    _precalculate_bandwidth,
    _restore_bandwidth,
)


def _windows_chunk_path(locator) -> str:
    return f"{locator.config['out']}_windows_chunks.csv"


def _windows_output_path(locator) -> str:
    return f"{locator.config['out']}_windows_holdouts_predlocs.csv"


def _resume_completed_windows(chunk_path: str) -> set[str]:
    if not os.path.exists(chunk_path):
        return set()
    try:
        existing = pd.read_csv(chunk_path)
    except (pd.errors.EmptyDataError, FileNotFoundError):
        return set()
    if existing.empty or "window_label" not in existing.columns:
        return set()
    return set(existing["window_label"].astype(str).tolist())


def _append_window_to_chunks(chunk_path: str, label: str, rows: list) -> None:
    write_header = not os.path.exists(chunk_path) or os.path.getsize(chunk_path) == 0
    with open(chunk_path, "a") as f:
        if write_header:
            f.write("sampleID,x_pred,y_pred,window_label\n")
        for r in rows:
            f.write(f"{r['sampleID']},{r['x_pred']},{r['y_pred']},{label}\n")


def _load_positions(locator, verbose: bool) -> None:
    """Populate locator.positions (and .chromosomes when available)."""
    if hasattr(locator, "positions") and locator.positions is not None:
        return
    if hasattr(locator, "_genotype_df"):
        locator.positions = np.array(locator._genotype_df.columns, dtype=int)
        return
    if locator.config.get("zarr"):
        import zarr

        callset = zarr.open_group(locator.config["zarr"], mode="r")
        locator.positions = callset["variants/POS"][:]
        return
    if locator.config.get("vcf"):
        if verbose:
            print("Loading SNP positions from VCF...")
        import allel

        vcf = allel.read_vcf(locator.config["vcf"], fields=["POS", "CHROM"])
        if vcf is None or "variants/POS" not in vcf:
            raise ValueError(
                f"Could not load positions from VCF: {locator.config['vcf']}"
            )
        locator.positions = vcf["variants/POS"]
        if "variants/CHROM" in vcf:
            locator.chromosomes = vcf["variants/CHROM"]
        if verbose:
            print(f"Loaded {len(locator.positions)} SNP positions")
        return
    raise ValueError(
        "SNP positions required for windowed analysis. Use VCF, zarr input "
        "or genotype DataFrame with position-labeled columns."
    )


def _resolve_train_test_split(
    locator,
    samples,
    locs,
    na_mask,
    na_action,
    holdout_indices,
    holdout_sample_ids,
    k,
):
    """Build the shared IndexSet used for every window's train/test split."""
    from locator.data.indexset import IndexSet

    n_samples = len(samples)

    if holdout_sample_ids is not None:
        samples_list = samples.tolist() if hasattr(samples, "tolist") else list(samples)
        try:
            holdout_indices = [samples_list.index(sid) for sid in holdout_sample_ids]
        except ValueError:
            missing = [sid for sid in holdout_sample_ids if sid not in samples_list]
            raise ValueError(f"Sample IDs not found in samples list: {missing}")
        k = len(holdout_indices)

    if holdout_indices is not None:
        holdout_idx = np.array(holdout_indices)
        train_mask = np.ones(n_samples, dtype=bool)
        train_mask[holdout_idx] = False
        train_idx = np.where(train_mask)[0]

        if na_mask is not None and na_action in {"exclude", "separate"}:
            valid_mask = ~na_mask
            holdout_idx = holdout_idx[valid_mask[holdout_idx]]
            train_idx = train_idx[valid_mask[train_idx]]

        return IndexSet(
            indices={"train": train_idx, "test": holdout_idx},
            total_samples=n_samples,
            na_mask=na_mask,
        )

    return IndexSet.random_split(
        n=n_samples,
        splits={"train": 1.0 - k / n_samples, "test": k / n_samples},
        seed=locator.config.get("seed", 42),
        na_mask=na_mask,
        na_action=(na_action if na_action != "separate" else "exclude"),
    )


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
    resume: bool = True,
    gpu_mem_mb: int = DEFAULT_GPU_MEM_MB,
) -> Union[pd.DataFrame, None]:
    """Run windowed holdout analysis across multiple GPUs via a ``WindowActor`` pool."""
    _ensure_ray_initialized()

    locator.samples = samples
    locator.genotypes = genotypes

    na_action, _ = locator._validate_na_action(
        samples, na_action, "Windows holdout analysis"
    )

    _load_positions(locator, verbose)

    sample_data, locs = locator._resolve_locations(samples)
    na_mask = np.isnan(locs[:, 0])
    if not na_mask.any():
        na_mask = None

    index_set = _resolve_train_test_split(
        locator,
        samples,
        locs,
        na_mask,
        na_action,
        holdout_indices,
        holdout_sample_ids,
        k,
    )

    if window_stop is None:
        window_stop = max(locator.positions)

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

    bw_train_mask = np.ones(len(samples), dtype=bool)
    bw_train_mask[index_set.test] = False
    bw_train_mask &= ~np.isnan(locs[:, 0])
    bw_calculated, bw_original = _precalculate_bandwidth(
        locator,
        locs[bw_train_mask],
        f"windows_holdouts_n{int(bw_train_mask.sum())}",
        verbose,
    )

    from locator.data.filters import normalize_locs

    (
        meanlong,
        sdlong,
        meanlat,
        sdlat,
        unnormedlocs,
        normalized_locs,
    ) = normalize_locs(locs)

    chunk_path = _windows_chunk_path(locator)
    done_labels = _resume_completed_windows(chunk_path) if resume else set()
    if done_labels and verbose:
        print(
            f"Resume: {len(done_labels)} / {len(windows)} windows already "
            f"present in {chunk_path}; skipping those."
        )

    windows_to_run: List[Dict[str, Any]] = []
    for wi, win in enumerate(windows):
        label = win.get("label", f"pos{win['start']}")
        if label in done_labels:
            continue
        snp_indices = (
            np.where(win["indices"])[0].tolist()
            if "indices" in win
            else np.where(
                (locator.positions >= win["start"]) & (locator.positions < win["stop"])
            )[0].tolist()
        )
        windows_to_run.append(
            {
                "window_idx": wi,
                "window_label": label,
                "window_chromosome": win.get("chromosome"),
                "window_start": win["start"],
                "window_stop": win["stop"],
                "snp_indices": snp_indices,
            }
        )

    if windows_to_run:
        # Note: windowed analysis must materialize the full genotype array on
        # the actor because every window slices a different SNP subset; we
        # can't pre-filter to a small array as the other dispatchers do.
        # Continuous-dosage (GL) input is a bare 2D ndarray with no ``.values``;
        # ship it directly. Hard-call inputs are GenotypeArray/DataFrame-like.
        genotypes_array = genotypes if is_dosage_matrix(genotypes) else genotypes.values
        data_ref = ray.put(
            {
                "genotypes_array": genotypes_array,
                "samples": samples,
                "sample_data": sample_data,
                "config": locator.config,
                "index_set": index_set,
                "meanlong": meanlong,
                "sdlong": sdlong,
                "meanlat": meanlat,
                "sdlat": sdlat,
                "unnormedlocs": unnormedlocs,
                "normalized_locs": normalized_locs,
            }
        )

        actors = make_window_actors(
            gpu_ids=gpu_ids,
            gpu_fraction=gpu_fraction,
            data_ref=data_ref,
            gpu_mem_mb=gpu_mem_mb,
        )
        if verbose:
            print(
                f"Spawned {len(actors)} WindowActor actors across GPUs {gpu_ids}; "
                f"dispatching {len(windows_to_run)} windows "
                f"(gpu_fraction={gpu_fraction})."
            )

        pool = ActorPool(actors)
        start_time = time.time()

        pbar = None
        if verbose:
            try:
                from tqdm import tqdm

                pbar = tqdm(total=len(windows_to_run), desc="Windows")
            except ImportError:
                pass

        completed = 0
        for result in pool.map_unordered(
            lambda actor, w: actor.run_window.remote(
                w["window_idx"],
                w["window_label"],
                w["window_chromosome"],
                w["window_start"],
                w["window_stop"],
                w["snp_indices"],
            ),
            windows_to_run,
        ):
            if result["rows"]:
                _append_window_to_chunks(
                    chunk_path, result["window_label"], result["rows"]
                )
            completed += 1
            if pbar is not None:
                pbar.update(1)

        if pbar is not None:
            pbar.close()

        total_time = time.time() - start_time
        if verbose:
            print(
                f"\nCompleted {completed} windows in {total_time:.1f}s "
                f"({total_time / max(completed, 1):.1f}s per window)."
            )

        for actor in actors:
            ray.kill(actor)
    elif verbose:
        print("All windows already complete.")

    _restore_bandwidth(locator, bw_calculated, bw_original)

    if not return_df:
        return None

    if not os.path.exists(chunk_path):
        if verbose:
            print("Warning: No windows produced predictions.")
        return None

    chunks = pd.read_csv(chunk_path)
    if chunks.empty:
        if verbose:
            print("Warning: No windows produced predictions.")
        return None

    wide = chunks.pivot_table(
        index="sampleID",
        columns="window_label",
        values=["x_pred", "y_pred"],
        aggfunc="first",
    )
    wide.columns = [f"{val.replace('_pred', '')}_{label}" for val, label in wide.columns]
    wide = wide.reset_index()

    if save_full_pred_matrix:
        wide.to_csv(_windows_output_path(locator), index=False)

    return wide
