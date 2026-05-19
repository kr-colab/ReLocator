"""K-fold cross-validation parallel methods.

This module dispatches folds across a pool of long-lived Ray actors
(``FoldWorker``) rather than fresh Ray tasks. The actor approach keeps the
TF runtime + autograph + XLA caches hot across folds, eliminating the
multi-second cold-start tax that dominated wall time in LOO sweeps.

The dispatcher also writes per-fold predictions to disk as they complete,
so a kill/crash mid-sweep doesn't discard already-trained folds — re-running
with ``resume=True`` (default) picks up where the previous run left off.
"""

from __future__ import annotations

import os
import time
from typing import List, Optional, Union

import numpy as np
import pandas as pd
import ray
from ray.util import ActorPool

from ._actor import DEFAULT_GPU_MEM_MB, make_fold_workers
from ._helpers import (
    _ensure_ray_initialized,
    _precalculate_bandwidth,
    _restore_bandwidth,
)


def _kfold_output_path(locator) -> str:
    return f"{locator.config['out']}_kfold_holdouts_predlocs.csv"


def _resume_completed_folds(output_path: str) -> set[int]:
    """Return the set of fold indices already present in ``output_path``."""
    if not os.path.exists(output_path):
        return set()
    try:
        existing = pd.read_csv(output_path)
    except (pd.errors.EmptyDataError, FileNotFoundError):
        return set()
    if existing.empty or "fold" not in existing.columns:
        return set()
    return set(existing["fold"].astype(int).tolist())


def _append_fold_to_csv(output_path: str, fold: int, rows: list) -> None:
    """Append one fold's predictions to the running CSV, with header on first write."""
    write_header = not os.path.exists(output_path) or os.path.getsize(output_path) == 0
    with open(output_path, "a") as f:
        if write_header:
            f.write("sampleID,x_pred,y_pred,fold\n")
        for r in rows:
            f.write(f"{r['sampleID']},{r['x_pred']},{r['y_pred']},{fold}\n")


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
    resume: bool = True,
    gpu_mem_mb: int = DEFAULT_GPU_MEM_MB,
) -> Union[pd.DataFrame, None]:
    """Run true k-fold cross-validation across multiple GPUs via Ray actors.

    Args:
        locator: Locator instance (config + helper methods).
        genotypes: GenotypeArray (or pre-filtered dosage matrix).
        samples: Sample IDs corresponding to genotypes.
        k: Number of folds.
        gpu_ids: List of GPU IDs to dispatch across.
        gpu_fraction: Fraction of one GPU reserved per actor. ``1.0`` is one
            actor per GPU; ``0.5`` packs two actors per GPU; ``0.0`` runs CPU
            only. The actor also caps TF GPU memory at this fraction so
            co-tenants can't OOM each other.
        return_df: Return aggregated DataFrame at the end.
        save_full_pred_matrix: Write the aggregated CSV at the end (the
            running per-fold CSV is always written regardless).
        verbose: Print progress + per-fold timing.
        na_action: ``'separate' | 'exclude' | 'fail'``; defaults to locator's.
        resume: Skip folds already present in the on-disk CSV.
        gpu_mem_mb: Total GPU memory in MB used to compute the per-actor cap.
            Default 80GB matches A100; override for other devices.

    Returns
    -------
        DataFrame with columns ``[sampleID, x_pred, y_pred, fold]`` (or
        ``None`` if ``return_df=False``).
    """
    _ensure_ray_initialized()

    na_action, _ = locator._validate_na_action(samples, na_action, "K-fold CV")

    sample_data, locs = locator._resolve_locations(samples)
    na_mask = np.isnan(locs[:, 0])
    n_total_samples = len(locs)
    n_with_coords = int(np.sum(~na_mask))

    if k > n_with_coords:
        raise ValueError(
            f"k ({k}) must be <= number of samples with known locations "
            f"({n_with_coords})"
        )

    from locator.data.indexset import IndexSet

    if "seed" in locator.config and locator.config["seed"] is not None:
        kfold_seed = locator.config["seed"]
    else:
        kfold_seed = np.random.randint(0, 2**31)

    fold_index_sets = [
        IndexSet.from_k_fold(
            n=n_total_samples,
            k=k,
            fold=i,
            seed=kfold_seed,
            na_mask=na_mask,
        )
        for i in range(k)
    ]

    bw_calculated, bw_original = _precalculate_bandwidth(
        locator,
        locs[~na_mask],
        f"kfold_k{k}_n{n_with_coords}",
        verbose,
    )

    filtered_genotypes = locator._filter_genotypes(genotypes)
    if verbose:
        print(
            f"Pre-filtered genotypes: {genotypes.shape[0]:,} → "
            f"{filtered_genotypes.shape[0]:,} SNPs"
        )

    output_path = _kfold_output_path(locator)
    done_folds = _resume_completed_folds(output_path) if resume else set()
    if done_folds and verbose:
        print(
            f"Resume: {len(done_folds)} / {k} folds already present in "
            f"{output_path}; skipping those."
        )

    folds_to_run = [i for i in range(k) if i not in done_folds]

    if not folds_to_run:
        if verbose:
            print("All folds already complete.")
        _restore_bandwidth(locator, bw_calculated, bw_original)
        return pd.read_csv(output_path) if return_df else None

    data_ref = ray.put(
        {
            "filtered_genotypes": filtered_genotypes,
            "samples": samples,
            "sample_data": sample_data,
            "locs": locs,
            "config": locator.config,
            "na_mask": na_mask,
        }
    )

    actors = make_fold_workers(
        gpu_ids=gpu_ids,
        gpu_fraction=gpu_fraction,
        data_ref=data_ref,
        gpu_mem_mb=gpu_mem_mb,
    )
    if verbose:
        print(
            f"Spawned {len(actors)} FoldWorker actors across GPUs {gpu_ids}; "
            f"dispatching {len(folds_to_run)} folds (gpu_fraction={gpu_fraction})."
        )

    pool = ActorPool(actors)
    fold_args = [(i, fold_index_sets[i].test.tolist()) for i in folds_to_run]
    start_time = time.time()

    pbar = None
    if verbose:
        try:
            from tqdm import tqdm

            pbar = tqdm(total=len(folds_to_run), desc="Folds")
        except ImportError:
            pass

    completed = 0
    for result in pool.map_unordered(
        lambda actor, args: actor.run_fold.remote(args[0], args[1]),
        fold_args,
    ):
        _append_fold_to_csv(output_path, result["fold"], result["rows"])
        completed += 1
        if pbar is not None:
            pbar.update(1)

    if pbar is not None:
        pbar.close()

    total_time = time.time() - start_time
    if verbose:
        per_fold = total_time / max(completed, 1)
        print(
            f"\nCompleted {completed} folds in {total_time:.1f}s "
            f"({per_fold:.1f}s per fold)."
        )

    _restore_bandwidth(locator, bw_calculated, bw_original)

    # Kill actors so the TF GPU memory is released before the caller returns.
    for actor in actors:
        ray.kill(actor)

    df = pd.read_csv(output_path)

    expected = {samples[i] for i in range(len(samples)) if not na_mask[i]}
    actual = set(df["sampleID"].unique())
    missing = expected - actual
    extra = actual - expected
    if missing or extra:
        print("WARNING: Sample mismatch in final results!")
        print(f"  expected {len(expected)} unique samples, got {len(actual)}")
        if missing:
            print(
                f"  missing: {sorted(missing)[:10]}{'...' if len(missing) > 10 else ''}"
            )
        if extra:
            print(f"  extra:   {sorted(extra)[:10]}{'...' if len(extra) > 10 else ''}")

    if not save_full_pred_matrix:
        # The on-disk CSV is the per-fold checkpoint; if the caller doesn't
        # want a final aggregate the file still exists. Nothing else to do.
        pass

    return df if return_df else None


def parallel_leave_one_out(
    locator,
    genotypes,
    samples,
    gpu_ids: List[int] = [0, 1],
    gpu_fraction: float = 1.0,
    return_df: bool = True,
    save_full_pred_matrix: bool = True,
    na_action: Optional[str] = None,
    resume: bool = True,
    gpu_mem_mb: int = DEFAULT_GPU_MEM_MB,
) -> Union[pd.DataFrame, None]:
    """Leave-one-out cross-validation via the k-fold dispatcher.

    Equivalent to ``parallel_k_fold_holdouts`` with ``k = n_samples_with_coords``.
    Writes to ``{out}_leave_one_out_predlocs.csv`` instead of the k-fold path
    so partial LOO and k-fold runs against the same Locator output prefix
    don't collide.
    """
    status = locator.get_sample_status(samples)
    n_known = status["n_known"]
    if n_known == 0:
        raise ValueError("No samples with known coordinates for leave-one-out CV")

    # Route LOO results through a dedicated checkpoint path. We temporarily
    # patch the Locator's ``out`` prefix so the k-fold helper writes to the
    # LOO file; this avoids replicating the dispatcher logic.
    original_out = locator.config["out"]
    locator.config["out"] = original_out + "__loo"
    try:
        df = parallel_k_fold_holdouts(
            locator=locator,
            genotypes=genotypes,
            samples=samples,
            k=n_known,
            gpu_ids=gpu_ids,
            gpu_fraction=gpu_fraction,
            return_df=return_df,
            save_full_pred_matrix=False,
            verbose=True,
            na_action=na_action,
            resume=resume,
            gpu_mem_mb=gpu_mem_mb,
        )
    finally:
        locator.config["out"] = original_out

    if df is not None and save_full_pred_matrix:
        loo_path = f"{original_out}_leave_one_out_predlocs.csv"
        df.to_csv(loo_path, index=False)

    return df
