"""Holdout-replicate parallel methods.

A holdout replicate is just a k-fold-style train_holdout call with a chosen
set of held-out indices; the work is the same as one fold of k-fold CV. This
module dispatches replicates across the same ``FoldWorker`` actor pool used
by ``locator.parallel.kfold`` and aggregates predictions into the wide-format
DataFrame ``[sampleID, x_rep0, y_rep0, x_rep1, y_rep1, ...]`` that the
existing API contract requires.

Per-replicate predictions are streamed to a long-format checkpoint CSV
(``{out}_holdouts_chunks.csv``) as each replicate completes; the wide-format
output is pivoted from that file at the end. ``resume=True`` (default) skips
replicates already present in the checkpoint.
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


def _holdout_chunk_path(locator) -> str:
    return f"{locator.config['out']}_holdouts_chunks.csv"


def _holdout_output_path(locator) -> str:
    return f"{locator.config['out']}_holdouts_predlocs.csv"


def _resume_completed_reps(chunk_path: str) -> set[int]:
    if not os.path.exists(chunk_path):
        return set()
    try:
        existing = pd.read_csv(chunk_path)
    except (pd.errors.EmptyDataError, FileNotFoundError):
        return set()
    if existing.empty or "rep" not in existing.columns:
        return set()
    return set(existing["rep"].astype(int).tolist())


def _append_rep_to_chunks(chunk_path: str, rep: int, rows: list) -> None:
    write_header = not os.path.exists(chunk_path) or os.path.getsize(chunk_path) == 0
    with open(chunk_path, "a") as f:
        if write_header:
            f.write("sampleID,x_pred,y_pred,rep\n")
        for r in rows:
            f.write(f"{r['sampleID']},{r['x_pred']},{r['y_pred']},{rep}\n")


def _resolve_holdout_indices(
    holdout_indices,
    holdout_sample_ids,
    samples,
    known_idx,
    k,
    n_reps,
):
    """Materialise per-replicate holdout index arrays from the various input forms."""
    if holdout_sample_ids is not None:
        samples_list = samples.tolist() if hasattr(samples, "tolist") else list(samples)
        if isinstance(holdout_sample_ids[0], str):
            try:
                fixed = [samples_list.index(sid) for sid in holdout_sample_ids]
            except ValueError:
                missing = [sid for sid in holdout_sample_ids if sid not in samples_list]
                raise ValueError(f"Sample IDs not found in samples list: {missing}")
            return [fixed for _ in range(n_reps)], len(holdout_sample_ids)
        all_indices = []
        for rep_ids in holdout_sample_ids:
            try:
                rep_indices = [samples_list.index(sid) for sid in rep_ids]
            except ValueError:
                missing = [sid for sid in rep_ids if sid not in samples_list]
                raise ValueError(f"Sample IDs not found in samples list: {missing}")
            all_indices.append(rep_indices)
        return all_indices, k

    all_indices = []
    for rep in range(n_reps):
        if holdout_indices is not None and rep < len(holdout_indices):
            all_indices.append(list(holdout_indices[rep]))
        else:
            all_indices.append(np.random.choice(known_idx, k, replace=False).tolist())
    return all_indices, k


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
    resume: bool = True,
    gpu_mem_mb: int = DEFAULT_GPU_MEM_MB,
) -> Union[pd.DataFrame, None]:
    """Run multiple holdout replicates in parallel via a ``FoldWorker`` actor pool.

    See ``parallel_k_fold_holdouts`` for the GPU/memory semantics. Output is
    in wide format (``x_rep0, y_rep0, x_rep1, ...``) to match the existing
    API contract; a long-format per-replicate checkpoint is written
    alongside for resume safety.
    """
    _ensure_ray_initialized()

    na_action, _ = locator._validate_na_action(samples, na_action, "Holdout analysis")
    sample_data, locs = locator._resolve_locations(samples)

    known_idx = np.where(~np.isnan(locs[:, 0]))[0]
    if k >= len(known_idx):
        raise ValueError(
            f"k ({k}) must be less than number of samples with known locations "
            f"({len(known_idx)})"
        )

    bw_calculated, bw_original = _precalculate_bandwidth(
        locator,
        locs[known_idx],
        f"holdouts_k{k}_n{len(known_idx)}",
        verbose,
    )

    all_holdout_indices, k = _resolve_holdout_indices(
        holdout_indices,
        holdout_sample_ids,
        samples,
        known_idx,
        k,
        n_reps,
    )
    n_reps = len(all_holdout_indices)

    filtered_genotypes = locator._filter_genotypes(genotypes)
    if verbose:
        print(
            f"Pre-filtered genotypes: {genotypes.shape[0]:,} → "
            f"{filtered_genotypes.shape[0]:,} SNPs"
        )

    chunk_path = _holdout_chunk_path(locator)
    done_reps = _resume_completed_reps(chunk_path) if resume else set()
    if done_reps and verbose:
        print(
            f"Resume: {len(done_reps)} / {n_reps} replicates already present in "
            f"{chunk_path}; skipping those."
        )

    reps_to_run = [i for i in range(n_reps) if i not in done_reps]

    if reps_to_run:
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

        actors = make_fold_workers(
            gpu_ids=gpu_ids,
            gpu_fraction=gpu_fraction,
            data_ref=data_ref,
            gpu_mem_mb=gpu_mem_mb,
        )
        if verbose:
            print(
                f"Spawned {len(actors)} FoldWorker actors across GPUs {gpu_ids}; "
                f"dispatching {len(reps_to_run)} replicates "
                f"(gpu_fraction={gpu_fraction})."
            )

        pool = ActorPool(actors)
        rep_args = [(i, all_holdout_indices[i]) for i in reps_to_run]

        pbar = None
        if verbose:
            try:
                from tqdm import tqdm

                pbar = tqdm(total=len(reps_to_run), desc="Replicates")
            except ImportError:
                pass

        start_time = time.time()
        completed = 0
        for result in pool.map_unordered(
            lambda actor, args: actor.run_holdout.remote(args[0], args[1]),
            rep_args,
        ):
            _append_rep_to_chunks(chunk_path, result["idx"], result["rows"])
            completed += 1
            if pbar is not None:
                pbar.update(1)

        if pbar is not None:
            pbar.close()

        total_time = time.time() - start_time
        if verbose:
            per_rep = total_time / max(completed, 1)
            print(
                f"\nCompleted {completed} replicates in {total_time:.1f}s "
                f"({per_rep:.1f}s per replicate)."
            )

        for actor in actors:
            ray.kill(actor)
    else:
        if verbose:
            print("All replicates already complete.")

    _restore_bandwidth(locator, bw_calculated, bw_original)

    if not return_df:
        return None

    # Pivot the long-format chunks to the wide-format output contract.
    chunks = pd.read_csv(chunk_path)
    wide = chunks.pivot_table(
        index="sampleID", columns="rep", values=["x_pred", "y_pred"], aggfunc="first"
    )
    wide.columns = [
        f"{val}_rep{rep}".replace("x_pred", "x").replace("y_pred", "y")
        for val, rep in wide.columns
    ]
    wide = wide.reset_index()

    if save_full_pred_matrix:
        wide.to_csv(_holdout_output_path(locator), index=False)

    return wide
