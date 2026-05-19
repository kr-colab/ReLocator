"""Ensemble training parallel methods.

Each fold trains a single model via ``Locator._train_single_fold`` and
returns model metadata + (optionally) a weights file. The dispatcher
aggregates fold outputs into the ensemble structure on the driver side.

Uses the ``EnsembleActor`` Ray actor so the TF runtime + Locator are paid
once per GPU slot rather than per fold; XLA / autograph caches survive.
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import ray
from ray.util import ActorPool

from ._actor import DEFAULT_GPU_MEM_MB, make_ensemble_actors
from ._helpers import (
    _ensure_ray_initialized,
    _precalculate_bandwidth,
    _restore_bandwidth,
)


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
    gpu_mem_mb: int = DEFAULT_GPU_MEM_MB,
) -> Dict[str, Any]:
    """Train an ensemble of k models across multiple GPUs using Ray actors.

    See ``parallel_k_fold_holdouts`` for GPU/memory semantics. The return
    dict matches the original API: ``histories``, ``models``,
    ``normalization_params``, ``fold_info``, ``training_time``, ``parallel``.
    """
    _ensure_ray_initialized()

    if verbose:
        mixed_precision_enabled = locator.setup_ensemble_gpu_optimization(
            use_mixed_precision
        )
        if mixed_precision_enabled:
            print("Mixed precision training enabled for ensemble")
    else:
        locator.setup_ensemble_gpu_optimization(use_mixed_precision)

    locator.samples = samples
    fold_info = locator.create_ensemble_folds(
        genotypes, samples, k, training_set_indices, na_action
    )
    filtered_genotypes = locator._filter_genotypes(genotypes)
    _, locs = locator._resolve_locations(samples)

    augment_config = None
    if augment_data:
        augment_config = {"enabled": True, "flip_rate": flip_rate}
        locator.config["augmentation"] = augment_config

    sample_data = getattr(locator, "_sample_data_df", None)

    na_mask = np.isnan(locs[:, 0]) | np.isnan(locs[:, 1])
    bw_calculated, bw_original = _precalculate_bandwidth(
        locator,
        locs[~na_mask],
        f"ensemble_k{k}_n{int((~na_mask).sum())}",
        verbose,
    )

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

    actors = make_ensemble_actors(
        gpu_ids=gpu_ids,
        gpu_fraction=gpu_fraction,
        data_ref=data_ref,
        gpu_mem_mb=gpu_mem_mb,
    )
    if verbose:
        print(
            f"Spawned {len(actors)} EnsembleActor actors across GPUs {gpu_ids}; "
            f"training {k} folds (gpu_fraction={gpu_fraction})."
        )

    pool = ActorPool(actors)
    start_time = time.time()

    pbar = None
    if verbose:
        try:
            from tqdm import tqdm

            pbar = tqdm(total=k, desc="Folds")
        except ImportError:
            pass

    results = []
    for result in pool.map_unordered(
        lambda actor, fold_idx: actor.run_ensemble_fold.remote(fold_idx),
        list(range(k)),
    ):
        results.append(result)
        if pbar is not None:
            pbar.set_postfix_str(
                f"Last: Fold {result['fold']}, Final loss: {result['final_loss']:.4f}"
            )
            pbar.update(1)

    if pbar is not None:
        pbar.close()

    total_time = time.time() - start_time
    if verbose:
        print(
            f"\nCompleted ensemble training in {total_time:.1f}s "
            f"({total_time / max(k, 1):.1f}s per fold)"
        )

    for actor in actors:
        ray.kill(actor)

    _restore_bandwidth(locator, bw_calculated, bw_original)

    results_sorted = sorted(results, key=lambda x: x["fold"])

    locator._ensemble_genotypes = genotypes
    locator._ensemble_fold_info = fold_info
    locator._ensemble_models = []
    locator._ensemble_histories = []
    locator._ensemble_norm_params = []

    class _HistoryStub:
        def __init__(self, history_dict):
            self.history = history_dict

    for result in results_sorted:
        m = result["model_info"]
        m["model"] = None
        m["train_indices"] = np.array(m["train_indices"])
        m["val_indices"] = np.array(m["val_indices"])
        m["history"] = _HistoryStub(result["history"])
        locator._ensemble_models.append(m)
        locator._ensemble_histories.append(m["history"])
        locator._ensemble_norm_params.append(m["norm_params"])

    avg_norm_params = locator._average_normalization_params(
        locator._ensemble_norm_params
    )
    locator.meanlong = avg_norm_params["meanlong"]
    locator.sdlong = avg_norm_params["sdlong"]
    locator.meanlat = avg_norm_params["meanlat"]
    locator.sdlat = avg_norm_params["sdlat"]

    if use_model_manager and save_fold_models:
        from locator.ensemble_model_manager import EnsembleModelManager

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
        for m in locator._ensemble_models:
            wf = m["weights_file"]
            if wf and os.path.exists(wf):
                models_loaded = True
                model = locator._create_model(input_shape=filtered_genotypes.shape[0])
                model.load_weights(wf)
                m["model"] = model
            elif verbose and wf:
                print(f"Warning: Expected weights file not found: {wf}")

        if models_loaded:
            model_manager.save_ensemble(locator._ensemble_models, ensemble_metadata)
            if verbose:
                print(f"Saved ensemble to {model_manager.ensemble_dir}")
        elif verbose:
            print(
                "Models were saved individually by workers; skipping ensemble manager."
            )

    return {
        "histories": locator._ensemble_histories,
        "models": locator._ensemble_models,
        "normalization_params": avg_norm_params,
        "fold_info": fold_info,
        "training_time": total_time,
        "parallel": True,
    }
