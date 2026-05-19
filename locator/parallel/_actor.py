"""Ray actor that runs many folds against a hot TF runtime.

The previous task-based dispatcher recreated the Locator + Keras runtime per
fold, paying a fresh XLA / autograph compile every time. For 100+ LOO folds
that overhead dominates wall time. The actor here is created once per GPU
slot (driven by ``gpu_fraction``) and serves folds via ``run_fold`` — the
Locator instance, TF imports, and the JIT cache all survive between folds;
only model weights are rebuilt.
"""

from __future__ import annotations

from typing import Any, Dict, List

import ray

from ._helpers import _create_worker_locator, _setup_worker_env

DEFAULT_GPU_MEM_MB = 80_000  # A100 80GB; safe default for the lab box.


@ray.remote
class FoldWorker:
    """Long-lived Ray actor that processes folds against a hot TF runtime.

    Lifecycle:
        ``__init__``       configure GPU memory cap, import TF, instantiate Locator
        ``run_fold(idx, holdout_indices)``  rebuild model + train + predict
        ``ready()``        cheap probe used to surface init errors before dispatch

    One actor per scheduling slot — ``num_gpus`` reserves the slot; the
    memory cap below enforces it. The Locator + filtered genotype array
    live on the actor and are reused for every fold it handles.
    """

    def __init__(
        self,
        gpu_id: int,
        gpu_fraction: float,
        data_ref,
        gpu_mem_mb: int = DEFAULT_GPU_MEM_MB,
    ):
        _setup_worker_env(gpu_id)

        import tensorflow as tf

        gpus = tf.config.list_physical_devices("GPU")
        if gpus and gpu_id != -1:
            # Hard cap matching ``num_gpus`` so co-tenant actors can't
            # collectively blow the device budget when XLA workspaces peak.
            # ~5% margin for TF runtime overhead.
            limit_mb = max(512, int(gpu_fraction * gpu_mem_mb * 0.95))
            try:
                tf.config.set_logical_device_configuration(
                    gpus[0],
                    [tf.config.LogicalDeviceConfiguration(memory_limit=limit_mb)],
                )
            except RuntimeError:
                # set_logical_device_configuration is illegal once TF has
                # touched the GPU. Memory growth keeps us alive but doesn't
                # cap, so co-tenants can still OOM each other.
                tf.config.experimental.set_memory_growth(gpus[0], True)

        tf.get_logger().setLevel("ERROR")
        self._tf = tf
        # Ray auto-dereferences the ObjectRef when it's passed as an actor
        # method arg; ``data_ref`` arrives as the resolved dict.
        self._data: Dict[str, Any] = data_ref
        # Single Locator reused across folds; train_holdout overwrites all
        # per-fold state so this is safe.
        self._locator = _create_worker_locator(self._data, "actor")
        self._base_out = self._data["config"]["out"]

    def ready(self) -> bool:
        """Probe that init completed without raising. Used to fail fast."""
        return True

    def run_fold(self, fold_idx: int, holdout_indices: List[int]) -> Dict[str, Any]:
        """Train one fold and return predictions for the held-out samples."""
        loc = self._locator
        # Per-fold output prefix so any per-fold artifacts don't collide.
        loc.config["out"] = f"{self._base_out}_fold{fold_idx}"

        loc.train_holdout(
            genotypes=None,
            samples=self._data["samples"],
            holdout_indices=holdout_indices,
            filtered_genotypes=self._data["filtered_genotypes"],
        )
        preds = loc.predict_holdout(
            verbose=False,
            return_df=True,
            save_preds_to_disk=False,
            plot_summary=False,
            plot_map=False,
        )
        return {
            "fold": fold_idx,
            "rows": preds[["sampleID", "x_pred", "y_pred"]].to_dict("records"),
        }


def make_fold_workers(
    gpu_ids: List[int],
    gpu_fraction: float,
    data_ref,
    gpu_mem_mb: int = DEFAULT_GPU_MEM_MB,
) -> List[Any]:
    """Spawn one ``FoldWorker`` actor per scheduling slot, distributed across GPUs.

    Number of actors = ``len(gpu_ids) * round(1 / gpu_fraction)``; each actor
    reserves ``gpu_fraction`` of one GPU via Ray. Actors are pinned to GPUs
    round-robin.
    """
    if not gpu_ids:
        gpu_assignments = [-1]  # CPU-only
        gpu_fraction = 0.0
    else:
        workers_per_gpu = max(1, int(round(1.0 / gpu_fraction)))
        gpu_assignments = [
            gpu_ids[i % len(gpu_ids)] for i in range(len(gpu_ids) * workers_per_gpu)
        ]

    actors = []
    for gpu_id in gpu_assignments:
        actor = FoldWorker.options(num_gpus=gpu_fraction).remote(
            gpu_id=gpu_id,
            gpu_fraction=gpu_fraction,
            data_ref=data_ref,
            gpu_mem_mb=gpu_mem_mb,
        )
        actors.append(actor)

    # Block until every actor has run __init__ — surfaces config errors
    # (bad GPU id, memory cap rejection, etc.) before we dispatch folds.
    ray.get([a.ready.remote() for a in actors])
    return actors
