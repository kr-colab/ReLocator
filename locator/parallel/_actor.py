"""Ray actors that run many units of parallel work against a hot TF runtime.

The previous task-based dispatchers (one Ray task per fold / replicate /
window) recreated the Locator + Keras runtime per unit of work, paying a
fresh XLA / autograph compile every time. For LOO with hundreds of folds
or windowed analysis with thousands of windows, that overhead dominates
wall time. Each actor here is created once per GPU slot (driven by
``gpu_fraction``) and serves many units of work via a single method; the
Locator instance, TF imports, and the JIT cache all survive between calls.

Three actor classes share an init helper and differ only in their work
method:
    FoldWorker      train_holdout + predict_holdout (k-fold/LOO/holdout)
    EnsembleActor   _train_single_fold (ensemble training)
    WindowActor     train_window + predict_holdout (windowed analysis)
"""

from __future__ import annotations

from typing import Any, Dict, List

import ray

from ._helpers import _create_worker_locator, _setup_worker_env

DEFAULT_GPU_MEM_MB = 80_000  # A100 80GB; safe default for the lab box.


def _init_actor_runtime(gpu_id: int, gpu_fraction: float, gpu_mem_mb: int):
    """Common GPU + TF setup for a Ray actor. Returns the imported ``tf`` module.

    Sets ``CUDA_VISIBLE_DEVICES``, then before TF touches the device installs a
    hard memory cap matching ``gpu_fraction`` so co-tenant actors can't
    collectively blow the budget when XLA workspaces peak. Falls back to
    ``memory_growth`` if the cap can't be set (TF already initialized).
    """
    _setup_worker_env(gpu_id)
    import tensorflow as tf

    gpus = tf.config.list_physical_devices("GPU")
    if gpus and gpu_id != -1:
        limit_mb = max(512, int(gpu_fraction * gpu_mem_mb * 0.95))
        try:
            tf.config.set_logical_device_configuration(
                gpus[0],
                [tf.config.LogicalDeviceConfiguration(memory_limit=limit_mb)],
            )
        except RuntimeError:
            tf.config.experimental.set_memory_growth(gpus[0], True)

    tf.get_logger().setLevel("ERROR")
    return tf


@ray.remote
class FoldWorker:
    """Ray actor for k-fold / LOO / holdout-replicate work.

    Each ``run_holdout`` call rebuilds the Locator's model from scratch (TF
    runtime + autograph cache survives), trains with a different set of
    held-out indices, and returns predictions for those samples.
    """

    def __init__(
        self,
        gpu_id: int,
        gpu_fraction: float,
        data_ref,
        gpu_mem_mb: int = DEFAULT_GPU_MEM_MB,
    ):
        self._tf = _init_actor_runtime(gpu_id, gpu_fraction, gpu_mem_mb)
        # Ray auto-dereferences the ObjectRef when it's passed as an actor
        # method arg; ``data_ref`` arrives as the resolved dict.
        self._data: Dict[str, Any] = data_ref
        self._locator = _create_worker_locator(self._data, "actor")
        self._base_out = self._data["config"]["out"]

    def ready(self) -> bool:
        """Probe that init completed without raising. Used to fail fast."""
        return True

    def run_holdout(self, idx: int, holdout_indices: List[int]) -> Dict[str, Any]:
        """Train one fold/replicate and return predictions for the held-out samples."""
        loc = self._locator
        loc.config["out"] = f"{self._base_out}_fold{idx}"
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
            "idx": idx,
            "rows": preds[["sampleID", "x_pred", "y_pred"]].to_dict("records"),
        }


@ray.remote
class EnsembleActor:
    """Ray actor for ensemble training (one model per fold).

    Each ``run_ensemble_fold`` call runs ``_train_single_fold`` and returns the
    fold's weights-file path + history + per-fold normalization parameters so
    the dispatcher can rebuild the ensemble structure on the driver side.
    """

    def __init__(
        self,
        gpu_id: int,
        gpu_fraction: float,
        data_ref,
        gpu_mem_mb: int = DEFAULT_GPU_MEM_MB,
    ):
        self._tf = _init_actor_runtime(gpu_id, gpu_fraction, gpu_mem_mb)
        self._data: Dict[str, Any] = data_ref
        self._locator = _create_worker_locator(self._data, "actor")
        self._base_out = self._data["config"]["out"]

    def ready(self) -> bool:
        return True

    def run_ensemble_fold(self, fold_idx: int) -> Dict[str, Any]:
        loc = self._locator
        loc.config["out"] = f"{self._base_out}_fold{fold_idx}"
        save_fold_models = self._data["save_fold_models"]

        model_info = loc._train_single_fold(
            fold_idx=fold_idx,
            index_set=self._data["fold_info"]["index_sets"][fold_idx],
            filtered_genotypes=self._data["filtered_genotypes"],
            samples=self._data["samples"],
            locs=self._data["locs"],
            augment_config=self._data.get("augment_config"),
            save_fold_models=save_fold_models,
            patience_multiplier=self._data["patience_multiplier"],
            verbose=False,
        )

        weights_file = f"{loc.config['out']}.weights.h5" if save_fold_models else None

        return {
            "fold": fold_idx,
            "model_info": {
                "fold": model_info["fold"],
                "weights_file": weights_file,
                "norm_params": model_info["norm_params"],
                "train_indices": model_info["train_indices"].tolist(),
                "val_indices": model_info["val_indices"].tolist(),
            },
            "history": {
                "loss": model_info["history"].history.get("loss", []),
                "val_loss": model_info["history"].history.get("val_loss", []),
            },
            "final_loss": (
                float(model_info["history"].history["loss"][-1])
                if model_info["history"].history.get("loss")
                else float("nan")
            ),
            "final_val_loss": (
                float(model_info["history"].history["val_loss"][-1])
                if model_info["history"].history.get("val_loss")
                else float("nan")
            ),
        }


@ray.remote
class WindowActor:
    """Ray actor for windowed-analysis work (one model per genomic window).

    The Locator's full genotype array is materialised once on the actor; each
    ``run_window`` call slices a different set of SNP indices and trains a
    model for that window using a shared train/holdout split.
    """

    def __init__(
        self,
        gpu_id: int,
        gpu_fraction: float,
        data_ref,
        gpu_mem_mb: int = DEFAULT_GPU_MEM_MB,
    ):
        self._tf = _init_actor_runtime(gpu_id, gpu_fraction, gpu_mem_mb)
        self._data: Dict[str, Any] = data_ref
        self._locator = _create_worker_locator(self._data, "actor")
        self._base_out = self._data["config"]["out"]

        # Materialise the full genotypes once on the actor; window calls only
        # slice ``window_snp_indices`` off it. Continuous-dosage (GL) input is a
        # 2D float matrix and is kept as-is; hard calls become a GenotypeArray.
        from locator.data.filters import is_dosage_matrix

        arr = self._data["genotypes_array"]
        if is_dosage_matrix(arr):
            self._genotypes = arr
        else:
            import allel

            self._genotypes = allel.GenotypeArray(arr)

        loc = self._locator
        loc.genotypes = self._genotypes
        loc.index_set = self._data["index_set"]
        loc.meanlong = self._data["meanlong"]
        loc.sdlong = self._data["sdlong"]
        loc.meanlat = self._data["meanlat"]
        loc.sdlat = self._data["sdlat"]
        loc.unnormedlocs = self._data["unnormedlocs"]

    def ready(self) -> bool:
        return True

    def run_window(
        self,
        window_idx: int,
        window_label: str,
        window_chromosome,
        window_start: int,
        window_stop: int,
        snp_indices: List[int],
    ) -> Dict[str, Any]:
        if len(snp_indices) == 0:
            return {
                "window_idx": window_idx,
                "window_label": window_label,
                "window_chromosome": window_chromosome,
                "window_start": window_start,
                "window_stop": window_stop,
                "rows": None,
                "n_snps": 0,
            }

        loc = self._locator
        loc.config["out"] = f"{self._base_out}_win{window_idx}"
        loc.train_window(
            genotypes=self._genotypes,
            samples=self._data["samples"],
            window_snp_indices=snp_indices,
            index_set=self._data["index_set"],
            normalized_locs=self._data["normalized_locs"],
        )
        preds = loc.predict_holdout(
            verbose=False,
            return_df=True,
            save_preds_to_disk=False,
            plot_summary=False,
            plot_map=False,
        )
        return {
            "window_idx": window_idx,
            "window_label": window_label,
            "window_chromosome": window_chromosome,
            "window_start": window_start,
            "window_stop": window_stop,
            "rows": (
                preds[["sampleID", "x_pred", "y_pred"]].to_dict("records")
                if preds is not None
                else None
            ),
            "n_snps": len(snp_indices),
        }


def _spawn(actor_cls, gpu_ids, gpu_fraction, data_ref, gpu_mem_mb):
    """Spawn ``actor_cls`` instances pinned round-robin across ``gpu_ids``.

    Count of actors = ``len(gpu_ids) * round(1 / gpu_fraction)``, or 1 if
    ``gpu_ids`` is empty (CPU only).
    """
    if not gpu_ids:
        gpu_assignments = [-1]
        effective_fraction = 0.0
    else:
        workers_per_gpu = max(1, int(round(1.0 / gpu_fraction)))
        # Quantise the per-actor reservation so workers_per_gpu of them fit
        # *exactly* in one GPU. Passing the caller's gpu_fraction verbatim
        # (e.g. 0.34 × 3 = 1.02 per GPU) over-subscribes Ray's resource
        # accounting and leaves the last actor unschedulable, hanging the
        # pool's ``ready()`` barrier indefinitely. Actors are persistent
        # (unlike tasks), so unschedulable actors never get a turn.
        effective_fraction = 1.0 / workers_per_gpu
        gpu_assignments = [
            gpu_ids[i % len(gpu_ids)] for i in range(len(gpu_ids) * workers_per_gpu)
        ]

    actors = [
        actor_cls.options(num_gpus=effective_fraction).remote(
            gpu_id=gpu_id,
            gpu_fraction=effective_fraction,
            data_ref=data_ref,
            gpu_mem_mb=gpu_mem_mb,
        )
        for gpu_id in gpu_assignments
    ]
    # Block until every actor has finished __init__; surfaces config errors
    # (bad GPU id, memory cap rejection) before we dispatch work.
    ray.get([a.ready.remote() for a in actors])
    return actors


def make_fold_workers(gpu_ids, gpu_fraction, data_ref, gpu_mem_mb=DEFAULT_GPU_MEM_MB):
    """Spawn a pool of ``FoldWorker`` actors."""
    return _spawn(FoldWorker, gpu_ids, gpu_fraction, data_ref, gpu_mem_mb)


def make_ensemble_actors(gpu_ids, gpu_fraction, data_ref, gpu_mem_mb=DEFAULT_GPU_MEM_MB):
    """Spawn a pool of ``EnsembleActor`` actors."""
    return _spawn(EnsembleActor, gpu_ids, gpu_fraction, data_ref, gpu_mem_mb)


def make_window_actors(gpu_ids, gpu_fraction, data_ref, gpu_mem_mb=DEFAULT_GPU_MEM_MB):
    """Spawn a pool of ``WindowActor`` actors."""
    return _spawn(WindowActor, gpu_ids, gpu_fraction, data_ref, gpu_mem_mb)
