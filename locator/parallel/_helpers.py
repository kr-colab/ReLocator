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
