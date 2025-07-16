"""
Parallel analysis methods for multi-GPU execution.
"""

try:
    from .parallel_analysis import (
        parallel_holdouts,
        parallel_k_fold_holdouts,
        parallel_leave_one_out,
        parallel_train_ensemble,
        parallel_windows_holdouts,
    )

    __all__ = [
        "parallel_k_fold_holdouts",
        "parallel_leave_one_out",
        "parallel_holdouts",
        "parallel_train_ensemble",
        "parallel_windows_holdouts",
    ]
except ImportError:
    # Ray not installed - likely during docs build
    # Define stub functions to allow documentation to build
    def _not_available(*args, **kwargs):
        raise ImportError(
            "Ray is required for parallel analysis methods. "
            "Install with: pip install locator[ray]"
        )

    parallel_k_fold_holdouts = _not_available
    parallel_leave_one_out = _not_available
    parallel_holdouts = _not_available
    parallel_train_ensemble = _not_available
    parallel_windows_holdouts = _not_available

    __all__ = [
        "parallel_k_fold_holdouts",
        "parallel_leave_one_out",
        "parallel_holdouts",
        "parallel_train_ensemble",
        "parallel_windows_holdouts",
    ]
