"""
Parallel analysis methods for multi-GPU execution.
"""

try:
    from .ensemble import parallel_train_ensemble
    from .holdout import parallel_holdouts
    from .kfold import parallel_k_fold_holdouts, parallel_leave_one_out
    from .windowed import (
        parallel_windows_holdouts,
        parallel_windows_leave_one_out,
    )

    __all__ = [
        "parallel_k_fold_holdouts",
        "parallel_leave_one_out",
        "parallel_holdouts",
        "parallel_train_ensemble",
        "parallel_windows_holdouts",
        "parallel_windows_leave_one_out",
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
    parallel_windows_leave_one_out = _not_available

    __all__ = [
        "parallel_k_fold_holdouts",
        "parallel_leave_one_out",
        "parallel_holdouts",
        "parallel_train_ensemble",
        "parallel_windows_holdouts",
        "parallel_windows_leave_one_out",
    ]
