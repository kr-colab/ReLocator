"""
Parallel analysis methods for multi-GPU execution.
"""

from .parallel_analysis import (
    parallel_k_fold_holdouts,
    parallel_leave_one_out
)

__all__ = [
    'parallel_k_fold_holdouts',
    'parallel_leave_one_out'
]