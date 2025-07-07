"""IndexSet for memory-efficient data splitting without copying arrays."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import numpy as np


@dataclass(frozen=True)
class IndexSet:
    """Container for dataset indices that avoids copying data.

    This class stores indices for different data splits (train/val/test)
    to enable memory-efficient data access without creating copies of
    large genotype arrays.

    Attributes:
        indices: Dictionary mapping split names to numpy arrays of indices
        total_samples: Total number of samples in the dataset
        na_mask: Optional boolean mask indicating samples without coordinates
    """

    indices: Dict[str, np.ndarray]
    total_samples: int
    na_mask: Optional[np.ndarray] = None

    def __post_init__(self):
        """Validate the IndexSet after initialization."""
        # Verify no overlapping indices
        all_indices = []
        for split_indices in self.indices.values():
            all_indices.extend(split_indices.tolist())

        if len(all_indices) != len(set(all_indices)):
            raise ValueError("IndexSet contains overlapping indices between splits")

        # Verify indices are within bounds
        max_idx = max(all_indices) if all_indices else -1
        if max_idx >= self.total_samples:
            raise ValueError(
                f"Index {max_idx} exceeds total_samples {self.total_samples}"
            )

    @property
    def train(self) -> np.ndarray:
        """Get training indices (backward compatibility)."""
        return self.indices.get("train", np.array([], dtype=int))

    @property
    def val(self) -> np.ndarray:
        """Get validation indices (backward compatibility)."""
        return self.indices.get("val", np.array([], dtype=int))

    @property
    def test(self) -> np.ndarray:
        """Get test indices (backward compatibility)."""
        return self.indices.get("test", np.array([], dtype=int))

    @property
    def hold(self) -> np.ndarray:
        """Get holdout/prediction indices (backward compatibility)."""
        # Try 'hold' first, then 'test' for compatibility
        return self.indices.get(
            "hold", self.indices.get("test", np.array([], dtype=int))
        )

    def get_split(self, name: str) -> np.ndarray:
        """Get indices for a named split."""
        if name not in self.indices:
            raise KeyError(
                f"Split '{name}' not found. Available splits: {list(self.indices.keys())}"
            )
        return self.indices[name]

    def split_sizes(self) -> Dict[str, int]:
        """Get the size of each split."""
        return {name: len(indices) for name, indices in self.indices.items()}

    @classmethod
    def random_split(
        cls,
        n: int,
        splits: Optional[Dict[str, float]] = None,
        seed: Optional[int] = None,
        na_mask: Optional[np.ndarray] = None,
        na_action: str = "separate",
    ) -> IndexSet:
        """Create random train/val/test splits.

        Args:
            n: Total number of samples
            splits: Dictionary mapping split names to proportions (must sum to ≤ 1.0)
                   Default: {"train": 0.8, "val": 0.1, "test": 0.1}
            seed: Random seed for reproducibility
            na_mask: Boolean mask indicating samples without coordinates
            na_action: How to handle NA samples ('separate', 'exclude', 'fail')

        Returns:
            IndexSet with random splits
        """
        if splits is None:
            splits = {"train": 0.8, "val": 0.1, "test": 0.1}

        # Validate splits
        total_prop = sum(splits.values())
        if total_prop > 1.0 + 1e-10:
            raise ValueError(f"Split proportions sum to {total_prop}, must be ≤ 1.0")

        # Handle NA samples
        if na_mask is not None:
            if na_action == "fail" and np.any(na_mask):
                raise ValueError(
                    "Samples without coordinates found but na_action='fail'"
                )
            elif na_action == "exclude" or na_action == "separate":
                # Only use samples with coordinates for train/val/test
                valid_indices = np.where(~na_mask)[0]
                n_valid = len(valid_indices)
                if n_valid == 0:
                    raise ValueError("No samples with valid coordinates")
            else:
                valid_indices = np.arange(n)
                n_valid = n
        else:
            valid_indices = np.arange(n)
            n_valid = n

        # Set random seed
        rng = np.random.RandomState(seed)

        # Shuffle indices
        shuffled = valid_indices.copy()
        rng.shuffle(shuffled)

        # Create splits
        indices = {}
        start = 0
        for i, (name, prop) in enumerate(splits.items()):
            if i == len(splits) - 1:
                # Last split gets remaining samples
                indices[name] = shuffled[start:]
            else:
                size = int(np.round(prop * n_valid))
                indices[name] = shuffled[start : start + size]
                start += size

        # Handle NA samples in 'separate' mode
        if na_mask is not None and na_action == "separate" and np.any(na_mask):
            # Add NA samples as a separate 'predict' split
            indices["predict"] = np.where(na_mask)[0]

        return cls(indices=indices, total_samples=n, na_mask=na_mask)

    @classmethod
    def from_k_fold(
        cls,
        n: int,
        k: int,
        fold: int,
        seed: Optional[int] = None,
        na_mask: Optional[np.ndarray] = None,
    ) -> IndexSet:
        """Create train/test split for k-fold cross-validation.

        Args:
            n: Total number of samples
            k: Number of folds
            fold: Which fold to use as test set (0-indexed)
            seed: Random seed for reproducibility
            na_mask: Boolean mask indicating samples without coordinates

        Returns:
            IndexSet with train and test splits
        """
        if fold >= k or fold < 0:
            raise ValueError(f"Fold {fold} out of range for {k}-fold CV")

        # Handle NA samples - k-fold requires known coordinates
        if na_mask is not None and np.any(na_mask):
            valid_indices = np.where(~na_mask)[0]
            n_valid = len(valid_indices)
        else:
            valid_indices = np.arange(n)
            n_valid = n

        # Shuffle indices
        rng = np.random.RandomState(seed)
        shuffled = valid_indices.copy()
        rng.shuffle(shuffled)

        # Create folds
        fold_size = n_valid // k
        test_start = fold * fold_size
        test_end = test_start + fold_size if fold < k - 1 else n_valid

        test_indices = shuffled[test_start:test_end]
        train_indices = np.concatenate([shuffled[:test_start], shuffled[test_end:]])

        return cls(
            indices={"train": train_indices, "test": test_indices},
            total_samples=n,
            na_mask=na_mask,
        )

    @classmethod
    def from_groups(
        cls,
        groups: np.ndarray,
        test_groups: List[Union[int, str]],
        na_mask: Optional[np.ndarray] = None,
    ) -> IndexSet:
        """Create train/test split based on group membership.

        Useful for spatial or temporal cross-validation where you want
        to hold out entire groups (e.g., geographic regions).

        Args:
            groups: Array of group labels for each sample
            test_groups: List of group labels to use as test set
            na_mask: Boolean mask indicating samples without coordinates

        Returns:
            IndexSet with train and test splits
        """
        n = len(groups)
        test_mask = np.isin(groups, test_groups)

        # Handle NA samples
        if na_mask is not None:
            # Exclude NA samples from both train and test
            test_indices = np.where(test_mask & ~na_mask)[0]
            train_indices = np.where(~test_mask & ~na_mask)[0]
        else:
            test_indices = np.where(test_mask)[0]
            train_indices = np.where(~test_mask)[0]

        return cls(
            indices={"train": train_indices, "test": test_indices},
            total_samples=n,
            na_mask=na_mask,
        )

    @classmethod
    def from_manual(
        cls,
        train: np.ndarray,
        test: Optional[np.ndarray] = None,
        val: Optional[np.ndarray] = None,
        predict: Optional[np.ndarray] = None,
        total_samples: Optional[int] = None,
    ) -> IndexSet:
        """Create IndexSet from manually specified indices.

        Args:
            train: Training indices
            test: Test indices
            val: Validation indices
            predict: Prediction indices (samples without labels)
            total_samples: Total number of samples (inferred if not provided)

        Returns:
            IndexSet with specified splits
        """
        indices = {"train": train}

        if test is not None:
            indices["test"] = test
        if val is not None:
            indices["val"] = val
        if predict is not None:
            indices["predict"] = predict

        # Infer total samples if not provided
        if total_samples is None:
            all_indices = []
            for split_indices in indices.values():
                all_indices.extend(split_indices.tolist())
            total_samples = max(all_indices) + 1 if all_indices else 0

        return cls(indices=indices, total_samples=total_samples)
