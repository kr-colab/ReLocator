"""Tests for IndexSet data structure."""

import numpy as np
import pytest

from locator.data import IndexSet


class TestIndexSetBasic:
    """Test basic IndexSet functionality."""

    def test_create_indexset(self):
        """Test creating a basic IndexSet."""
        indices = {
            "train": np.array([0, 1, 2, 3, 4]),
            "val": np.array([5, 6]),
            "test": np.array([7, 8, 9]),
        }
        idx_set = IndexSet(indices=indices, total_samples=10)

        assert len(idx_set.train) == 5
        assert len(idx_set.val) == 2
        assert len(idx_set.test) == 3
        assert idx_set.total_samples == 10

    def test_backward_compatibility(self):
        """Test backward compatibility properties."""
        indices = {"train": np.array([0, 1, 2]), "test": np.array([3, 4])}
        idx_set = IndexSet(indices=indices, total_samples=5)

        # Should return empty array for missing splits
        assert len(idx_set.val) == 0
        assert isinstance(idx_set.val, np.ndarray)

        # hold should alias to test
        assert np.array_equal(idx_set.hold, idx_set.test)

    def test_validation_overlapping_indices(self):
        """Test that overlapping indices raise an error."""
        indices = {
            "train": np.array([0, 1, 2, 3]),
            "test": np.array([3, 4, 5]),  # 3 overlaps with train
        }

        with pytest.raises(ValueError, match="overlapping indices"):
            IndexSet(indices=indices, total_samples=6)

    def test_validation_out_of_bounds(self):
        """Test that out-of-bounds indices raise an error."""
        indices = {
            "train": np.array([0, 1, 2]),
            "test": np.array([3, 4, 10]),  # 10 exceeds total_samples
        }

        with pytest.raises(ValueError, match="exceeds total_samples"):
            IndexSet(indices=indices, total_samples=5)

    def test_get_split(self):
        """Test getting named splits."""
        indices = {"train": np.array([0, 1]), "custom": np.array([2, 3])}
        idx_set = IndexSet(indices=indices, total_samples=4)

        assert np.array_equal(idx_set.get_split("train"), np.array([0, 1]))
        assert np.array_equal(idx_set.get_split("custom"), np.array([2, 3]))

        with pytest.raises(KeyError):
            idx_set.get_split("nonexistent")

    def test_split_sizes(self):
        """Test getting split sizes."""
        indices = {
            "train": np.array([0, 1, 2, 3]),
            "val": np.array([4, 5]),
            "test": np.array([6, 7, 8]),
        }
        idx_set = IndexSet(indices=indices, total_samples=9)

        sizes = idx_set.split_sizes()
        assert sizes == {"train": 4, "val": 2, "test": 3}


class TestIndexSetRandomSplit:
    """Test random splitting functionality."""

    def test_random_split_default(self):
        """Test default 80/10/10 split."""
        idx_set = IndexSet.random_split(n=100, seed=42)

        assert idx_set.total_samples == 100
        assert len(idx_set.train) == 80
        assert len(idx_set.val) == 10
        assert len(idx_set.test) == 10

        # Check no overlap
        all_indices = np.concatenate([idx_set.train, idx_set.val, idx_set.test])
        assert len(np.unique(all_indices)) == 100

    def test_random_split_custom(self):
        """Test custom split proportions."""
        splits = {"train": 0.7, "val": 0.15, "test": 0.15}
        idx_set = IndexSet.random_split(n=100, splits=splits, seed=42)

        assert len(idx_set.train) == 70
        assert len(idx_set.val) == 15
        assert len(idx_set.test) == 15

    def test_random_split_validation(self):
        """Test split proportion validation."""
        # Proportions > 1.0 should fail
        with pytest.raises(ValueError, match="must be ≤ 1.0"):
            IndexSet.random_split(n=100, splits={"train": 0.8, "test": 0.3})

    def test_random_split_reproducibility(self):
        """Test that same seed gives same split."""
        idx1 = IndexSet.random_split(n=50, seed=123)
        idx2 = IndexSet.random_split(n=50, seed=123)

        assert np.array_equal(idx1.train, idx2.train)
        assert np.array_equal(idx1.val, idx2.val)
        assert np.array_equal(idx1.test, idx2.test)

    def test_random_split_with_na_separate(self):
        """Test random split with NA handling in separate mode."""
        na_mask = np.array(
            [False, False, True, False, True, False, False, True, False, False]
        )
        idx_set = IndexSet.random_split(
            n=10, seed=42, na_mask=na_mask, na_action="separate"
        )

        # Should have 7 samples with coordinates split among train/val/test
        total_with_coords = len(idx_set.train) + len(idx_set.val) + len(idx_set.test)
        assert total_with_coords == 7

        # Should have predict split with 3 NA samples
        assert len(idx_set.get_split("predict")) == 3
        assert np.array_equal(idx_set.get_split("predict"), np.array([2, 4, 7]))

    def test_random_split_with_na_exclude(self):
        """Test random split with NA handling in exclude mode."""
        na_mask = np.array([False, False, True, False, True])
        idx_set = IndexSet.random_split(
            n=5, seed=42, na_mask=na_mask, na_action="exclude"
        )

        # Should only include samples with coordinates
        all_indices = np.concatenate([idx_set.train, idx_set.val, idx_set.test])
        assert len(all_indices) == 3
        assert not np.any(na_mask[all_indices])

    def test_random_split_with_na_fail(self):
        """Test random split with NA handling in fail mode."""
        na_mask = np.array([False, False, True, False, False])

        with pytest.raises(ValueError, match="Samples without coordinates found"):
            IndexSet.random_split(n=5, na_mask=na_mask, na_action="fail")


class TestIndexSetKFold:
    """Test k-fold cross-validation functionality."""

    def test_k_fold_basic(self):
        """Test basic k-fold splitting."""
        for fold in range(5):
            idx_set = IndexSet.from_k_fold(n=100, k=5, fold=fold, seed=42)

            assert len(idx_set.test) == 20
            assert len(idx_set.train) == 80

            # Check no overlap
            assert len(np.intersect1d(idx_set.train, idx_set.test)) == 0

    def test_k_fold_coverage(self):
        """Test that k-fold covers all samples."""
        all_test_indices = []

        for fold in range(5):
            idx_set = IndexSet.from_k_fold(n=25, k=5, fold=fold, seed=42)
            all_test_indices.extend(idx_set.test.tolist())

        # All samples should appear exactly once in test sets
        assert sorted(all_test_indices) == list(range(25))

    def test_k_fold_validation(self):
        """Test k-fold parameter validation."""
        with pytest.raises(ValueError, match="out of range"):
            IndexSet.from_k_fold(n=100, k=5, fold=5)  # fold should be 0-4

    def test_k_fold_with_na(self):
        """Test k-fold with NA samples."""
        na_mask = np.array(
            [False, False, True, False, True, False, False, True, False, False]
        )
        idx_set = IndexSet.from_k_fold(n=10, k=3, fold=0, seed=42, na_mask=na_mask)

        # Should only include samples with coordinates
        all_indices = np.concatenate([idx_set.train, idx_set.test])
        assert len(all_indices) == 7
        assert not np.any(na_mask[all_indices])


class TestIndexSetGroups:
    """Test group-based splitting functionality."""

    def test_groups_basic(self):
        """Test basic group-based splitting."""
        groups = np.array([1, 1, 2, 2, 3, 3, 4, 4])
        idx_set = IndexSet.from_groups(groups, test_groups=[2, 4])

        assert np.array_equal(idx_set.test, np.array([2, 3, 6, 7]))
        assert np.array_equal(idx_set.train, np.array([0, 1, 4, 5]))

    def test_groups_string_labels(self):
        """Test group splitting with string labels."""
        groups = np.array(["A", "A", "B", "B", "C", "C"])
        idx_set = IndexSet.from_groups(groups, test_groups=["B"])

        assert np.array_equal(idx_set.test, np.array([2, 3]))
        assert np.array_equal(idx_set.train, np.array([0, 1, 4, 5]))

    def test_groups_with_na(self):
        """Test group splitting with NA samples."""
        groups = np.array([1, 1, 2, 2, 3, 3])
        na_mask = np.array([False, True, False, True, False, False])
        idx_set = IndexSet.from_groups(groups, test_groups=[2], na_mask=na_mask)

        # Should exclude NA samples from both train and test
        assert np.array_equal(idx_set.test, np.array([2]))  # Only index 2, not 3
        assert np.array_equal(idx_set.train, np.array([0, 4, 5]))  # Excludes index 1


class TestIndexSetManual:
    """Test manual index specification."""

    def test_manual_basic(self):
        """Test basic manual index creation."""
        train = np.array([0, 1, 2])
        test = np.array([3, 4])
        val = np.array([5])

        idx_set = IndexSet.from_manual(train=train, test=test, val=val)

        assert np.array_equal(idx_set.train, train)
        assert np.array_equal(idx_set.test, test)
        assert np.array_equal(idx_set.val, val)
        assert idx_set.total_samples == 6

    def test_manual_infer_total(self):
        """Test inferring total samples from indices."""
        idx_set = IndexSet.from_manual(
            train=np.array([0, 5, 10]), test=np.array([15, 20])
        )

        assert idx_set.total_samples == 21  # max index + 1

    def test_manual_with_predict(self):
        """Test manual creation with predict split."""
        idx_set = IndexSet.from_manual(
            train=np.array([0, 1, 2]),
            test=np.array([3, 4]),
            predict=np.array([5, 6, 7]),
            total_samples=8,
        )

        assert len(idx_set.get_split("predict")) == 3
        assert np.array_equal(idx_set.get_split("predict"), np.array([5, 6, 7]))
