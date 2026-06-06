"""Tests for index-based TensorFlow dataset creation and the genotype table."""

import numpy as np
import pytest
import tensorflow as tf

from locator.data import (
    IndexSet,
    build_genotype_table,
    flip_genotypes_tf,
    make_tf_dataset,
    make_tf_dataset_from_arrays,
)
from locator.models import IndexedGenotypeModel, create_network


class TestMakeTFDataset:
    """Test the index-based make_tf_dataset function."""

    def setup_method(self):
        """Create test data."""
        np.random.seed(42)
        self.n_samples = 50
        self.coordinates = np.random.randn(self.n_samples, 2).astype(np.float32)

        # Create IndexSet with train/val/test splits
        self.index_set = IndexSet.random_split(
            n=self.n_samples, splits={"train": 0.6, "val": 0.2, "test": 0.2}, seed=42
        )

    def test_basic_dataset_creation(self):
        """Dataset yields (sample_index, coordinate) batches."""
        dataset = make_tf_dataset(
            coordinates=self.coordinates,
            index_set=self.index_set,
            split="train",
            batch_size=10,
            training=True,
            prefetch=False,
        )

        batch_count = -1
        for batch_count, (idx, coord) in enumerate(dataset):
            assert idx.shape == (10,)
            assert coord.shape == (10, 2)
            assert idx.dtype == tf.int32
            assert coord.dtype == tf.float32
            # Coordinates must stay aligned to their sample index.
            np.testing.assert_array_equal(coord.numpy(), self.coordinates[idx.numpy()])

        # 30 training samples / batch 10, drop_remainder defaults to training.
        assert batch_count == 2

    def test_repeat_yields_full_unique_batches(self):
        """repeat=True with batch==split size yields infinite, dup-free batches.

        This is the contract the fast fold-fit path relies on: it caps the batch
        to the split size so a split smaller than the default batch is not padded
        with duplicate samples under repeat().
        """
        n_train = len(self.index_set.train)
        dataset = make_tf_dataset(
            coordinates=self.coordinates,
            index_set=self.index_set,
            split="train",
            batch_size=n_train,
            training=True,
            prefetch=False,
            drop_remainder=True,
            repeat=True,
        )
        it = iter(dataset)
        for _ in range(3):  # repeats indefinitely; each batch covers the split once
            idx = next(it)[0].numpy()
            assert idx.shape == (n_train,)
            assert len(set(idx.tolist())) == n_train  # no duplicates within a batch

    def test_dataset_with_sample_weights(self):
        """Dataset yields (index, coordinate, weight) when weights are given."""
        train_size = len(self.index_set.train)
        sample_weights = np.random.rand(train_size).astype(np.float32)

        dataset = make_tf_dataset(
            coordinates=self.coordinates,
            index_set=self.index_set,
            split="train",
            batch_size=10,
            sample_weights=sample_weights,
            training=True,
            prefetch=False,
        )

        for batch in dataset.take(1):
            assert len(batch) == 3
            idx, coord, weights = batch
            assert idx.shape == (10,)
            assert coord.shape == (10, 2)
            assert weights.shape == (10,)
            assert weights.dtype == tf.float32

    def test_validation_split_keeps_partial_batch(self):
        """Non-training splits keep the final partial batch."""
        dataset = make_tf_dataset(
            coordinates=self.coordinates,
            index_set=self.index_set,
            split="val",
            batch_size=4,
            training=False,
            prefetch=False,
        )

        n_val = len(self.index_set.get_split("val"))
        total = sum(int(idx.shape[0]) for idx, _ in dataset)
        assert total == n_val

    def test_invalid_split_raises_error(self):
        """An unknown split name raises KeyError."""
        with pytest.raises(KeyError):
            make_tf_dataset(
                coordinates=self.coordinates,
                index_set=self.index_set,
                split="nonexistent",
                batch_size=10,
            )

    def test_mismatched_weights_raises_error(self):
        """A weight array that does not match the split size raises ValueError."""
        wrong_weights = np.random.rand(100)

        with pytest.raises(ValueError, match="Sample weights length"):
            make_tf_dataset(
                coordinates=self.coordinates,
                index_set=self.index_set,
                split="train",
                batch_size=10,
                sample_weights=wrong_weights,
            )


class TestBuildGenotypeTable:
    """Test the GPU-resident genotype table builder."""

    def test_sample_major_layout(self):
        """The table is the sample-major transpose of the input."""
        geno = np.random.randint(0, 3, size=(60, 15)).astype(np.int8)
        table = build_genotype_table(geno)

        assert table.shape == (15, 60)  # (n_samples, n_snps)
        np.testing.assert_array_equal(table.numpy(), geno.T)

    def test_preserves_int8_dtype(self):
        """int8 hard-call inputs stay int8."""
        geno = np.random.randint(0, 3, size=(30, 8)).astype(np.int8)
        assert build_genotype_table(geno).dtype == tf.int8

    def test_preserves_float32_dtype(self):
        """float32 dosage inputs stay float32."""
        geno = np.random.rand(30, 8).astype(np.float32) * 2.0
        assert build_genotype_table(geno).dtype == tf.float32


class TestIndexedGenotypeModel:
    """Test the on-device gather wrapper model."""

    def test_batched_gather_matches_manual_gather(self):
        """wrapper(idx) equals inner(gather(table, idx)) for the same network."""
        np.random.seed(0)
        n_snps, n_samples = 40, 20
        geno = np.random.randint(0, 3, size=(n_snps, n_samples)).astype(np.int8)

        inner = create_network(input_shape=n_snps, width=8, n_layers=2)
        table = build_genotype_table(geno)
        wrapper = IndexedGenotypeModel(inner, table)

        idx = tf.constant([5, 0, 12, 19, 3], dtype=tf.int32)
        out_wrapper = wrapper(idx, training=False).numpy()

        manual_features = tf.cast(tf.gather(geno.T, idx), tf.float32)
        out_manual = inner(manual_features, training=False).numpy()

        np.testing.assert_allclose(out_wrapper, out_manual, rtol=1e-5, atol=1e-5)

    def test_site_order_column_gather(self):
        """site_order resamples SNP columns after the per-sample row gather."""
        np.random.seed(1)
        n_snps, n_samples = 40, 20
        geno = np.random.randint(0, 3, size=(n_snps, n_samples)).astype(np.int8)
        site_order = np.random.choice(n_snps, n_snps, replace=True)

        inner = create_network(input_shape=n_snps, width=8, n_layers=2)
        table = build_genotype_table(geno)
        wrapper = IndexedGenotypeModel(inner, table, site_order=site_order)

        idx = tf.constant([2, 7, 11], dtype=tf.int32)
        out_wrapper = wrapper(idx, training=False).numpy()

        g = tf.gather(geno.T, idx)
        g = tf.gather(g, site_order, axis=1)
        out_manual = inner(tf.cast(g, tf.float32), training=False).numpy()

        np.testing.assert_allclose(out_wrapper, out_manual, rtol=1e-5, atol=1e-5)

    def test_save_weights_delegates_to_inner(self, tmp_path):
        """Weights round-trip through inner so the on-disk format is unchanged."""
        n_snps = 30
        inner = create_network(input_shape=n_snps, width=8, n_layers=2)
        geno = np.random.randint(0, 3, size=(n_snps, 10)).astype(np.int8)
        wrapper = IndexedGenotypeModel(inner, build_genotype_table(geno))

        path = str(tmp_path / "model.weights.h5")
        wrapper.save_weights(path)

        # A bare network of the same architecture can load the file.
        reloaded = create_network(input_shape=n_snps, width=8, n_layers=2)
        reloaded.load_weights(path)
        for w_a, w_b in zip(inner.get_weights(), reloaded.get_weights(), strict=True):
            np.testing.assert_array_equal(w_a, w_b)


class TestFlipGenotypesTF:
    """Test the genotype flipping augmentation function."""

    def test_flip_genotypes_basic(self):
        """Test basic genotype flipping."""
        genotypes = tf.constant([0.0, 1.0, 0.0, 1.0, 2.0], dtype=tf.float32)
        tf.random.set_seed(42)

        flipped = flip_genotypes_tf(genotypes, flip_rate=0.8)

        # 2s (missing) are never flipped
        original_2s = tf.where(genotypes == 2.0)
        flipped_2s = tf.gather(flipped, original_2s)
        assert tf.reduce_all(flipped_2s == 2.0)

        assert flipped.shape == genotypes.shape

    def test_flip_preserves_missing_values(self):
        """Missing values (2) are never flipped."""
        genotypes = tf.constant([[0.0, 1.0, 2.0], [2.0, 0.0, 1.0]], dtype=tf.float32)

        for _ in range(10):
            flipped = flip_genotypes_tf(genotypes, flip_rate=0.5)
            assert tf.reduce_all(tf.where(genotypes == 2.0, flipped == 2.0, True))


class TestMakeTFDatasetFromArrays:
    """Test the legacy feature-based compatibility function."""

    def test_single_dataset_creation(self):
        """Test creating a single training dataset."""
        train_gen = np.random.rand(30, 100).astype(np.float32)
        train_locs = np.random.randn(30, 2).astype(np.float32)

        dataset = make_tf_dataset_from_arrays(
            train_gen=train_gen,
            train_locs=train_locs,
            batch_size=10,
            cache=False,
            prefetch=False,
        )

        assert isinstance(dataset, tf.data.Dataset)

        for batch in dataset.take(1):
            features, labels = batch
            assert features.shape == (10, 100)
            assert labels.shape == (10, 2)

    def test_multiple_datasets_creation(self):
        """Test creating train/test/val datasets."""
        train_gen = np.random.rand(30, 100).astype(np.float32)
        train_locs = np.random.randn(30, 2).astype(np.float32)
        test_gen = np.random.rand(10, 100).astype(np.float32)
        test_locs = np.random.randn(10, 2).astype(np.float32)
        val_gen = np.random.rand(10, 100).astype(np.float32)
        val_locs = np.random.randn(10, 2).astype(np.float32)

        train_ds, test_ds, val_ds = make_tf_dataset_from_arrays(
            train_gen=train_gen,
            train_locs=train_locs,
            test_gen=test_gen,
            test_locs=test_locs,
            val_gen=val_gen,
            val_locs=val_locs,
            batch_size=5,
            cache=False,
            prefetch=False,
        )

        for ds in [train_ds, test_ds, val_ds]:
            assert isinstance(ds, tf.data.Dataset)

        # Training drops the partial final batch; validation/test keep it.
        assert sum(1 for _ in train_ds) == 6  # 30 / 5
        assert sum(1 for _ in test_ds) == 2  # 10 / 5
        assert sum(1 for _ in val_ds) == 2  # 10 / 5
