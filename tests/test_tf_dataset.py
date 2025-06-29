"""Tests for unified TensorFlow dataset creation."""

import numpy as np
import tensorflow as tf
import pytest
from locator.data import IndexSet, make_tf_dataset, make_tf_dataset_from_arrays, flip_genotypes_tf


class TestMakeTFDataset:
    """Test the main make_tf_dataset function."""
    
    def setup_method(self):
        """Create test data."""
        np.random.seed(42)
        self.n_snps = 100
        self.n_samples = 50
        self.genotypes = np.random.randint(0, 3, size=(self.n_snps, self.n_samples)).astype(np.float32)
        self.coordinates = np.random.randn(self.n_samples, 2).astype(np.float32)
        
        # Create IndexSet with train/val/test splits
        self.index_set = IndexSet.random_split(
            n=self.n_samples,
            splits={"train": 0.6, "val": 0.2, "test": 0.2},
            seed=42
        )
    
    def test_basic_dataset_creation(self):
        """Test basic dataset creation without weights or augmentation."""
        dataset = make_tf_dataset(
            genotypes=self.genotypes,
            coordinates=self.coordinates,
            index_set=self.index_set,
            split="train",
            batch_size=10,
            training=True,
            cache=False,  # Disable caching for testing
            prefetch=False
        )
        
        # Check that we can iterate through the dataset
        batch_count = 0
        for batch in dataset:
            features, labels = batch
            batch_count += 1
            
            # Check shapes
            assert features.shape == (10, self.n_snps)  # (batch_size, n_features)
            assert labels.shape == (10, 2)  # (batch_size, 2)
            
            # Check dtypes
            assert features.dtype == tf.float32
            assert labels.dtype == tf.float32
        
        # We should have 3 batches (30 samples / 10 batch_size)
        assert batch_count == 3
    
    def test_dataset_with_sample_weights(self):
        """Test dataset creation with sample weights."""
        # Create sample weights
        train_size = len(self.index_set.train)
        sample_weights = np.random.rand(train_size).astype(np.float32)
        
        dataset = make_tf_dataset(
            genotypes=self.genotypes,
            coordinates=self.coordinates,
            index_set=self.index_set,
            split="train",
            batch_size=10,
            sample_weights=sample_weights,
            training=True,
            cache=False,
            prefetch=False
        )
        
        # Check that dataset returns 3 elements
        for batch in dataset.take(1):
            assert len(batch) == 3
            features, labels, weights = batch
            
            assert features.shape == (10, self.n_snps)
            assert labels.shape == (10, 2)
            assert weights.shape == (10,)
            assert weights.dtype == tf.float32
    
    def test_dataset_with_augmentation(self):
        """Test dataset with augmentation enabled."""
        augment_config = {"enabled": True, "flip_rate": 0.1}
        
        dataset = make_tf_dataset(
            genotypes=self.genotypes,
            coordinates=self.coordinates,
            index_set=self.index_set,
            split="train",
            batch_size=10,
            augment=augment_config,
            training=True,
            cache=False,
            prefetch=False
        )
        
        # Just verify we can iterate - augmentation is stochastic
        for batch in dataset.take(1):
            features, labels = batch
            assert features.shape == (10, self.n_snps)
            assert labels.shape == (10, 2)
    
    def test_dataset_with_site_order(self):
        """Test bootstrap resampling with site_order."""
        # Create random site order for bootstrap
        site_order = np.random.choice(self.n_snps, self.n_snps, replace=True)
        
        dataset = make_tf_dataset(
            genotypes=self.genotypes,
            coordinates=self.coordinates,
            index_set=self.index_set,
            split="train",
            batch_size=10,
            site_order=site_order,
            training=False,
            cache=False,
            prefetch=False
        )
        
        # Verify shapes are correct
        for batch in dataset.take(1):
            features, labels = batch
            assert features.shape[1] == self.n_snps  # Same number of SNPs
    
    def test_invalid_split_raises_error(self):
        """Test that invalid split name raises error."""
        with pytest.raises(KeyError):
            make_tf_dataset(
                genotypes=self.genotypes,
                coordinates=self.coordinates,
                index_set=self.index_set,
                split="nonexistent",
                batch_size=10
            )
    
    def test_mismatched_weights_raises_error(self):
        """Test that mismatched weight length raises error."""
        wrong_weights = np.random.rand(100)  # Wrong size
        
        with pytest.raises(ValueError, match="Sample weights length"):
            make_tf_dataset(
                genotypes=self.genotypes,
                coordinates=self.coordinates,
                index_set=self.index_set,
                split="train",
                batch_size=10,
                sample_weights=wrong_weights
            )
    
    def test_mixed_precision_dtype(self):
        """Test dataset creation with float16 dtype."""
        dataset = make_tf_dataset(
            genotypes=self.genotypes,
            coordinates=self.coordinates,
            index_set=self.index_set,
            split="train",
            batch_size=10,
            dtype_policy="float16",
            training=False,
            cache=False,
            prefetch=False
        )
        
        for batch in dataset.take(1):
            features, labels = batch
            assert features.dtype == tf.float16
            assert labels.dtype == tf.float32  # Coords always float32


class TestFlipGenotypesTF:
    """Test the genotype flipping augmentation function."""
    
    def test_flip_genotypes_basic(self):
        """Test basic genotype flipping."""
        # Create test genotypes with known values
        genotypes = tf.constant([0.0, 1.0, 0.0, 1.0, 2.0], dtype=tf.float32)
        
        # Set seed for reproducibility
        tf.random.set_seed(42)
        
        # Apply flipping with high rate to ensure some flips
        flipped = flip_genotypes_tf(genotypes, flip_rate=0.8)
        
        # Check that 2s (missing) are never flipped
        original_2s = tf.where(genotypes == 2.0)
        flipped_2s = tf.gather(flipped, original_2s)
        assert tf.reduce_all(flipped_2s == 2.0)
        
        # Check shape is preserved
        assert flipped.shape == genotypes.shape
    
    def test_flip_preserves_missing_values(self):
        """Test that missing values (2) are never flipped."""
        genotypes = tf.constant([[0.0, 1.0, 2.0], [2.0, 0.0, 1.0]], dtype=tf.float32)
        
        # Apply flipping many times
        for _ in range(10):
            flipped = flip_genotypes_tf(genotypes, flip_rate=0.5)
            # Check all 2s remain 2s
            assert tf.reduce_all(tf.where(genotypes == 2.0, flipped == 2.0, True))


class TestMakeTFDatasetFromArrays:
    """Test the legacy compatibility function."""
    
    def test_single_dataset_creation(self):
        """Test creating a single training dataset."""
        train_gen = np.random.rand(30, 100).astype(np.float32)
        train_locs = np.random.randn(30, 2).astype(np.float32)
        
        dataset = make_tf_dataset_from_arrays(
            train_gen=train_gen,
            train_locs=train_locs,
            batch_size=10,
            cache=False,
            prefetch=False
        )
        
        # Should return single dataset
        assert isinstance(dataset, tf.data.Dataset)
        
        # Check shapes
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
            prefetch=False
        )
        
        # Check all three datasets
        for ds in [train_ds, test_ds, val_ds]:
            assert isinstance(ds, tf.data.Dataset)
            
        # Verify different behaviors
        # Training should have drop_remainder=True by default
        train_batch_count = sum(1 for _ in train_ds)
        assert train_batch_count == 6  # 30 / 5
        
        # Test/val should have drop_remainder=False by default
        test_batch_count = sum(1 for _ in test_ds)
        assert test_batch_count == 2  # 10 / 5
        
        val_batch_count = sum(1 for _ in val_ds)
        assert val_batch_count == 2  # 10 / 5