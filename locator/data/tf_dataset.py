"""Unified TensorFlow dataset creation with memory-efficient data access."""

from __future__ import annotations
from typing import Optional, Dict, Tuple, Callable, Union
import numpy as np
import tensorflow as tf
from .indexset import IndexSet


def make_tf_dataset(
    genotypes: np.ndarray,
    coordinates: np.ndarray,
    index_set: IndexSet,
    split: str,
    batch_size: int = 256,
    shuffle_buffer: int = 1024,
    augment: Optional[Dict] = None,
    sample_weights: Optional[np.ndarray] = None,
    training: bool = True,
    cache: bool = True,
    prefetch: bool = True,
    drop_remainder: Optional[bool] = None,
    dtype_policy: Optional[str] = None,
    site_order: Optional[np.ndarray] = None,
) -> tf.data.Dataset:
    """Create an efficient tf.data pipeline that gathers rows on-the-fly.
    
    This function creates a memory-efficient TensorFlow dataset that uses
    tf.gather to access data by indices rather than copying arrays. It
    provides consistent handling of sample weights, augmentation, and
    optimization across all training scenarios.
    
    Args:
        genotypes: Full genotype array of shape (n_snps, n_samples)
        coordinates: Full coordinate array of shape (n_samples, 2)
        index_set: IndexSet containing train/val/test/predict indices
        split: Which split to use ('train', 'val', 'test', 'predict')
        batch_size: Batch size for the dataset
        shuffle_buffer: Buffer size for shuffling (only used if training=True)
        augment: Optional augmentation config dict with 'enabled' and 'flip_rate'
        sample_weights: Optional array of sample weights (must match split size)
        training: Whether this is for training (enables shuffling)
        cache: Whether to cache data in memory
        prefetch: Whether to use prefetching
        drop_remainder: Whether to drop incomplete batches (defaults to value of training)
        dtype_policy: Optional dtype policy ('float32', 'float16', 'mixed_float16')
        site_order: Optional array of SNP indices for bootstrap resampling
        
    Returns:
        tf.data.Dataset with structure:
            - Without weights: (features, labels)
            - With weights: (features, labels, sample_weights)
    """
    # Get indices for the requested split
    indices = index_set.get_split(split)
    
    if len(indices) == 0:
        raise ValueError(f"Split '{split}' has no samples")
    
    # Determine data type based on policy
    if dtype_policy is None:
        policy = tf.keras.mixed_precision.global_policy()
        compute_dtype = policy.compute_dtype
    elif dtype_policy == 'float16':
        compute_dtype = tf.float16
    elif dtype_policy == 'mixed_float16':
        compute_dtype = tf.float16
    else:
        compute_dtype = tf.float32
    
    # Set drop_remainder default
    if drop_remainder is None:
        drop_remainder = training
    
    # Convert arrays to tensors for efficient access
    genotypes_tensor = tf.constant(genotypes, dtype=compute_dtype)
    coordinates_tensor = tf.constant(coordinates, dtype=tf.float32)
    
    # Create the base dataset from indices
    indices_dataset = tf.data.Dataset.from_tensor_slices(indices)
    
    # Define the data loading function
    def load_sample(idx):
        """Load a single sample by index."""
        # Get genotypes for this sample
        sample_genotypes = tf.gather(genotypes_tensor, idx, axis=1)
        
        if site_order is not None:
            # Bootstrap resampling: reorder SNPs
            sample_genotypes = tf.gather(sample_genotypes, site_order)
        
        # Get coordinates
        sample_coords = tf.gather(coordinates_tensor, idx)
        
        return sample_genotypes, sample_coords
    
    # Map indices to data
    # Use fixed parallelism to avoid excessive forking
    dataset = indices_dataset.map(
        load_sample,
        num_parallel_calls=4,  # Fixed instead of AUTOTUNE to reduce overhead
        deterministic=not training
    )
    
    # Add sample weights if provided
    if sample_weights is not None:
        if len(sample_weights) != len(indices):
            raise ValueError(
                f"Sample weights length ({len(sample_weights)}) must match "
                f"split size ({len(indices)})"
            )
        
        # Create weights dataset
        weights_dataset = tf.data.Dataset.from_tensor_slices(
            tf.constant(sample_weights, dtype=tf.float32)
        )
        
        # Zip with main dataset
        dataset = tf.data.Dataset.zip((dataset, weights_dataset))
        
        # Restructure to (features, labels, weights)
        dataset = dataset.map(
            lambda data_tuple, weight: (data_tuple[0], data_tuple[1], weight),
            num_parallel_calls=4  # Fixed instead of AUTOTUNE
        )
    
    # Apply caching before any randomness
    if cache:
        dataset = dataset.cache()
    
    # Apply augmentation if enabled
    if augment and augment.get('enabled', False):
        flip_rate = augment.get('flip_rate', 0.05)
        
        if sample_weights is not None:
            # With weights: (features, labels, weights)
            def augment_with_weights(features, labels, weights):
                augmented_features = flip_genotypes_tf(features, flip_rate)
                return augmented_features, labels, weights
            
            dataset = dataset.map(
                augment_with_weights,
                num_parallel_calls=4  # Fixed instead of AUTOTUNE
            )
        else:
            # Without weights: (features, labels)
            def augment_without_weights(features, labels):
                augmented_features = flip_genotypes_tf(features, flip_rate)
                return augmented_features, labels
            
            dataset = dataset.map(
                augment_without_weights,
                num_parallel_calls=4  # Fixed instead of AUTOTUNE
            )
    
    # Shuffle if training
    if training and shuffle_buffer > 0:
        dataset = dataset.shuffle(
            buffer_size=min(shuffle_buffer, len(indices)),
            reshuffle_each_iteration=True
        )
    
    # Batch the dataset
    dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)
    
    # Prefetch for performance
    if prefetch:
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    # Apply optimization options
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
    options.experimental_optimization.apply_default_optimizations = True
    # Disable map parallelization to avoid excessive forking
    options.experimental_optimization.map_parallelization = False
    # Use fixed thread pool instead of 0 to prevent process forking
    options.threading.private_threadpool_size = 4
    # Limit inter-op parallelism to reduce overhead
    options.threading.max_intra_op_parallelism = 1
    dataset = dataset.with_options(options)
    
    return dataset


def flip_genotypes_tf(genotypes: tf.Tensor, flip_rate: float = 0.05) -> tf.Tensor:
    """Randomly flip genotype values with given probability.
    
    This is a TensorFlow implementation of genotype flipping for data
    augmentation. It randomly flips allele values (0→1, 1→0, 2 stays 2).
    
    Args:
        genotypes: Tensor of genotype values
        flip_rate: Probability of flipping each value
        
    Returns:
        Augmented genotypes tensor
    """
    # Create random mask
    mask = tf.random.uniform(tf.shape(genotypes)) < flip_rate
    
    # Only flip values that are 0 or 1 (not missing data encoded as 2)
    is_flippable = tf.less(genotypes, 2.0)
    mask = tf.logical_and(mask, is_flippable)
    
    # Flip: 0→1, 1→0
    flipped = tf.where(mask, 1.0 - genotypes, genotypes)
    
    return flipped


def make_tf_dataset_from_arrays(
    train_gen: np.ndarray,
    train_locs: np.ndarray,
    test_gen: Optional[np.ndarray] = None,
    test_locs: Optional[np.ndarray] = None,
    val_gen: Optional[np.ndarray] = None,
    val_locs: Optional[np.ndarray] = None,
    batch_size: int = 256,
    **kwargs
) -> Union[tf.data.Dataset, Tuple[tf.data.Dataset, ...]]:
    """Legacy function to create datasets from pre-split arrays.
    
    This function provides backward compatibility for code that already
    has split arrays. It converts them to the new IndexSet format.
    
    Args:
        train_gen: Training genotypes of shape (n_train, n_features)
        train_locs: Training locations of shape (n_train, 2)
        test_gen: Optional test genotypes
        test_locs: Optional test locations
        val_gen: Optional validation genotypes
        val_locs: Optional validation locations
        batch_size: Batch size
        **kwargs: Additional arguments passed to make_tf_dataset
        
    Returns:
        Single dataset or tuple of datasets (train, test, val)
    """
    # Transpose to get (n_features, n_samples) shape expected by make_tf_dataset
    n_features = train_gen.shape[1]
    n_train = train_gen.shape[0]
    
    # Create combined arrays
    total_samples = n_train
    indices_dict = {"train": np.arange(n_train)}
    
    arrays_list = [train_gen.T]
    locs_list = [train_locs]
    
    if test_gen is not None:
        n_test = test_gen.shape[0]
        indices_dict["test"] = np.arange(total_samples, total_samples + n_test)
        arrays_list.append(test_gen.T)
        locs_list.append(test_locs)
        total_samples += n_test
    
    if val_gen is not None:
        n_val = val_gen.shape[0]
        indices_dict["val"] = np.arange(total_samples, total_samples + n_val)
        arrays_list.append(val_gen.T)
        locs_list.append(val_locs)
        total_samples += n_val
    
    # Stack arrays
    all_genotypes = np.hstack(arrays_list)
    all_locs = np.vstack(locs_list)
    
    # Create IndexSet
    index_set = IndexSet(indices=indices_dict, total_samples=total_samples)
    
    # Create datasets
    datasets = []
    
    # Training dataset
    train_dataset = make_tf_dataset(
        genotypes=all_genotypes,
        coordinates=all_locs,
        index_set=index_set,
        split="train",
        batch_size=batch_size,
        training=True,
        **kwargs
    )
    datasets.append(train_dataset)
    
    # Test dataset if provided
    if test_gen is not None:
        test_dataset = make_tf_dataset(
            genotypes=all_genotypes,
            coordinates=all_locs,
            index_set=index_set,
            split="test",
            batch_size=batch_size,
            training=False,
            shuffle_buffer=0,
            **kwargs
        )
        datasets.append(test_dataset)
    
    # Validation dataset if provided
    if val_gen is not None:
        val_dataset = make_tf_dataset(
            genotypes=all_genotypes,
            coordinates=all_locs,
            index_set=index_set,
            split="val",
            batch_size=batch_size,
            training=False,
            shuffle_buffer=0,
            **kwargs
        )
        datasets.append(val_dataset)
    
    # Return single dataset or tuple
    return datasets[0] if len(datasets) == 1 else tuple(datasets)