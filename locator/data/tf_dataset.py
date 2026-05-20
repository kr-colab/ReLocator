"""Index-based TensorFlow dataset creation for GPU-resident genotype training."""

from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np
import tensorflow as tf

from .indexset import IndexSet


def build_genotype_table(filtered_genotypes: np.ndarray) -> tf.Tensor:
    """Build a GPU-resident, sample-major genotype tensor.

    The genotype matrix for realistic runs is small enough to live on the GPU
    for the whole run -- its size is ``n_snps * n_samples * dtype_bytes``
    (e.g. 1M x 236 int8 ~= 236 MB, 2M x 236 int8 ~= 472 MB, 4x that for float32
    dosage). Building it once and gathering rows on-device removes the per-fold
    re-materialisation and per-epoch host-to-device copies of the old pipeline.

    Args:
        filtered_genotypes: Filtered genotype matrix of shape
            ``(n_snps, n_samples)`` -- int8 allele counts for hard calls, or
            float32 for genotype-likelihood dosage.

    Returns
    -------
        A ``tf.Tensor`` of shape ``(n_samples, n_snps)`` with the source dtype
        preserved, placed on the GPU when one is available.
    """
    arr = np.asarray(filtered_genotypes)
    # Sample-major rows make each sample's SNP vector contiguous, so the
    # per-batch row gather inside the model is coalesced. The native dtype is
    # kept (int8 stays int8); the cast to compute dtype happens per batch.
    sample_major = np.ascontiguousarray(arr.T)
    device = "/GPU:0" if tf.config.list_physical_devices("GPU") else "/CPU:0"
    with tf.device(device):
        return tf.constant(sample_major)


def make_tf_dataset(
    coordinates: np.ndarray,
    index_set: IndexSet,
    split: str,
    batch_size: int = 256,
    sample_weights: Optional[np.ndarray] = None,
    training: bool = True,
    shuffle_buffer: int = 1024,
    drop_remainder: Optional[bool] = None,
    prefetch: bool = True,
) -> tf.data.Dataset:
    """Create an index-based tf.data pipeline for training or validation.

    The pipeline carries only sample indices and their coordinates -- a few
    kilobytes per batch. Genotypes are gathered on the GPU inside
    ``IndexedGenotypeModel``, so the genotype matrix never enters this pipeline
    and there is no per-epoch host-to-device genotype traffic.

    Args:
        coordinates: Full coordinate array of shape ``(n_samples, 2)``.
        index_set: IndexSet containing the train/val/test/predict splits.
        split: Which split to use ('train', 'val', 'test', 'predict').
        batch_size: Batch size for the dataset.
        sample_weights: Optional per-sample weights, aligned to the split's
            index order (length must equal the split size).
        training: Whether this is for training (enables shuffling).
        shuffle_buffer: Non-zero enables shuffling when ``training`` is True.
        drop_remainder: Whether to drop the final partial batch
            (defaults to the value of ``training``).
        prefetch: Whether to prefetch batches.

    Returns
    -------
        A ``tf.data.Dataset`` yielding ``(sample_index, coordinate)`` batches,
        or ``(sample_index, coordinate, sample_weight)`` when weights are given.
    """
    indices = np.asarray(index_set.get_split(split))

    if len(indices) == 0:
        raise ValueError(f"Split '{split}' has no samples")

    if drop_remainder is None:
        drop_remainder = training

    indices = indices.astype(np.int32)
    coords = np.asarray(coordinates)[indices].astype(np.float32)

    if sample_weights is not None:
        if len(sample_weights) != len(indices):
            raise ValueError(
                f"Sample weights length ({len(sample_weights)}) must match "
                f"split size ({len(indices)})"
            )
        weights = np.asarray(sample_weights, dtype=np.float32)
        dataset = tf.data.Dataset.from_tensor_slices((indices, coords, weights))
    else:
        dataset = tf.data.Dataset.from_tensor_slices((indices, coords))

    if training and shuffle_buffer > 0:
        dataset = dataset.shuffle(
            buffer_size=len(indices), reshuffle_each_iteration=True
        )

    dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)

    if prefetch:
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


def flip_genotypes_tf(genotypes: tf.Tensor, flip_rate: float = 0.05) -> tf.Tensor:
    """Randomly flip genotype values with a given probability.

    Randomly flips allele values (0 to 1, 1 to 0); 2 (missing) is never flipped.
    Used for training-time data augmentation.

    Args:
        genotypes: Tensor of genotype values.
        flip_rate: Probability of flipping each value.

    Returns
    -------
        Augmented genotypes tensor.
    """
    # Create random mask
    mask = tf.random.uniform(tf.shape(genotypes)) < flip_rate

    # Only flip values that are 0 or 1 (not missing data encoded as 2)
    is_flippable = tf.less(genotypes, 2.0)
    mask = tf.logical_and(mask, is_flippable)

    # Flip: 0 to 1, 1 to 0
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
    **kwargs,
) -> Union[tf.data.Dataset, Tuple[tf.data.Dataset, ...]]:
    """Legacy helper: build feature-based datasets from pre-split arrays.

    Kept for backward compatibility. Unlike :func:`make_tf_dataset` (which is
    index-based), this yields ``(genotype_features, coordinates)`` batches
    directly from the supplied sample-major arrays.

    Args:
        train_gen: Training genotypes of shape ``(n_train, n_features)``.
        train_locs: Training locations of shape ``(n_train, 2)``.
        test_gen: Optional test genotypes.
        test_locs: Optional test locations.
        val_gen: Optional validation genotypes.
        val_locs: Optional validation locations.
        batch_size: Batch size.
        **kwargs: Accepts ``cache`` and ``prefetch`` flags.

    Returns
    -------
        A single dataset, or a tuple of datasets (train, test, val).
    """
    cache = kwargs.get("cache", True)
    prefetch = kwargs.get("prefetch", True)

    def _build(gen, locs, training):
        ds = tf.data.Dataset.from_tensor_slices(
            (np.asarray(gen, dtype=np.float32), np.asarray(locs, dtype=np.float32))
        )
        if cache:
            ds = ds.cache()
        if training:
            ds = ds.shuffle(len(gen), reshuffle_each_iteration=True)
        ds = ds.batch(batch_size, drop_remainder=training)
        if prefetch:
            ds = ds.prefetch(tf.data.AUTOTUNE)
        return ds

    datasets = [_build(train_gen, train_locs, training=True)]
    if test_gen is not None:
        datasets.append(_build(test_gen, test_locs, training=False))
    if val_gen is not None:
        datasets.append(_build(val_gen, val_locs, training=False))

    return datasets[0] if len(datasets) == 1 else tuple(datasets)
