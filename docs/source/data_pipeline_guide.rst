Data Pipeline Guide
===================

This guide covers the memory-efficient data pipeline architecture introduced in Locator,
including the ``IndexSet`` class for zero-copy data splitting and the unified ``tf.data``
pipeline for optimal training performance.

Overview
--------

The data pipeline architecture provides:

* **Memory-efficient data handling** through index-based operations instead of array copies
* **Unified tf.data pipeline** for consistent, high-performance data loading
* **Built-in data augmentation** support
* **Automatic batching and prefetching** optimization
* **Consistent preprocessing** across all analysis methods

Key Components
--------------

IndexSet: Memory-Efficient Data Splitting
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``IndexSet`` class manages train/test/validation splits using indices rather than copying data:

.. code-block:: python

    from locator.data import IndexSet

    # Create a random 80/20 train/test split
    index_set = IndexSet.random_split(
        n=1000,
        splits={"train": 0.8, "test": 0.2}
    )

    # Access indices for each split
    train_indices = index_set.train
    test_indices = index_set.test

    # Use with your data (no copying!)
    train_data = full_data[train_indices]

Advanced splitting options:

.. code-block:: python

    # K-fold cross-validation
    index_sets = IndexSet.k_fold(n=1000, k=5)
    for fold, idx_set in enumerate(index_sets):
        print(f"Fold {fold}: {len(idx_set.train)} train, {len(idx_set.test)} test")

    # Group-based splitting (e.g., by population)
    index_set = IndexSet.group_split(
        n=1000,
        groups=population_labels,
        test_groups=["pop1", "pop2"]
    )

    # Handling samples with missing data
    na_mask = np.isnan(coordinates[:, 0])
    index_set = IndexSet.random_split(
        n=1000,
        splits={"train": 0.8, "test": 0.2},
        na_mask=na_mask,
        na_action="separate"  # Creates a 'predict' split
    )

Unified TensorFlow Data Pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``make_tf_dataset`` function creates optimized tf.data pipelines:

.. code-block:: python

    from locator.data import make_tf_dataset

    # Create training dataset with all optimizations
    train_dataset = make_tf_dataset(
        genotypes=genotype_array,      # Shape: (n_snps, n_samples)
        coordinates=coordinate_array,   # Shape: (n_samples, 2)
        index_set=index_set,
        split="train",
        batch_size=256,
        training=True,                  # Enables shuffling
        cache=True,                     # Cache in memory
        augment_flip_rate=0.05,        # Data augmentation
        sample_weights=weights_array    # Optional sample weights
    )

    # Use directly with model.fit()
    model.fit(train_dataset, epochs=100, ...)

Data Preprocessing Utilities
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Centralized preprocessing functions with tracking:

.. code-block:: python

    from locator.data import filter_snps, normalize_locs, impute_missing
    from locator.data import FilterStats, NormalizationParams

    # Filter SNPs and get statistics
    filtered_geno, stats = filter_snps(
        genotypes,
        min_mac=2,
        max_snps=10000,
        impute=True
    )
    print(f"Retained {stats.n_snps_retained}/{stats.n_snps_original} SNPs")

    # Normalize coordinates with parameters
    norm_params, normalized_coords = normalize_locs(coordinates)
    # Apply same normalization to new data
    new_normalized = norm_params.apply(new_coordinates)
    # Reverse transformation
    original_coords = norm_params.reverse(normalized_coords)

Usage Examples
--------------

Basic Training with Memory-Efficient Pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import numpy as np
    from locator import Locator
    from locator.data import IndexSet, make_tf_dataset

    # Initialize Locator
    loc = Locator({
        "out": "results/analysis",
        "sample_data": "samples.txt",
        "max_epochs": 1000
    })

    # Load data
    genotypes, samples = loc.load_genotypes(zarr="data.zarr")
    sample_data, coordinates = loc.sort_samples(samples)

    # The memory-efficient pipeline is used automatically in train()
    loc.train(genotypes=genotypes, samples=samples)

Manual Pipeline Construction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For custom workflows, you can build the pipeline manually:

.. code-block:: python

    from locator.data import filter_snps, normalize_locs, IndexSet, make_tf_dataset

    # Preprocess data
    filtered_geno, filter_stats = filter_snps(genotypes, min_mac=2)
    norm_params, norm_coords = normalize_locs(coordinates)

    # Create data splits
    index_set = IndexSet.random_split(
        n=len(samples),
        splits={"train": 0.8, "val": 0.1, "test": 0.1}
    )

    # Build datasets
    train_dataset = make_tf_dataset(
        genotypes=filtered_geno,
        coordinates=norm_coords,
        index_set=index_set,
        split="train",
        batch_size=256,
        training=True,
        augment_flip_rate=0.05
    )

    val_dataset = make_tf_dataset(
        genotypes=filtered_geno,
        coordinates=norm_coords,
        index_set=index_set,
        split="val",
        batch_size=256,
        training=False
    )


Working with Sample Weights
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from locator.utils import weight_samples

    # Calculate sample weights based on geographic density
    weights_dict = weight_samples(
        method="gaussian",
        trainlocs=coordinates[train_indices],
        trainsamps=samples[train_indices],
        bandwidth=100  # km
    )

    # Include weights in dataset
    train_dataset = make_tf_dataset(
        genotypes=genotypes,
        coordinates=coordinates,
        index_set=index_set,
        split="train",
        sample_weights=weights_dict['sample_weights'],
        batch_size=256
    )


Data Augmentation
~~~~~~~~~~~~~~~~~

Built-in augmentation may improve model generalization:

.. code-block:: python

    # Genotype flipping augmentation
    train_dataset = make_tf_dataset(
        genotypes=genotypes,
        coordinates=coordinates,
        index_set=index_set,
        split="train",
        augment_flip_rate=0.05,  # Flip 5% of genotypes
        training=True
    )


API Reference
-------------

For detailed API documentation, see:

* :class:`locator.data.IndexSet` - Index-based data splitting
* :func:`locator.data.make_tf_dataset` - TensorFlow dataset creation
* :func:`locator.data.filter_snps` - SNP filtering with statistics
* :func:`locator.data.normalize_locs` - Coordinate normalization
* :class:`locator.data.FilterStats` - Filtering statistics
* :class:`locator.data.NormalizationParams` - Normalization parameters

Troubleshooting
---------------

Out of Memory Errors
~~~~~~~~~~~~~~~~~~~~

If you encounter memory issues:

1. Ensure the efficient pipeline is enabled (default)
2. Reduce batch size
3. Enable gradient accumulation for effective larger batches
4. Use ``cache=False`` for very large datasets

.. code-block:: python

    loc = Locator({
        "use_efficient_pipeline": True,
        "batch_size": 128,  # Smaller batches
        "gradient_accumulation_steps": 4,  # Effective batch size 512
    })

Slow Data Loading
~~~~~~~~~~~~~~~~~

For optimal performance:

1. Use Zarr format instead of VCF for large datasets
2. Enable caching for datasets that fit in memory
3. Ensure prefetching is enabled (default)

.. code-block:: python

    # Optimal settings for datasets that fit in memory
    dataset = make_tf_dataset(
        genotypes=genotypes,
        coordinates=coordinates,
        index_set=index_set,
        split="train",
        cache=True,  # Cache after preprocessing
        prefetch_buffer=tf.data.AUTOTUNE
    )
