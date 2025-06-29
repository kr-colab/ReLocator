Data Pipeline Guide
===================

This guide covers the memory-efficient data pipeline architecture introduced in Locator, 
including the ``IndexSet`` class for zero-copy data splitting and the unified ``tf.data`` 
pipeline for optimal training performance.

Overview
--------

The new data pipeline architecture provides:

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

Bootstrap Analysis with Efficient Resampling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The new pipeline enables memory-efficient bootstrap resampling:

.. code-block:: python

    # Bootstrap resampling without data copies
    for boot in range(n_bootstraps):
        # Resample sites (SNPs) instead of copying data
        site_indices = np.random.choice(n_snps, n_snps, replace=True)
        
        # Create view without copying
        boot_dataset = make_tf_dataset(
            genotypes=genotypes,
            coordinates=coordinates,
            index_set=index_set,
            split="train",
            site_order=site_indices,  # Resampling happens in tf.gather
            batch_size=256
        )
        
        # Train on bootstrap sample
        model.fit(boot_dataset, ...)

Cross-Validation with K-Fold Splits
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from locator.data import IndexSet
    
    # Create k-fold splits
    k_fold_indices = IndexSet.k_fold(n=len(samples), k=5)
    
    for fold, fold_index_set in enumerate(k_fold_indices):
        print(f"Training fold {fold + 1}/5")
        
        # Datasets are created efficiently for each fold
        train_dataset = make_tf_dataset(
            genotypes=genotypes,
            coordinates=coordinates,
            index_set=fold_index_set,
            split="train",
            batch_size=256
        )
        
        val_dataset = make_tf_dataset(
            genotypes=genotypes,
            coordinates=coordinates,
            index_set=fold_index_set,
            split="test",  # 'test' is validation in k-fold
            batch_size=256
        )
        
        # Train model for this fold
        model.fit(train_dataset, validation_data=val_dataset, ...)

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

Performance Optimization
------------------------

Memory Usage
~~~~~~~~~~~~

The new pipeline significantly reduces memory usage:

.. code-block:: python

    # Old approach (creates copies)
    train_gen = genotypes[:, train_idx]  # Copy
    test_gen = genotypes[:, test_idx]    # Copy
    
    # New approach (no copies)
    dataset = make_tf_dataset(
        genotypes=genotypes,  # Original array
        index_set=index_set,  # Just indices
        split="train"
    )

For a dataset with 10,000 SNPs and 1,000 samples:

* Old approach: ~152 MB (3 copies for train/test/val)
* New approach: ~76 MB (original data only)

GPU Optimization
~~~~~~~~~~~~~~~~

The pipeline automatically optimizes for GPU usage:

.. code-block:: python

    # Automatic optimizations include:
    # - Prefetching to GPU
    # - Parallel data loading
    # - Optimized batch sizes
    
    dataset = make_tf_dataset(
        genotypes=genotypes,
        coordinates=coordinates,
        index_set=index_set,
        split="train",
        batch_size=256,
        cache=True,  # Cache in GPU memory if available
        prefetch_buffer=tf.data.AUTOTUNE
    )

Data Augmentation
~~~~~~~~~~~~~~~~~

Built-in augmentation improves model generalization:

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

Migration Guide
---------------

Updating Existing Code
~~~~~~~~~~~~~~~~~~~~~~

If you have existing Locator code, most functionality works unchanged. Locator now
always uses the efficient tf.data pipeline for optimal memory usage and performance.

Custom Training Loops
~~~~~~~~~~~~~~~~~~~~~

For custom training loops, replace array slicing with IndexSet:

.. code-block:: python

    # Old approach
    train_idx = np.random.choice(n_samples, int(0.8 * n_samples), replace=False)
    test_idx = np.setdiff1d(range(n_samples), train_idx)
    train_gen = genotypes[:, train_idx]
    test_gen = genotypes[:, test_idx]
    
    # New approach
    index_set = IndexSet.random_split(n=n_samples, splits={"train": 0.8, "test": 0.2})
    train_dataset = make_tf_dataset(
        genotypes=genotypes,
        coordinates=coordinates,
        index_set=index_set,
        split="train"
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