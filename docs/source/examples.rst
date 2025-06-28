Examples
========

This section provides examples of how to use the Locator package for various analysis scenarios.

Basic Usage
-----------

.. code-block:: python

    from locator import Locator
    
    # Initialize Locator with configuration
    loc = Locator({
        "out": "my_analysis",
        "sample_data": "samples.txt",
        "zarr": "genotypes.zarr"
    })
    
    # Load genotype data
    genotypes, samples = loc.load_genotypes(zarr="genotypes.zarr")
    
    # Train the model (uses memory-efficient pipeline automatically)
    loc.train(genotypes=genotypes, samples=samples)
    
    # Make predictions
    predictions = loc.predict(return_df=True)
    
    # Plot results
    loc.plot_history(loc.history)

Advanced Analysis
-----------------

.. code-block:: python

    # Run windowed analysis
    window_results = loc.run_windows(
        genotypes=genotypes,
        samples=samples,
        window_size=1e6
    )
    
    # Run jacknife analysis
    jacknife_results = loc.run_jacknife(
        genotypes=genotypes,
        samples=samples,
        prop=0.1
    )
    
    # Run bootstrap analysis
    bootstrap_results = loc.run_bootstraps(
        genotypes=genotypes,
        samples=samples,
        n_bootstraps=100
    )

Ensemble Analysis
-----------------

.. code-block:: python

    from locator import EnsembleLocator
    
    # Initialize ensemble
    ensemble = EnsembleLocator(
        base_config={"out": "ensemble_analysis"},
        k_folds=5
    )
    
    # Train ensemble
    ensemble.train(genotypes=genotypes, samples=samples)
    
    # Make predictions
    ensemble_predictions = ensemble.predict()

Handling Missing Coordinates
----------------------------

This example shows how to work with datasets where some samples lack geographic coordinates.

.. code-block:: python

    from locator import Locator
    import pandas as pd
    
    # Sample data with some missing coordinates
    sample_data = pd.DataFrame({
        'sampleID': ['A', 'B', 'C', 'D', 'E'],
        'x': [10.5, 20.3, np.nan, 15.2, np.nan],
        'y': [45.2, 50.1, np.nan, 48.3, np.nan]
    })
    
    # Initialize with default 'separate' mode
    loc = Locator({
        "out": "na_example",
        "sample_data": sample_data,
        "na_action": "separate"  # Default: train on known, predict on unknown
    })
    
    # Check data quality
    loc.check_data(genotypes, samples, verbose=True)
    # Output:
    # ===== Data Summary =====
    # Total samples: 5
    # Samples with coordinates: 3
    # Samples without coordinates: 2
    # ...
    
    # Train on samples with coordinates (A, B, D)
    # and predict locations for samples without (C, E)
    loc.train(genotypes=genotypes, samples=samples)
    predictions = loc.predict(return_df=True)
    
    # The predictions DataFrame will include predicted
    # locations for samples C and E

Excluding Samples Without Coordinates
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Use 'exclude' mode to only work with samples that have coordinates
    loc_exclude = Locator({
        "out": "exclude_example",
        "sample_data": sample_data,
        "na_action": "exclude"
    })
    
    # Only samples A, B, and D will be used
    loc_exclude.train(genotypes=genotypes, samples=samples)
    
    # Bootstrap analysis with only known-location samples
    bootstrap_results = loc_exclude.run_bootstraps(
        genotypes=genotypes,
        samples=samples,
        n_bootstraps=50
    )

Strict Mode - Fail on Missing Coordinates
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Use 'fail' mode to ensure all samples have coordinates
    loc_strict = Locator({
        "out": "strict_example",
        "sample_data": sample_data,
        "na_action": "fail"
    })
    
    # This will raise an error because samples C and E lack coordinates
    try:
        loc_strict.train(genotypes=genotypes, samples=samples)
    except ValueError as e:
        print(f"Error: {e}")
        # Error: Found 2 samples without coordinates. Set na_action='separate' or 'exclude' to proceed.

Mixed Analysis Modes
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Start with default 'separate' mode
    loc = Locator({
        "out": "mixed_example",
        "sample_data": sample_data
    })
    
    # Train with all samples (separate mode)
    loc.train(genotypes=genotypes, samples=samples)
    
    # But use exclude mode for k-fold cross-validation
    # (since holdout methods need coordinates for evaluation)
    kfold_results = loc.run_k_fold_holdouts(
        genotypes=genotypes,
        samples=samples,
        k=3,
        na_action="exclude"  # Override instance setting
    )

Memory-Efficient Data Pipeline
------------------------------

The new data pipeline provides memory-efficient operations and advanced features.

Using IndexSet for Custom Splits
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from locator import Locator
    from locator.data import IndexSet
    
    # Create custom data splits without copying arrays
    n_samples = len(samples)
    
    # 70/15/15 train/val/test split
    index_set = IndexSet.random_split(
        n=n_samples,
        splits={"train": 0.7, "val": 0.15, "test": 0.15}
    )
    
    # Access indices for each split
    print(f"Training samples: {len(index_set.train)}")
    print(f"Validation samples: {len(index_set.val)}")
    print(f"Test samples: {len(index_set.test)}")
    
    # Use with your data - no copying!
    train_genotypes = genotypes[:, index_set.train]
    val_genotypes = genotypes[:, index_set.val]

Bootstrap Analysis with Site Resampling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from locator import Locator
    import numpy as np
    
    # Initialize Locator
    loc = Locator({
        "out": "bootstrap_analysis",
        "sample_data": "samples.txt",
        "use_efficient_pipeline": True  # Default
    })
    
    # Load data
    genotypes, samples = loc.load_genotypes(zarr="genotypes.zarr")
    
    # Memory-efficient bootstrap (no data copies)
    n_bootstraps = 100
    n_snps = genotypes.shape[0]
    
    for boot in range(n_bootstraps):
        # Resample SNP indices
        site_indices = np.random.choice(n_snps, n_snps, replace=True)
        
        # Train with resampled sites (handled efficiently in pipeline)
        loc.train(
            genotypes=genotypes,
            samples=samples,
            boot=boot,
            site_order=site_indices  # Resampling without copying
        )
        
        # Make predictions
        loc.predict(boot=boot)

Data Augmentation
~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Enable data augmentation for better generalization
    loc = Locator({
        "out": "augmented_analysis",
        "sample_data": "samples.txt",
        "augmentation": {
            "enabled": True,
            "flip_rate": 0.05  # Randomly flip 5% of genotypes
        }
    })
    
    # Augmentation is applied during training automatically
    loc.train(genotypes=genotypes, samples=samples)

Custom TensorFlow Dataset Pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from locator.data import filter_snps, normalize_locs, IndexSet, make_tf_dataset
    
    # Preprocess data with tracking
    filtered_geno, filter_stats = filter_snps(
        genotypes, 
        min_mac=2, 
        max_snps=10000,
        impute=True
    )
    print(f"Retained {filter_stats.n_snps_retained} of {filter_stats.n_snps_original} SNPs")
    
    # Normalize coordinates
    norm_params, norm_coords = normalize_locs(coordinates)
    
    # Create efficient data pipeline
    index_set = IndexSet.random_split(n=len(samples), splits={"train": 0.8, "test": 0.2})
    
    train_dataset = make_tf_dataset(
        genotypes=filtered_geno,
        coordinates=norm_coords,
        index_set=index_set,
        split="train",
        batch_size=256,
        training=True,
        cache=True,  # Cache in memory
        augment_flip_rate=0.05
    )
    
    # Use with custom training loop
    for batch_genotypes, batch_coords in train_dataset:
        # Your custom training step
        pass

Working with Sample Weights
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from locator.utils import weight_samples
    
    # Calculate geographic density weights
    weights_dict = weight_samples(
        method="gaussian",
        trainlocs=coordinates[train_indices],
        trainsamps=samples[train_indices],
        bandwidth=100  # km
    )
    
    # Use weights in training
    loc = Locator({
        "out": "weighted_analysis",
        "sample_data": "samples.txt",
        "weight_samples": {
            "enabled": True,
            "method": "gaussian",
            "bandwidth": 100
        }
    })
    
    # Weights are applied automatically
    loc.train(genotypes=genotypes, samples=samples)

Loading and Using Saved Models
------------------------------

The new pipeline includes model persistence features.

Saving Model with Metadata
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Train and save model with all preprocessing parameters
    loc = Locator({
        "out": "my_model",
        "sample_data": "samples.txt",
        "min_mac": 3,
        "max_SNPs": 5000,
        "impute_missing": True
    })
    
    loc.train(genotypes=genotypes, samples=samples)
    # Model weights and metadata saved to my_model.weights.h5

Loading Model in New Session
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Load model and metadata
    loc2 = Locator({"out": "predictions"})
    
    # Load the saved model
    metadata = loc2.load_model("my_model.weights.h5")
    print(f"Model trained on {metadata['n_samples']} samples")
    print(f"Normalization params: {metadata['normalization']}")
    
    # Make predictions with proper preprocessing
    new_predictions = loc2.predict_from_weights(
        weights_path="my_model.weights.h5",
        genotypes=new_genotypes,
        samples=new_samples,
        sample_data_file="new_samples.txt"
    )

Command Line Usage
~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    # Make predictions using saved weights
    locator \
        --predict_from_weights my_model.weights.h5 \
        --zarr new_data.zarr \
        --sample_data new_samples.txt \
        --out new_predictions 