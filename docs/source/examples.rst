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
    import numpy as np

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
        "sample_data": "samples.txt"
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

    from locator import Locator
    from locator.plotting import plot_sample_weights

    # Use kernel density (KD) weighting to upweight undersampled regions
    loc = Locator({
        "out": "weighted_analysis",
        "sample_data": "samples.txt",
        "weight_samples": {
            "enabled": True,
            "method": "KD",  # Kernel density method
            "bandwidth": None  # Auto-calculate optimal bandwidth
        }
    })

    # Weights are applied automatically during training
    loc.train(genotypes=genotypes, samples=samples)

    # Visualize the sample weights
    plot_sample_weights(loc, "sample_weight_distribution")

    # Alternative: Use histogram binning method
    loc_hist = Locator({
        "out": "hist_weighted",
        "sample_data": "samples.txt",
        "weight_samples": {
            "enabled": True,
            "method": "hist",
            "xbins": 20,
            "ybins": 20
        }
    })

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

GPU Optimization Examples
-------------------------

Automatic GPU Optimization
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from locator import Locator

    # GPU optimizations are enabled by default
    loc = Locator({
        "out": "gpu_optimized",
        "sample_data": "samples.txt",
        # Automatic mixed precision and batch size optimization
    })

    # Monitor GPU usage during training
    loc.train(genotypes=genotypes, samples=samples)

    # For memory-constrained GPUs
    loc_constrained = Locator({
        "out": "memory_limited",
        "sample_data": "samples.txt",
        "gpu_batch_size": 64,  # Smaller batch size
        "gradient_accumulation_steps": 4  # Simulate larger batches
    })

Multi-GPU Parallel Analysis
---------------------------

K-Fold Cross-Validation Across GPUs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from locator import Locator
    from locator.parallel import parallel_k_fold_holdouts
    from locator.plotting import plot_error_summary

    # Initialize locator
    loc = Locator({
        "out": "parallel_kfold",
        "sample_data": "samples.txt",
        "width": 256,
        "nlayers": 10
    })

    # Run 10-fold CV across 4 GPUs
    predictions = parallel_k_fold_holdouts(
        loc, genotypes, samples,
        k=10,
        gpu_ids=[0, 1, 2, 3],  # Use 4 GPUs
        return_df=True,
        verbose=True
    )

    # Visualize results
    plot_error_summary(
        predictions,
        "samples.txt",
        "parallel_kfold_errors",
        use_geodesic=True
    )

Parallel Bootstrap Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from locator.parallel import parallel_holdouts

    # Run 100 bootstrap replicates across 2 GPUs
    bootstrap_results = parallel_holdouts(
        loc, genotypes, samples,
        k=len(samples),  # Bootstrap: sample with replacement
        n_reps=100,
        gpu_ids=[0, 1],
        return_df=True
    )

Parallel Windows Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from locator.parallel import parallel_windows_holdouts

    # Analyze specific samples across genomic windows
    worst_samples = ['HG001', 'HG002', 'HG003']

    window_results = parallel_windows_holdouts(
        loc, genotypes, samples,
        holdout_sample_ids=worst_samples,
        window_size=int(1e6),  # 1Mb windows
        gpu_ids=[0, 1, 2, 3],
        return_df=True
    )

Visualization Examples
----------------------

Visualizing Prediction Uncertainty
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from locator import Locator
    from locator.plotting import plot_predictions

    # Run jacknife analysis
    loc = Locator({"out": "jacknife_viz", "sample_data": "samples.txt"})
    genotypes, samples = loc.load_genotypes(zarr="genotypes.zarr")

    jack_preds = loc.run_jacknife(
        genotypes, samples,
        prop=0.1,
        n_replicates=100,
        return_df=True
    )

    # Visualize prediction distributions for specific samples
    plot_predictions(
        jack_preds,
        loc,
        "jacknife_uncertainty",
        samples=['sample_001', 'sample_002', 'sample_003'],
        n_cols=3,
        plot_map=True  # Use geographic projection
    )

Comparing Analysis Methods
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Compare jacknife vs bootstrap predictions
    boot_preds = loc.run_bootstraps(
        genotypes, samples,
        n_bootstraps=100,
        return_df=True
    )

    # Plot same samples from both analyses
    test_samples = jack_preds['sampleID'].unique()[:6]

    plot_predictions(jack_preds, loc, "jacknife_comparison",
                    samples=test_samples, n_cols=2)
    plot_predictions(boot_preds, loc, "bootstrap_comparison",
                    samples=test_samples, n_cols=2)

Error Analysis Workflow
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from locator.plotting import plot_error_summary

    # After k-fold cross-validation
    kfold_preds = loc.run_k_fold_holdouts(
        genotypes, samples,
        k=10,
        return_df=True
    )

    # Create comprehensive error visualization
    plot_error_summary(
        kfold_preds,
        "samples.txt",
        "kfold_error_analysis",
        use_geodesic=True,  # Errors in km
        include_training_locs=True,  # Show geographic context
        width=15,  # Custom figure size
        height=8
    )

Complete Workflow Example
-------------------------

From Data to Publication Figure
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from locator import Locator
    from locator.parallel import parallel_k_fold_holdouts
    from locator.plotting import plot_error_summary, plot_predictions
    import matplotlib.pyplot as plt

    # 1. Setup and data loading
    config = {
        "out": "actinemys_analysis",
        "sample_data": "actinemys_samples.txt",
        "min_mac": 2,
        "max_SNPs": 50000,
        "width": 256,
        "nlayers": 10,
        "dropout_prop": 0.25,
        "weight_samples": {
            "enabled": True,
            "method": "KD"
        }
    }

    loc = Locator(config)
    genotypes, samples = loc.load_genotypes(zarr="actinemys.zarr")

    # 2. Check data quality
    loc.check_data(genotypes, samples, verbose=True)

    # 3. Run parallel k-fold CV
    predictions = parallel_k_fold_holdouts(
        loc, genotypes, samples,
        k=10,
        gpu_ids=[0, 1, 2, 3],
        return_df=True
    )

    # 4. Create publication figure
    plot_error_summary(
        predictions,
        "actinemys_samples.txt",
        "figure_2a",
        dpi=600,  # Publication quality
        width=7,  # Single column
        height=4
    )

    # 5. Identify worst predictions for further analysis
    import pandas as pd
    sample_data = pd.read_csv("actinemys_samples.txt", sep="\t")
    merged = predictions.merge(sample_data[['sampleID', 'x', 'y']], on='sampleID')
    merged['error_km'] = merged.apply(
        lambda r: ((r.x_pred - r.x)**2 + (r.y_pred - r.y)**2)**0.5 * 111.32,
        axis=1
    )
    worst_samples = merged.nlargest(6, 'error_km')['sampleID'].tolist()

    # 6. Run windowed analysis on worst samples
    from locator.parallel import parallel_windows_holdouts

    window_results = parallel_windows_holdouts(
        loc, genotypes, samples,
        holdout_sample_ids=worst_samples,
        window_size=int(5e5),
        gpu_ids=[0, 1],
        return_df=True
    )

    # 7. Visualize window predictions
    plot_predictions(
        window_results,
        loc,
        "figure_2b",
        samples=worst_samples,
        n_cols=3,
        dpi=600
    )
