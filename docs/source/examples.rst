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
    
    # Train the model
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