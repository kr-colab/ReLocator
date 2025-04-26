Examples
========

This section provides examples of how to use the Locator package for various analysis scenarios.

Basic Usage
-----------

.. code-block:: python

    import locator
    from locator.core import Locator
    
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

    from locator.core import EnsembleLocator
    
    # Initialize ensemble
    ensemble = EnsembleLocator(
        base_config={"out": "ensemble_analysis"},
        k_folds=5
    )
    
    # Train ensemble
    ensemble.train(genotypes=genotypes, samples=samples)
    
    # Make predictions
    ensemble_predictions = ensemble.predict() 