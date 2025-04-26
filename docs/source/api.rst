API Reference
=============

Core Module
-----------
.. module:: locator.core

.. autofunction:: setup_gpu

   Configure GPU settings for optimal usage.

   Args:
       gpu_number: Optional int or str specifying which GPU to use (0-based index).
                  If None, uses the first available GPU.

   Returns:
       bool: True if GPU is available and configured, False otherwise

Locator
^^^^^^^
.. autoclass:: Locator


EnsembleLocator
^^^^^^^^^^^^^^^
.. autoclass:: EnsembleLocator
   :members: 

 

Models Module
-------------
.. module:: locator.models

.. autofunction:: create_network

   Creates a neural network model for location prediction.

.. autofunction:: loss_with_range_penalty

   Custom loss function incorporating species range constraints.

.. autofunction:: rasterize_species_range

   Converts species range shapefile to raster format.

Utils Module
------------
.. module:: locator.utils

.. autofunction:: normalize_locs

   Normalizes geographic coordinates.

   Args:
       locs (numpy.ndarray): Array of [longitude, latitude] coordinates

   Returns:
       tuple: (mean_long, sd_long, mean_lat, sd_lat, normalized_locs)

.. autofunction:: filter_snps

   Filters SNPs based on minor allele count and other criteria.

   Args:
       genotypes (allel.GenotypeArray): Input genotype data
       min_mac (int): Minimum minor allele count
       max_snps (int, optional): Maximum number of SNPs to retain
       impute (bool): Whether to impute missing values




Configuration Options
---------------------
*This section provides an overview of the available configuration options.*

Default Configuration
^^^^^^^^^^^^^^^^^^^^^
The default configuration for Locator includes:

.. code-block:: python

   {
       # Data parameters
       "train_split": 0.9,
       "batch_size": 32,
       "min_mac": 2,
       "max_SNPs": None,
       "impute_missing": False,
       
       # Network architecture
       "width": 256,
       "nlayers": 8,
       "dropout_prop": 0.25,
       
       # Training parameters
       "max_epochs": 5000,
       "patience": 100,
       "learning_rate": 0.001,
       "min_epochs": 10,
       "min_delta": 1e-4,
       "restore_best_weights": True,
       
       # Optimizer parameters
       "optimizer_algo": "adam",
       "weight_decay": 0.004,
       
       # Output control
       "keras_verbose": 1,
       "prediction_frequency": 1,
       
       # Validation
       "validation_split": 0.1,
       
       # Data augmentation
       "augmentation": {
           "enabled": False,
           "flip_rate": 0.05
       },
       
       # Range penalty
       "use_range_penalty": False,
       "species_range_shapefile": None,
       "resolution": 0.05,
       "penalty_weight": 1.0
   }

Input Formats
^^^^^^^^^^^^^
Genotype Data
"""""""""""""
Supported input formats for genotype data:

1. VCF files (``.vcf`` or ``.vcf.gz``)
2. Zarr format (recommended for large datasets)
3. Pandas DataFrame with:
   - Samples as index
   - SNP positions as columns
   - Genotype counts (0,1,2) as values

Sample Data
"""""""""""
Required format for sample coordinate data:

- Tab-delimited file or DataFrame with columns:
  - ``sampleID``: Sample identifier
  - ``x``: Longitude
  - ``y``: Latitude

Output Formats
^^^^^^^^^^^^^^
Prediction Results
""""""""""""""""""
Default output files:

- ``{out}_predlocs.txt``: Main predictions
- ``{out}_history.txt``: Training history
- ``{out}_fitplot.pdf``: Training plots
- ``{out}.weights.h5``: Model weights

For special analyses:

- ``{out}_bootstrap_predlocs.csv``: Bootstrap results
- ``{out}_jacknife_predlocs.csv``: Jacknife results
- ``{out}_windows_predlocs.csv``: Windowed analysis results
- ``{out}_holdout_predlocs.csv``: Holdout analysis results

Error Handling
^^^^^^^^^^^^^^
Common error messages and their solutions:

GPU Errors
""""""""""
- ``GPU memory allocation error``: Reduce batch size or model size
- ``CUDA initialization error``: Check GPU drivers and TensorFlow installation

Examples
--------

This section provides examples of how to use the Locator package for various analysis scenarios.

Basic Usage
^^^^^^^^^^^

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
^^^^^^^^^^^^^^^^^

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
^^^^^^^^^^^^^^^^^

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
