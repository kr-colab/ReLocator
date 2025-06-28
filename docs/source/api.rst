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
   :members:
   :inherited-members:
   :show-inheritance:


Ensemble Module
---------------
.. module:: locator.ensemble

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

Data Module
-----------
.. module:: locator.data

This module contains the memory-efficient data pipeline components.

IndexSet
^^^^^^^^
.. autoclass:: IndexSet
   :members:
   :show-inheritance:

   Memory-efficient data splitting using indices instead of array copies.

   Class Methods:
       - random_split: Create random train/test/validation splits
       - k_fold: Create k-fold cross-validation splits
       - group_split: Create splits based on group membership
       - leave_one_out: Create leave-one-out splits

   Attributes:
       - indices: Dictionary mapping split names to index arrays
       - n: Total number of samples

Data Pipeline Functions
^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: make_tf_dataset

   Create an optimized TensorFlow dataset for training or evaluation.

   Args:
       genotypes: Genotype array of shape (n_snps, n_samples)
       coordinates: Coordinate array of shape (n_samples, 2)
       index_set: IndexSet object defining data splits
       split: Which split to use ('train', 'test', 'val', etc.)
       batch_size: Batch size for the dataset
       training: Whether this is for training (enables shuffling)
       cache: Whether to cache the dataset in memory
       augment_flip_rate: Rate for genotype flipping augmentation
       sample_weights: Optional array of sample weights
       site_order: Optional array for SNP resampling (bootstrap)

   Returns:
       tf.data.Dataset: Optimized TensorFlow dataset

Preprocessing Functions
^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: filter_snps

   Filter SNPs based on criteria and return statistics.

   Args:
       genotypes: GenotypeArray to filter
       min_mac: Minimum minor allele count
       max_snps: Maximum number of SNPs to retain
       impute: Whether to impute missing data

   Returns:
       tuple: (filtered_array, FilterStats)

.. autofunction:: normalize_locs

   Normalize geographic coordinates to zero mean and unit variance.

   Args:
       locs: Array of [longitude, latitude] coordinates

   Returns:
       tuple: (NormalizationParams, normalized_coordinates)

.. autofunction:: impute_missing

   Impute missing genotype values.

   Args:
       allele_counts: Allele count array with missing values
       verbose: Whether to print progress

   Returns:
       numpy.ndarray: Imputed allele counts

Data Classes
^^^^^^^^^^^^
.. autoclass:: FilterStats
   :members:

   Statistics from SNP filtering operations.

.. autoclass:: NormalizationParams
   :members:

   Parameters for coordinate normalization with apply/reverse methods.

Utils Module
------------
.. module:: locator.utils

.. autofunction:: weight_samples

   Calculate sample weights based on geographic density.

GPU Optimizer Module
--------------------
.. module:: locator.gpu_optimizer

.. autoclass:: GPUOptimizer
   :members:

   Utilities for optimizing GPU performance in TensorFlow.

.. autoclass:: GradientAccumulator
   :members:

   Helper class for gradient accumulation to simulate larger batch sizes.

.. autofunction:: create_optimized_training_config

   Create an optimized configuration for GPU training.

   Args:
       base_config (dict): Base configuration dictionary

   Returns:
       dict: Optimized configuration with GPU settings

Internal Modules (Implementation Details)
-----------------------------------------
*These modules contain the implementation of Locator functionality. Users typically interact with these through the main Locator class.*

Loaders Module
^^^^^^^^^^^^^^
.. module:: locator.loaders

.. autoclass:: DataLoaderMixin
   :members:
   :noindex:

Training Module
^^^^^^^^^^^^^^^
.. module:: locator.training

.. autoclass:: TrainingMixin
   :members:
   :noindex:

Prediction Module
^^^^^^^^^^^^^^^^^
.. module:: locator.prediction

.. autoclass:: PredictionMixin
   :members:
   :noindex:

Analysis Module
^^^^^^^^^^^^^^^
.. module:: locator.analysis

.. autoclass:: AnalysisMixin
   :members:
   :noindex:

Plotting Module
^^^^^^^^^^^^^^^
.. module:: locator.plotting

.. autoclass:: PlottingMixin
   :members:
   :noindex:




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
       "penalty_weight": 1.0,
       
       # GPU optimization (enabled by default)
       "use_mixed_precision": True,
       "gpu_batch_size": "auto",
       "use_efficient_pipeline": True,
       "gradient_accumulation_steps": 1,
       "gpu_memory_mode": "growth",
       "enable_xla": False
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
