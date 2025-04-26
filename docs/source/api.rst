API Reference
============

Core Module
----------

.. module:: locator.core

Locator
~~~~~~~

.. autoclass:: Locator
   :members:
   :special-members: __init__

   The main class for predicting geographic locations from genetic data.

EnsembleLocator
~~~~~~~~~~~~~~

.. autoclass:: EnsembleLocator
   :members:
   :special-members: __init__

   A class for managing an ensemble of Locator models.

Models Module
------------

.. module:: locator.models

.. autofunction:: create_network

   Creates a neural network model for location prediction.

.. autofunction:: loss_with_range_penalty

   Custom loss function incorporating species range constraints.

.. autofunction:: rasterize_species_range

   Converts species range shapefile to raster format.

Utils Module
-----------

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

Data Module
----------

.. module:: locator.data

.. autoclass:: DataGenerator
   :members:
   :special-members: __init__

   Generates batches of data for training.

.. autofunction:: load_genotype_data

   Loads genotype data from various file formats.

Metrics Module
-------------

.. module:: locator.metrics

.. autofunction:: evaluate_predictions

   Evaluates prediction accuracy using various metrics.

   Args:
       true_coords (numpy.ndarray): True coordinates
       predicted_coords (numpy.ndarray): Predicted coordinates

   Returns:
       dict: Dictionary of evaluation metrics

Visualization Module
------------------

.. module:: locator.visualization

.. autofunction:: plot_predictions

   Visualizes predicted vs true locations.

.. autofunction:: plot_error_summary

   Creates summary plots of prediction errors.

Configuration
------------

Default Configuration
~~~~~~~~~~~~~~~~~~~

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
~~~~~~~~~~~~

Genotype Data
************

Supported input formats for genotype data:

1. VCF files (``.vcf`` or ``.vcf.gz``)
2. Zarr format (recommended for large datasets)
3. Pandas DataFrame with:
   - Samples as index
   - SNP positions as columns
   - Genotype counts (0,1,2) as values

Sample Data
**********

Required format for sample coordinate data:

- Tab-delimited file or DataFrame with columns:
  - ``sampleID``: Sample identifier
  - ``x``: Longitude
  - ``y``: Latitude

Output Formats
~~~~~~~~~~~~~

Prediction Results
****************

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
-------------

Common error messages and their solutions:

GPU Errors
~~~~~~~~~

- ``GPU memory allocation error``: Reduce batch size or model size
- ``CUDA initialization error``: Check GPU drivers and TensorFlow installation
