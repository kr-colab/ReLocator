API Reference
=============

Core Module
-----------
.. module:: locator.core

.. autofunction:: setup_gpu

Locator
^^^^^^^
.. autoclass:: Locator
   :members:
   :inherited-members:
   :show-inheritance:


Ensemble Functionality
----------------------

The ensemble functionality is integrated into the main ``Locator`` class through the ``EnsembleMixin``.

.. module:: locator.ensemble_mixin

EnsembleMixin
^^^^^^^^^^^^^
.. autoclass:: EnsembleMixin
   :members:
   :show-inheritance:

.. module:: locator.ensemble_model_manager

EnsembleModelManager
^^^^^^^^^^^^^^^^^^^^
.. autoclass:: EnsembleModelManager
   :members:
   :show-inheritance:

Parallel Ensemble Training
^^^^^^^^^^^^^^^^^^^^^^^^^^

The parallel ensemble training function is available when Ray is installed:

.. code-block:: python

   from locator.parallel import parallel_train_ensemble

.. function:: parallel_train_ensemble(locator, genotypes, samples, k=5, gpu_ids=[0, 1], gpu_fraction=1.0, training_set_indices=None, na_action=None, augment_data=False, flip_rate=0.05, save_fold_models=True, use_model_manager=True, use_mixed_precision=None, patience_multiplier=1.0, verbose=True)

   Train ensemble models in parallel across multiple GPUs using Ray.

   :param locator: Locator instance with configuration
   :param genotypes: GenotypeArray containing genetic data
   :param samples: Array of sample IDs
   :param k: Number of folds/models in ensemble (default: 5)
   :param gpu_ids: List of GPU IDs to use (default: [0, 1])
   :param gpu_fraction: Fraction of GPU memory per worker (default: 1.0)
   :param training_set_indices: Optional indices to restrict training
   :param na_action: How to handle NA samples ('separate', 'exclude', 'fail')
   :param augment_data: Whether to apply data augmentation
   :param flip_rate: Rate for genotype flipping augmentation
   :param save_fold_models: Whether to save individual fold models
   :param use_model_manager: Whether to use model manager for storage
   :param use_mixed_precision: Whether to use mixed precision training
   :param patience_multiplier: Multiply patience for ensemble training
   :param verbose: Whether to show training progress
   :returns: dict containing histories, models, normalization_params, fold_info

   .. note::
      This function requires Ray to be installed. Install with ``pip install locator[ray]``.



Models Module
-------------
.. module:: locator.models

.. autofunction:: create_network

.. autofunction:: loss_with_range_penalty

.. autofunction:: rasterize_species_range

Data Module
-----------
.. module:: locator.data

This module contains the memory-efficient data pipeline components.

IndexSet
^^^^^^^^
.. autoclass:: IndexSet
   :members:
   :show-inheritance:

Data Pipeline Functions
^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: make_tf_dataset

Preprocessing Functions
^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: filter_snps

.. autofunction:: normalize_locs

.. autofunction:: impute_missing

Data Classes
^^^^^^^^^^^^
.. autoclass:: FilterStats
   :members:

.. autoclass:: NormalizationParams
   :members:

Utils Module
------------
.. module:: locator.utils

.. autofunction:: weight_samples

GPU Optimizer Module
--------------------
.. module:: locator.gpu_optimizer

.. autoclass:: GPUOptimizer
   :members:


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

Parallel Analysis Module
------------------------
.. module:: locator.parallel

This module provides Ray-based parallel implementations of analysis methods for multi-GPU execution.

.. autofunction:: parallel_k_fold_holdouts

.. autofunction:: parallel_leave_one_out

.. autofunction:: parallel_holdouts

.. autofunction:: parallel_windows_holdouts

Plotting Module
---------------
.. module:: locator.plotting

This module provides visualization functions for Locator predictions and analyses.

Standalone Functions
^^^^^^^^^^^^^^^^^^^^

.. autofunction:: plot_predictions

.. autofunction:: plot_error_summary

.. autofunction:: plot_sample_weights

.. autofunction:: kde_predict

PlottingMixin Class
^^^^^^^^^^^^^^^^^^^

.. autoclass:: PlottingMixin
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:




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
           "flip_rate": 0.05,
       },

       # Sample weighting
       "weight_samples": {
           "enabled": False,
           "method": "KD",
           "xbins": 10,
           "ybins": 10,
           "lam": 1.0,
           "bandwidth": None,
           "weightdf": None,
       },

       # Range penalty
       "use_range_penalty": False,
       "species_range_shapefile": None,
       "resolution": 0.05,
       "penalty_weight": 1.0,
       "out": "locator",

       # NA handling
       "na_action": "separate",

       # GPU optimization (enabled by default)
       "use_mixed_precision": True,
       "gpu_batch_size": "auto",
       "gradient_accumulation_steps": 1,
       "gpu_memory_mode": "growth",
       "enable_xla": False,

       # Performance optimization
       "optimize_tf_parallelism": True,
       "holdout_no_intermediate_saves": True,
       "save_fold_models": True,

       # Verbosity control
       "verbose_splits": False,
       "verbose_batch_size": False,
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
