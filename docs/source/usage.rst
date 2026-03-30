Usage Guide
===========

This guide covers how to use Locator for predicting geographic coordinates
from genotype matrices.

Basic Usage
-----------

Loading Data
~~~~~~~~~~~~

Locator supports multiple input formats for genotype data:

.. code-block:: python

   from locator import Locator

   # Create a Locator instance with configuration
   config = {
       "out": "my_analysis",
       "batch_size": 32,
       "width": 256,
       "nlayers": 8,
       "dropout_prop": 0.25,
   }

   locator = Locator(config)

   # Load data from various formats:
   #
   # 1. From VCF
   genotypes, samples = locator.load_genotypes(vcf="path/to/genotypes.vcf")
   #
   # 2. From zarr (recommended for large datasets)
   #    See :doc:`cli` for VCF-to-Zarr conversion instructions.
   genotypes, samples = locator.load_genotypes(zarr="path/to/genotypes.zarr")
   #
   # 3. From pandas DataFrame
   locator = Locator({
       "out": "my_analysis",
       "genotype_data": genotype_df,  # DataFrame: samples as index, SNPs as columns
       "sample_data": coords_df,      # DataFrame with sampleID, x, y columns
   })

Training and Prediction
-----------------------

Train the model and make predictions:

.. code-block:: python

   # Train the model
   history = locator.train(genotypes=genotypes, samples=samples)

   # Make predictions
   predictions = locator.predict(return_df=True)  # Returns DataFrame with sampleID, x, y

Holdout Analysis
----------------

Evaluate model performance by holding out samples:

.. code-block:: python

   # Hold out k samples during training
   locator.train_holdout(
       genotypes=genotypes,
       samples=samples,
       k=10,
   )

   # Get predictions for held-out samples
   holdout_preds = locator.predict_holdout(
       return_df=True,
       plot_summary=True,
   )

Ensemble Models
---------------

Train ensemble models using k-fold cross-validation for improved predictions
with uncertainty estimates:

.. code-block:: python

   # Train 5-fold ensemble and predict with uncertainty
   locator.train_ensemble(genotypes=genotypes, samples=samples, k=5)
   predictions = locator.predict_ensemble(
       genotypes=genotypes, samples=samples, return_std=True,
   )

See :doc:`ensemble_guide` for comprehensive ensemble documentation including
parallel multi-GPU training.

Windowed Analysis
-----------------

Analyze predictions across genomic windows:

.. code-block:: python

   # Run windowed analysis
   window_predictions = locator.run_windows(
       genotypes=genotypes,
       samples=samples,
       window_size=5e5,  # 500kb windows
       return_df=True,
   )

Jacknife Analysis
-----------------

Assess prediction uncertainty:

.. code-block:: python

   # Run jacknife analysis
   jacknife_predictions = locator.run_jacknife(
       genotypes=genotypes,
       samples=samples,
       prop=0.05,  # Proportion of SNPs to mask
       n_replicates=100,
       return_df=True,
   )

Using Range Masks
-----------------

Incorporate species range constraints:

.. code-block:: python

   # Configure model with range penalty
   config = {
       "out": "range_constrained",
       "use_range_penalty": True,
       "species_range_shapefile": "path/to/range.shp",
       "resolution": 0.05,
       "penalty_weight": 1.0,
   }

   locator = Locator(config)

Memory-Efficient Data Pipeline
------------------------------

Locator uses an efficient ``tf.data`` pipeline by default. ``IndexSet`` handles
train/test/validation splits using index arrays rather than copying genotype
matrices, providing up to 50% memory savings for large datasets.

GPU Configuration
-----------------

Locator includes automatic GPU optimizations that are **enabled by default**.
These provide 3-5x speedup on large datasets.

Basic GPU configuration:

.. code-block:: python

   # GPU optimizations are enabled by default
   config = {
       "out": "gpu_analysis",
       "gpu_number": 0,  # Use first GPU (optional)
   }

   # To disable GPU entirely
   config = {
       "out": "cpu_analysis",
       "disable_gpu": True,
   }

   # To disable specific optimizations
   config = {
       "out": "custom_gpu",
       "use_mixed_precision": False,  # Disable mixed precision
       "gpu_batch_size": 128,         # Use fixed batch size instead of auto
   }

GPU Configuration Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``use_mixed_precision`` (bool, default ``True``)
   Enables FP16 mixed-precision training for approximately 2x speedup on GPUs
   with Tensor Core support (NVIDIA Volta and newer).

``gpu_batch_size`` (``"auto"`` or int, default ``"auto"``)
   Controls training batch size. When set to ``"auto"``, Locator tunes the
   batch size based on available GPU memory. Set to a fixed integer to
   override automatic tuning.

``gpu_memory_mode`` (``"growth"`` or ``"full"``, default ``"growth"``)
   GPU memory allocation strategy. ``"growth"`` allocates memory
   incrementally as needed, which is friendlier to multi-process
   workflows. ``"full"`` pre-allocates all GPU memory for maximum
   throughput.

``enable_xla`` (bool, default ``False``)
   Enables XLA (Accelerated Linear Algebra) JIT compilation. Can improve
   performance for some model architectures, but increases initial
   compilation time.

``gradient_accumulation_steps`` (int, default ``1``)
   Number of forward passes before performing a weight update. Effectively
   simulates a larger batch size without requiring additional GPU memory.
   Useful when GPU memory is limited but a larger effective batch size is
   desired.

Data Augmentation
-----------------

Enable data augmentation during training:

.. code-block:: python

   config = {
       "out": "augmented",
       "augmentation": {
           "enabled": True,
           "flip_rate": 0.05,  # Rate at which to flip genotypes
       },
   }

Handling Missing Coordinates
----------------------------

Locator provides three modes for handling samples with missing coordinates:
``separate`` (default), ``exclude``, and ``fail``. See :doc:`na_handling_guide`
for full details and per-method behavior.

Multi-GPU Parallel Analysis
---------------------------

For large-scale analyses with multiple GPUs, Locator provides Ray-based
parallel implementations of its analysis methods. See
:doc:`parallel_analysis_guide` for comprehensive documentation on multi-GPU
analysis.

Next Steps
----------

* See the :doc:`api` reference for detailed information about all available
  functions and classes.
* Explore :doc:`parallel_analysis_guide` for multi-GPU workflows.
* Learn about visualization in :doc:`plotting_guide`.
* Learn how to contribute in :doc:`contributing`.
