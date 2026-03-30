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
   #    Supports zarr files from both bio2zarr (vcf2zarr) and scikit-allel.
   #    For large VCFs, convert once with bio2zarr for fast subsequent loads:
   #      bcftools index -t genotypes.vcf.gz
   #      vcf2zarr convert -p 8 genotypes.vcf.gz genotypes.zarr
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

Use k-fold cross-validation to train ensemble models for improved predictions:

.. code-block:: python

   # Train 5-fold ensemble
   ensemble_result = locator.train_ensemble(
       genotypes=genotypes,
       samples=samples,
       k=5,  # Number of folds
       save_fold_models=True,
       verbose=True,
   )

   # Get ensemble predictions with uncertainty
   predictions = locator.predict_ensemble(
       genotypes=genotypes,
       samples=samples,
       return_std=True,  # Include prediction uncertainty
   )

For parallel ensemble training across multiple GPUs:

.. code-block:: python

   from locator.parallel import parallel_train_ensemble

   # Train across 4 GPUs
   result = parallel_train_ensemble(
       locator=locator,
       genotypes=genotypes,
       samples=samples,
       k=5,
       gpu_ids=[0, 1, 2, 3],
   )

See :doc:`ensemble_guide` for comprehensive ensemble documentation.

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

Locator provides consistent handling of samples without geographic coordinates
through the ``na_action`` parameter:

.. code-block:: python

   # Configure NA handling behavior
   config = {
       "out": "na_handling_example",
       "na_action": "separate",  # Options: 'separate', 'exclude', 'fail'
   }

   locator = Locator(config)

Available NA Actions
~~~~~~~~~~~~~~~~~~~~

**'separate' (default)**
   Train on samples with known coordinates, predict on samples without
   coordinates. This is the default behavior that allows you to predict
   locations for new samples.

**'exclude'**
   Only use samples with known coordinates. Samples without coordinates are
   filtered out before training or analysis.

**'fail'**
   Raise an error if any samples lack coordinates. Use this to ensure all
   samples have location data.

Checking Data Quality
~~~~~~~~~~~~~~~~~~~~~

Use the ``check_data()`` method to understand your dataset:

.. code-block:: python

   # Check data before analysis
   locator.check_data(genotypes, samples, verbose=True)

   # Output example:
   # ===== Data Summary =====
   # Total samples: 231
   # Samples with coordinates: 211
   # Samples without coordinates: 20
   # Total SNPs: 1000
   #
   # Current NA handling mode: separate
   # - Will train on samples with known locations
   # - Can predict on samples without locations

Method-Level Control
~~~~~~~~~~~~~~~~~~~~

Override the instance-level NA handling for specific methods:

.. code-block:: python

   # Instance configured with 'separate'
   locator = Locator({"na_action": "separate"})

   # Override for a specific analysis
   locator.run_bootstraps(
       genotypes=genotypes,
       samples=samples,
       na_action="exclude",  # Only use samples with coordinates
   )

Important Notes on Holdout Methods
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Holdout-based methods require known coordinates for evaluation:

.. code-block:: python

   # These methods need coordinates to evaluate predictions
   locator.run_holdouts(genotypes, samples)  # 'separate' behaves like 'exclude'
   locator.run_k_fold_holdouts(genotypes, samples)  # Only uses samples with coords

   # Non-holdout methods can predict on NA samples with 'separate' mode
   locator.run_jacknife(genotypes, samples)  # Can predict NA samples
   locator.run_bootstraps(genotypes, samples)  # Can predict NA samples

Multi-GPU Parallel Analysis
---------------------------

For large-scale analyses with multiple GPUs, install the ``[ray]`` extra and
use Locator's parallel implementations (Ray is included in pixi's default
environment, or install with ``pip install locator[ray]``):

.. code-block:: python

   from locator.parallel import parallel_k_fold_holdouts

   # Run k-fold CV across 4 GPUs
   predictions = parallel_k_fold_holdouts(
       locator, genotypes, samples,
       k=10,
       gpu_ids=[0, 1, 2, 3],
       return_df=True,
   )

See :doc:`parallel_analysis_guide` for comprehensive documentation on
multi-GPU analysis.

Next Steps
----------

* See the :doc:`api` reference for detailed information about all available
  functions and classes.
* Explore :doc:`parallel_analysis_guide` for multi-GPU workflows.
* Learn about visualization in :doc:`plotting_guide`.
* Learn how to contribute in :doc:`contributing`.
