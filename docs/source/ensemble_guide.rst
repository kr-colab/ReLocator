Ensemble Models Guide
=====================

This guide covers using ensemble models in Locator for improved prediction accuracy through k-fold cross-validation.

.. contents:: Table of Contents
   :local:
   :depth: 2

Overview
--------

Ensemble models combine predictions from multiple neural networks trained on different subsets of data to provide:

- **Improved accuracy** through averaging multiple predictions
- **Uncertainty estimates** via prediction variance across models
- **Robustness** to overfitting on small datasets
- **Better generalization** to new samples

The ensemble functionality is integrated directly into the main ``Locator`` class through the ``EnsembleMixin``.

Basic Ensemble Training
-----------------------

Sequential Training
~~~~~~~~~~~~~~~~~~~

Train an ensemble using k-fold cross-validation:

.. code-block:: python

   from locator import Locator

   # Configure Locator
   config = {
       "out": "ensemble_analysis",
       "batch_size": 32,
       "width": 256,
       "nlayers": 8,
       "dropout_prop": 0.25,
       "max_epochs": 1000,
       "patience": 100
   }

   locator = Locator(config)

   # Load your data
   genotypes, samples = locator.load_genotypes(vcf="genotypes.vcf.gz")

   # Train 5-fold ensemble
   ensemble_result = locator.train_ensemble(
       genotypes=genotypes,
       samples=samples,
       k=5,  # Number of folds
       save_fold_models=True,  # Save individual fold models
       use_model_manager=True,  # Use efficient model storage
       verbose=True
   )

   # The result contains:
   # - histories: Training history for each fold
   # - models: Model information for each fold
   # - normalization_params: Averaged normalization parameters
   # - fold_info: Details about fold splits

Parallel Training (Multi-GPU)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For faster training across multiple GPUs:

.. code-block:: python

   from locator.parallel import parallel_train_ensemble

   # Train ensemble across 4 GPUs
   ensemble_result = parallel_train_ensemble(
       locator=locator,
       genotypes=genotypes,
       samples=samples,
       k=5,
       gpu_ids=[0, 1, 2, 3],  # Use GPUs 0-3
       save_fold_models=True,
       use_model_manager=True,
       verbose=True
   )

   # CPU-only mode
   ensemble_result = parallel_train_ensemble(
       locator=locator,
       genotypes=genotypes,
       samples=samples,
       k=5,
       gpu_ids=[],  # Empty list for CPU mode
       verbose=True
   )

Making Predictions
------------------

Basic Predictions
~~~~~~~~~~~~~~~~~

After training an ensemble, make predictions:

.. code-block:: python

   # Make ensemble predictions
   predictions = locator.predict_ensemble(
       genotypes=genotypes,
       samples=samples,
       return_std=True,  # Include uncertainty estimates
       include_fold_predictions=False,  # Just return ensemble mean
       save_predictions=True
   )

   # Result is a DataFrame with columns:
   # - sampleID: Sample identifier
   # - x: Predicted longitude (ensemble mean)
   # - y: Predicted latitude (ensemble mean)
   # - x_std: Longitude standard deviation across folds
   # - y_std: Latitude standard deviation across folds

Detailed Predictions
~~~~~~~~~~~~~~~~~~~~

Get individual fold predictions for analysis:

.. code-block:: python

   # Get predictions from each fold
   detailed_predictions = locator.predict_ensemble(
       genotypes=genotypes,
       samples=samples,
       return_std=True,
       include_fold_predictions=True,  # Include per-fold predictions
       save_predictions=True
   )

   # Additional columns:
   # - x_fold0, y_fold0: Predictions from fold 0
   # - x_fold1, y_fold1: Predictions from fold 1
   # - ... for all k folds

Advanced Features
-----------------

Training Optimizations
~~~~~~~~~~~~~~~~~~~~~~

The ensemble training includes several optimizations:

.. code-block:: python

   # Enable mixed precision training (automatic GPU detection)
   ensemble_result = locator.train_ensemble(
       genotypes=genotypes,
       samples=samples,
       k=5,
       use_mixed_precision=None,  # Auto-detect GPU capability
       patience_multiplier=1.5,   # Increase patience for ensemble
       verbose=True
   )

Data Augmentation
~~~~~~~~~~~~~~~~~

Apply data augmentation during ensemble training:

.. code-block:: python

   ensemble_result = locator.train_ensemble(
       genotypes=genotypes,
       samples=samples,
       k=5,
       augment_data=True,
       flip_rate=0.05,  # Flip 5% of genotypes
       verbose=True
   )

Partial Training Sets
~~~~~~~~~~~~~~~~~~~~~

Train on a subset of samples:

.. code-block:: python

   # Only use specific samples for training
   training_indices = [0, 1, 2, 5, 10, 15, 20]  # Sample indices

   ensemble_result = locator.train_ensemble(
       genotypes=genotypes,
       samples=samples,
       k=5,
       training_set_indices=training_indices,
       verbose=True
   )

Model Persistence
-----------------

Saving Ensemble Models
~~~~~~~~~~~~~~~~~~~~~~

Ensemble models are automatically saved using the ``EnsembleModelManager``:

.. code-block:: python

   # Models are saved to: {out}_ensemble/
   # - metadata.json: Ensemble configuration
   # - fold_0_model.json: Fold 0 architecture
   # - fold_0_weights.h5: Fold 0 weights
   # - fold_0_norm_params.json: Fold 0 normalization
   # - ... for all folds

Loading Ensemble Models
~~~~~~~~~~~~~~~~~~~~~~~

Load a previously trained ensemble:

.. code-block:: python

   # Load ensemble from disk
   locator = Locator(config)
   ensemble_info = locator.load_ensemble("ensemble_analysis_ensemble")

   # Make predictions with loaded ensemble
   predictions = locator.predict_ensemble_from_manager(
       genotypes=genotypes,
       samples=samples,
       save_predictions=True
   )

Memory-Efficient Prediction
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The model manager loads models on-demand to reduce memory usage:

.. code-block:: python

   # Models are loaded one at a time during prediction
   # This is handled automatically by predict_ensemble_from_manager()

Performance Considerations
--------------------------

Choosing k (Number of Folds)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **k=5**: Good balance of training data and ensemble diversity (default)
- **k=10**: More models but each trained on less data
- **k=3**: Faster training but less ensemble benefit
- **Large datasets**: Can use smaller k (3-5)
- **Small datasets**: May benefit from larger k (5-10)

GPU Utilization
~~~~~~~~~~~~~~~

For parallel training:

.. code-block:: python

   # Configure GPU usage
   parallel_train_ensemble(
       locator=locator,
       genotypes=genotypes,
       samples=samples,
       k=8,  # 8 folds
       gpu_ids=[0, 1, 2, 3],  # 4 GPUs
       gpu_fraction=0.5,  # Allow 2 models per GPU
       verbose=True
   )

   # With gpu_fraction=0.5 and 4 GPUs:
   # - Can train 8 models simultaneously
   # - Each model uses 50% of a GPU's memory

Pre-calculated Bandwidth
~~~~~~~~~~~~~~~~~~~~~~~~

For datasets using KDE sample weighting:

.. code-block:: python

   # Bandwidth is automatically pre-calculated once
   # and reused for all folds, saving computation time

   config = {
       "weight_samples": {
           "enabled": True,
           "method": "KD",
           "n_bandwidths": 100  # Grid points for optimization
       }
   }

Integration with Analysis Methods
---------------------------------

Ensemble with NA Handling
~~~~~~~~~~~~~~~~~~~~~~~~~

Combine ensemble training with NA sample handling:

.. code-block:: python

   # Configure NA handling
   config = {
       "out": "ensemble_na",
       "na_action": "separate"  # Train on known, predict on unknown
   }

   locator = Locator(config)

   # Train ensemble (only uses samples with coordinates)
   ensemble_result = locator.train_ensemble(
       genotypes=genotypes,
       samples=samples,
       k=5,
       na_action="separate",  # Can override instance setting
       verbose=True
   )

   # Predict on all samples (including those without coordinates)
   predictions = locator.predict_ensemble(
       genotypes=genotypes,
       samples=samples,
       return_std=True
   )

Example Workflow
----------------

Complete ensemble analysis workflow:

.. code-block:: python

   import numpy as np
   from locator import Locator
   from locator.parallel import parallel_train_ensemble
   from locator.plotting import plot_predictions

   # 1. Setup
   config = {
       "out": "species_ensemble",
       "sample_data": "samples.tsv",
       "batch_size": 32,
       "max_epochs": 1000,
       "patience": 100,
       "na_action": "separate"
   }

   locator = Locator(config)

   # 2. Load data
   genotypes, samples = locator.load_genotypes(vcf="genotypes.vcf.gz")
   locator.check_data(genotypes, samples)

   # 3. Train ensemble (parallel if multiple GPUs available)
   if len(locator.get_available_gpus()) > 1:
       result = parallel_train_ensemble(
           locator=locator,
           genotypes=genotypes,
           samples=samples,
           k=5,
           gpu_ids=[0, 1],  # Use 2 GPUs
           verbose=True
       )
   else:
       result = locator.train_ensemble(
           genotypes=genotypes,
           samples=samples,
           k=5,
           verbose=True
       )

   # 4. Make predictions
   predictions = locator.predict_ensemble(
       genotypes=genotypes,
       samples=samples,
       return_std=True,
       save_predictions=True
   )

   # 5. Analyze results
   # Calculate prediction uncertainty (mean std dev)
   mean_uncertainty = np.mean([
       predictions['x_std'].mean(),
       predictions['y_std'].mean()
   ])
   print(f"Mean prediction uncertainty: {mean_uncertainty:.2f} degrees")

   # 6. Visualize
   plot_predictions(
       predictions=predictions,
       locator=locator,
       out_prefix="species_ensemble_predictions",
       plot_error=True,  # Show prediction uncertainty
       plot_border=True
   )

   # 7. Save high-confidence predictions
   # (low uncertainty samples)
   high_conf = predictions[
       (predictions['x_std'] < mean_uncertainty) &
       (predictions['y_std'] < mean_uncertainty)
   ]
   high_conf.to_csv("high_confidence_predictions.csv", index=False)

See Also
--------

- :doc:`parallel_analysis_guide` - Multi-GPU parallel analysis
- :doc:`api` - Complete API reference
- :doc:`examples` - More usage examples
- :doc:`na_handling_guide` - Handling missing coordinates
