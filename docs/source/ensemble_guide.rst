Ensemble Models Guide
=====================

Ensemble models combine predictions from multiple neural networks trained on
different k-fold subsets of the data. Averaging across folds improves accuracy,
reduces overfitting, and provides per-sample uncertainty estimates. The
functionality is integrated into the ``Locator`` class through the
``EnsembleMixin``.

.. contents:: Table of Contents
   :local:
   :depth: 2

Basic Ensemble Training
-----------------------

Sequential Training
~~~~~~~~~~~~~~~~~~~

Train an ensemble using k-fold cross-validation:

.. code-block:: python

   from locator import Locator

   config = {
       "out": "ensemble_analysis",
       "batch_size": 32,
       "width": 256,
       "nlayers": 8,
       "dropout_prop": 0.25,
       "max_epochs": 1000,
       "patience": 100,
   }

   locator = Locator(config)
   genotypes, samples = locator.load_genotypes(vcf="genotypes.vcf.gz")

   ensemble_result = locator.train_ensemble(
       genotypes=genotypes,
       samples=samples,
       k=5,
       save_fold_models=True,
       use_model_manager=True,
       verbose=True,
   )

The returned ``ensemble_result`` contains training histories, model
information, averaged normalization parameters, and fold split details.

Parallel Training (Multi-GPU)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For faster training across multiple GPUs:

.. code-block:: python

   from locator.parallel import parallel_train_ensemble

   ensemble_result = parallel_train_ensemble(
       locator=locator,
       genotypes=genotypes,
       samples=samples,
       k=5,
       gpu_ids=[0, 1, 2, 3],
       save_fold_models=True,
       use_model_manager=True,
       verbose=True,
   )

Pass an empty list for CPU-only mode:

.. code-block:: python

   ensemble_result = parallel_train_ensemble(
       locator=locator,
       genotypes=genotypes,
       samples=samples,
       k=5,
       gpu_ids=[],
       verbose=True,
   )

Making Predictions
------------------

After training, call ``predict_ensemble`` to get averaged predictions across
folds. Set ``return_std=True`` for per-sample uncertainty and
``include_fold_predictions=True`` for individual fold outputs:

.. code-block:: python

   predictions = locator.predict_ensemble(
       genotypes=genotypes,
       samples=samples,
       return_std=True,
       include_fold_predictions=True,
       save_predictions=True,
   )

The result is a DataFrame with columns:

- ``sampleID``: sample identifier
- ``x``, ``y``: ensemble-mean longitude and latitude
- ``x_std``, ``y_std``: standard deviation across folds
- ``x_fold0``, ``y_fold0``, ...: per-fold predictions (when
  ``include_fold_predictions=True``)

Advanced Features
-----------------

Training Optimizations
~~~~~~~~~~~~~~~~~~~~~~

Mixed precision is auto-detected when a compatible GPU is present:

.. code-block:: python

   ensemble_result = locator.train_ensemble(
       genotypes=genotypes,
       samples=samples,
       k=5,
       use_mixed_precision=None,
       patience_multiplier=1.5,
       verbose=True,
   )

Data Augmentation
~~~~~~~~~~~~~~~~~

.. code-block:: python

   ensemble_result = locator.train_ensemble(
       genotypes=genotypes,
       samples=samples,
       k=5,
       augment_data=True,
       flip_rate=0.05,
       verbose=True,
   )

Partial Training Sets
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   training_indices = [0, 1, 2, 5, 10, 15, 20]

   ensemble_result = locator.train_ensemble(
       genotypes=genotypes,
       samples=samples,
       k=5,
       training_set_indices=training_indices,
       verbose=True,
   )

Model Persistence
-----------------

Saving
~~~~~~

When ``save_fold_models=True`` and ``use_model_manager=True``, models are
written to ``{out}_ensemble/`` automatically:

::

   ensemble_analysis_ensemble/
       metadata.json
       fold_0_model.json
       fold_0_weights.h5
       fold_0_norm_params.json
       ...

Loading
~~~~~~~

Reload a saved ensemble and predict:

.. code-block:: python

   locator = Locator(config)
   locator.load_ensemble("ensemble_analysis_ensemble")

   predictions = locator.predict_ensemble_from_manager(
       genotypes=genotypes,
       samples=samples,
       save_predictions=True,
   )

Models are loaded one at a time during prediction, so memory usage stays low
even for large ensembles.

See Also
--------

- :doc:`parallel_analysis_guide` -- Multi-GPU parallel analysis
- :doc:`api` -- Complete API reference
- :doc:`na_handling_guide` -- Handling missing coordinates
