Parallel Analysis Guide
=======================

Locator provides Ray-based parallel implementations of its analysis methods,
enabling efficient multi-GPU utilization for cross-validation and holdout
analyses. The parallel API mirrors the standard analysis API: parallel
functions accept a ``Locator`` instance as their first argument and add
``gpu_ids`` and ``gpu_fraction`` parameters. Return formats are identical
to the standard methods.

When to Use Parallel Analysis
-----------------------------

Use parallel analysis when you have multiple GPUs or want to run several
analyses concurrently on a single GPU via fractional allocation. For
single-GPU work with datasets that fill GPU memory, or for quick debugging
runs, the standard (non-parallel) analysis methods are simpler.

Installation
------------

The parallel analysis features require Ray as an additional dependency:

.. code-block:: bash

    # Install with parallel support
    pip install locator[ray]

    # Or install Ray separately
    pip install ray>=2.0.0

Quick Start
-----------

Basic parallel k-fold cross-validation:

.. code-block:: python

    from locator import Locator
    from locator.parallel import parallel_k_fold_holdouts

    # Initialize Locator
    locator = Locator({"out": "parallel_analysis"})

    # Load data
    genotypes, samples = locator.load_genotypes(zarr="genotypes.zarr")

    # Run parallel k-fold CV across 4 GPUs
    predictions = parallel_k_fold_holdouts(
        locator, genotypes, samples,
        k=10,
        gpu_ids=[0, 1, 2, 3],
        return_df=True
    )

Analysis Functions
------------------

parallel_k_fold_holdouts
~~~~~~~~~~~~~~~~~~~~~~~~

Run true k-fold cross-validation in parallel across multiple GPUs.

.. code-block:: python

    from locator.parallel import parallel_k_fold_holdouts

    predictions = parallel_k_fold_holdouts(
        locator,
        genotypes,
        samples,
        k=10,                     # Number of folds
        gpu_ids=[0, 1, 2, 3],    # GPUs to use
        gpu_fraction=1.0,        # GPU fraction per worker
        return_df=True,          # Return DataFrame
        verbose=True,            # Show progress
        na_action=None           # NA handling mode
    )

parallel_leave_one_out
~~~~~~~~~~~~~~~~~~~~~~

Parallel leave-one-out cross-validation (wrapper around k-fold with k=n_samples).

.. code-block:: python

    from locator.parallel import parallel_leave_one_out

    predictions = parallel_leave_one_out(
        locator,
        genotypes,
        samples,
        gpu_ids=[0, 1, 2, 3],
        gpu_fraction=1.0,
        return_df=True
    )

parallel_holdouts
~~~~~~~~~~~~~~~~~

Run multiple holdout replicates in parallel.

.. code-block:: python

    from locator.parallel import parallel_holdouts

    predictions = parallel_holdouts(
        locator,
        genotypes,
        samples,
        k=20,                    # Samples to hold out
        n_reps=100,             # Number of replicates
        gpu_ids=[0, 1, 2, 3],
        return_df=True
    )

parallel_windows_holdouts
~~~~~~~~~~~~~~~~~~~~~~~~~

Analyze genomic windows for holdout samples in parallel.

.. code-block:: python

    from locator.parallel import parallel_windows_holdouts

    window_predictions = parallel_windows_holdouts(
        locator,
        genotypes,
        samples,
        k=10,                    # Samples to hold out
        window_size=int(5e5),    # 500kb windows
        respect_chromosomes=True,
        gpu_ids=[0, 1, 2, 3],
        return_df=True
    )

GPU Fraction Settings
---------------------

The ``gpu_fraction`` parameter controls how many workers share each GPU.
Use a lower fraction to run more folds concurrently; use a higher fraction
when models are large or GPU memory is limited.

* ``gpu_fraction=1.0`` -- One worker per GPU (default, safest)
* ``gpu_fraction=0.5`` -- Two workers per GPU (moderate sharing)
* ``gpu_fraction=0.25`` -- Four workers per GPU (maximum parallelism)
* ``gpu_fraction=0.0`` -- CPU-only execution

When using ``gpu_fraction < 1.0``, you may need to reduce
``locator.config['gpu_batch_size']`` so that concurrent workers fit in
shared GPU memory.

Parallel Ensemble Training
--------------------------

Train ensemble models across multiple GPUs:

.. code-block:: python

    from locator import Locator
    from locator.parallel import parallel_train_ensemble

    config = {
        "out": "ensemble_analysis",
        "sample_data": "samples.tsv",
        "width": 256,
        "nlayers": 10,
        "max_epochs": 1000,
        "patience": 100
    }

    locator = Locator(config)
    genotypes, samples = locator.load_genotypes(vcf="genotypes.vcf.gz")

    # Train 5-fold ensemble across 4 GPUs
    ensemble_result = parallel_train_ensemble(
        locator=locator,
        genotypes=genotypes,
        samples=samples,
        k=5,
        gpu_ids=[0, 1, 2, 3],
        save_fold_models=True,
        use_model_manager=True,
        verbose=True
    )

    # Make ensemble predictions
    predictions = locator.predict_ensemble(
        genotypes=genotypes,
        samples=samples,
        return_std=True  # Include uncertainty estimates
    )
