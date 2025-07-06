Parallel Analysis Guide
=======================

Locator provides Ray-based parallel implementations of analysis methods that enable efficient multi-GPU utilization for large-scale cross-validation and analysis tasks.

Overview
--------

The parallel analysis module provides:

* **Multi-GPU k-fold cross-validation** with linear speedup
* **Parallel holdout analysis** across multiple replicates
* **Distributed windowed analysis** for genomic regions
* **Automatic GPU load balancing** across available devices
* **Memory-efficient data serialization** for inter-process communication

When to Use Parallel Analysis
-----------------------------

Use parallel analysis when:

* You have multiple GPUs available
* You have one GPU and you want to run multiple analyses simultaneously
* Running computationally intensive analyses (k-fold CV, many holdout replicates)
* Working with large datasets that benefit from distributed processing
* Need to reduce wall-clock time for cross-validation

Stick with standard (non-parallel) analysis when:

* Using a single CPU or you have a single GPU and your dataset is too large to fit multiple copies in GPU memory
* Running quick analyses or debugging
* Working in interactive environments where Ray might add complexity

Installation
------------

The parallel analysis features require Ray as an additional dependency:

.. code-block:: bash

    # Install with parallel support
    pip install locator[parallel]
    
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

Parallel Analysis Functions
---------------------------

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

**GPU Fraction Settings:**

* ``gpu_fraction=1.0``: One worker per GPU (default, safest)
* ``gpu_fraction=0.5``: Two workers per GPU (moderate sharing)
* ``gpu_fraction=0.25``: Four workers per GPU (maximum parallelism)
* ``gpu_fraction=0.0``: CPU only execution

parallel_leave_one_out
~~~~~~~~~~~~~~~~~~~~~~

Parallel leave-one-out cross-validation (wrapper around k-fold with k=n_samples):

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

Run multiple holdout replicates in parallel:

.. code-block:: python

    from locator.parallel import parallel_holdouts
    
    # Random holdouts
    predictions = parallel_holdouts(
        locator,
        genotypes,
        samples,
        k=20,                    # Samples to hold out
        n_reps=100,             # Number of replicates
        gpu_ids=[0, 1, 2, 3],
        return_df=True
    )
    
    # Specific samples by ID
    predictions = parallel_holdouts(
        locator,
        genotypes,
        samples,
        holdout_sample_ids=['sample_001', 'sample_002', 'sample_003'],
        n_reps=50,
        gpu_ids=[0, 1, 2, 3],
        return_df=True
    )

parallel_windows_holdouts
~~~~~~~~~~~~~~~~~~~~~~~~~

Analyze genomic windows for holdout samples in parallel:

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

Performance Considerations
--------------------------

GPU Memory and Batch Sizes
~~~~~~~~~~~~~~~~~~~~~~~~~~

When using ``gpu_fraction < 1.0``, workers share GPU memory:

.. code-block:: python

    # Conservative: One worker per GPU
    results = parallel_k_fold_holdouts(
        locator, genotypes, samples,
        gpu_ids=[0, 1],
        gpu_fraction=1.0  # Full GPU per worker
    )
    
    # Aggressive: Ten workers per GPU
    # Reduce batch size to fit in shared memory
    locator.config['gpu_batch_size'] = 32  # Smaller batches
    results = parallel_k_fold_holdouts(
        locator, genotypes, samples,
        gpu_ids=[0, 1],
        gpu_fraction=0.1  # Ten workers per GPU
    )

Data Serialization
~~~~~~~~~~~~~~~~~~

The parallel module uses efficient data serialization:

* Genotype arrays are serialized once and shared via temporary files
* Each worker reconstructs the GenotypeArray in its own process
* Sample metadata is included in the serialized data

Ray Configuration
~~~~~~~~~~~~~~~~~

Ray is initialized automatically, but you can configure it:

.. code-block:: python

    import ray
    
    # Initialize Ray with specific resources
    ray.init(
        num_cpus=32,
        num_gpus=4,
        object_store_memory=10_000_000_000  # 10GB object store
    )
    
    # Then run parallel analysis
    results = parallel_k_fold_holdouts(...)
    
    # Shutdown Ray when done
    ray.shutdown()

Example: Multi-GPU K-Fold CV
-----------------------------

Complete example with error analysis:

.. code-block:: python

    import numpy as np
    import pandas as pd
    from locator import Locator
    from locator.parallel import parallel_k_fold_holdouts
    
    # Configuration
    config = {
        "out": "multi_gpu_cv",
        "sample_data": "samples.tsv",
        "width": 256,
        "nlayers": 10,
        "dropout_prop": 0.25,
        "batch_size": 64
    }
    
    # Initialize and load data
    locator = Locator(config)
    genotypes, samples = locator.load_genotypes(zarr="genotypes.zarr")
    
    # Run 10-fold CV across 4 GPUs
    print("Running parallel 10-fold cross-validation...")
    predictions = parallel_k_fold_holdouts(
        locator,
        genotypes,
        samples,
        k=10,
        gpu_ids=[0, 1, 2, 3],  # Use 4 GPUs
        gpu_fraction=1.0,      # One fold per GPU at a time
        return_df=True,
        verbose=True
    )
    
    # Use plot_error_summary for comprehensive error analysis
    from locator.plotting import plot_error_summary
    
    # Create error visualization with statistics
    plot_error_summary(
        predictions,
        "samples.tsv",
        out_prefix="multi_gpu_cv_errors",
        plot_map=True,      # Show geographic distribution
        include_training_locs=True  # Show training context
    )
    
    # The plot automatically calculates and displays:
    # - Mean, median, and max error
    # - R² values for x and y coordinates
    # - Error distribution histogram
    # - Geographic error patterns
    
    # Save predictions for further analysis
    predictions.to_csv("kfold_cv_predictions.csv", index=False)

Example: Parallel Windowed Analysis
-----------------------------------

Analyze prediction accuracy across genomic windows:

.. code-block:: python

    from locator.parallel import parallel_windows_holdouts
    
    # Configuration for windowed analysis
    config = {
        "out": "window_analysis",
        "sample_data": "samples.tsv",
        "min_snps_per_window": 100  # Require at least 100 SNPs
    }
    
    locator = Locator(config)
    genotypes, samples = locator.load_genotypes(zarr="genotypes.zarr")
    
    # Run windowed analysis on worst-performing samples
    # First identify them from previous k-fold results
    worst_samples = ['HG001', 'HG002', 'HG003']  # Example IDs
    
    window_results = parallel_windows_holdouts(
        locator,
        genotypes,
        samples,
        holdout_sample_ids=worst_samples,
        window_size=int(1e6),  # 1Mb windows
        respect_chromosomes=True,
        gpu_ids=[0, 1, 2, 3],
        return_df=True,
        verbose=True
    )
    
    # Analyze window performance
    # Results contain predictions for each window
    print(f"Analyzed {len(window_results.columns)-1} windows")

Troubleshooting
---------------

Common Issues
~~~~~~~~~~~~~

**Ray initialization errors:**

.. code-block:: python

    # If Ray is already initialized
    ray.shutdown()
    
    # Reinitialize with specific configuration
    ray.init(ignore_reinit_error=True)

**GPU memory errors with multiple workers:**

.. code-block:: python

    # Reduce workers per GPU
    results = parallel_k_fold_holdouts(
        locator, genotypes, samples,
        gpu_fraction=1.0  # Use full GPU per worker
    )
    
    # Or reduce batch size
    locator.config['gpu_batch_size'] = 32

**Slow data serialization:**

Large datasets are serialized to temporary files. Ensure fast local storage:

.. code-block:: python

    # Set Ray temp directory to fast SSD
    import os
    os.environ['RAY_TMPDIR'] = '/fast/ssd/ray_tmp'

Performance Tips
~~~~~~~~~~~~~~~~

1. **Use full GPUs for memory-intensive models:**
   
   .. code-block:: python
   
       gpu_fraction=1.0  # Default and recommended

2. **Pre-calculate bandwidth for KDE weights:**
   
   The parallel functions automatically handle bandwidth pre-calculation
   when using KDE sample weighting.

3. **Monitor GPU utilization:**
   
   .. code-block:: bash
   
       # In another terminal
       watch -n 1 nvidia-smi

4. **Adjust based on model size:**
   
   * Small models (width≤128): Can use gpu_fraction=0.5
   * Large models (width≥512): Use gpu_fraction=1.0
   * Very large models: May need to reduce batch size

API Comparison
--------------

The parallel API mirrors the standard analysis API:

.. list-table:: API Comparison
   :header-rows: 1
   :widths: 50 50

   * - Standard API
     - Parallel API
   * - ``locator.run_k_fold_holdouts()``
     - ``parallel_k_fold_holdouts()``
   * - ``locator.run_leave_one_out()``
     - ``parallel_leave_one_out()``
   * - ``locator.run_holdouts()``
     - ``parallel_holdouts()``
   * - ``locator.run_windows_holdouts()``
     - ``parallel_windows_holdouts()``

Key differences:

* Parallel functions take ``locator`` as first argument
* Add ``gpu_ids`` and ``gpu_fraction`` parameters
* Return same output format as standard functions
* Require Ray installation

Best Practices
--------------

1. **Start with conservative settings:**
   
   Begin with ``gpu_fraction=1.0`` and adjust based on GPU memory usage.

2. **Use appropriate parallelism level:**
   
   * K-fold CV: Parallelize across folds
   * Many replicates: Parallelize across replicates
   * Few large tasks: Consider ``gpu_fraction < 1.0``

3. **Monitor and profile:**
   
   .. code-block:: python
   
       import time
       
       start = time.time()
       results = parallel_k_fold_holdouts(...)
       elapsed = time.time() - start
       
       print(f"Parallel: {elapsed:.1f}s")
       print(f"Theoretical sequential: {elapsed * len(gpu_ids):.1f}s")
       print(f"Speedup: {len(gpu_ids) * elapsed / elapsed:.1f}x")

4. **Clean up resources:**
   
   .. code-block:: python
   
       # After analysis
       ray.shutdown()

Future Enhancements
-------------------

Planned improvements to parallel analysis:

* Distributed computing support for cluster environments
* Shared memory optimization for very large datasets
* Automatic GPU selection based on availability
* Integration with Dask for CPU-parallel preprocessing
* Real-time progress monitoring dashboard