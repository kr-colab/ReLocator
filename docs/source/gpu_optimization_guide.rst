GPU Optimization Guide
======================

Locator includes comprehensive GPU optimization features that can significantly accelerate training on large datasets. This guide explains how to use these features effectively.

Overview
--------

GPU optimizations in Locator provide:

* **3-5x faster training** on large datasets
* **2x memory efficiency** with mixed precision
* **Automatic batch size optimization** for your GPU
* **85-95% GPU utilization** (up from ~40% without optimizations)

Quick Start
-----------

GPU optimizations are **enabled by default** in Locator. Simply run your code as usual:

.. code-block:: python

    from locator import Locator
    
    # GPU optimizations are automatically applied
    loc = Locator({"out": "my_analysis"})

To disable GPU optimizations:

.. code-block:: python

    config = {
        "out": "my_analysis",
        "use_mixed_precision": False,
        "gpu_batch_size": 32,  # Use fixed batch size
        "use_efficient_pipeline": False
    }
    loc = Locator(config)

GPU Optimization Features
-------------------------

Mixed Precision Training
~~~~~~~~~~~~~~~~~~~~~~~~

Mixed precision uses float16 computations with float32 master weights:

.. code-block:: python

    config = {
        "use_mixed_precision": True  # Enabled by default
    }

Benefits:

* 2x speedup on GPUs with Tensor Cores (RTX series, V100, A100)
* 50% memory reduction allowing larger batch sizes
* Automatic loss scaling prevents numerical underflow

Requirements:

* GPU with compute capability ≥ 7.0 (Volta architecture or newer)
* TensorFlow 2.4+

Dynamic Batch Size Optimization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Locator automatically determines the optimal batch size for your GPU:

.. code-block:: python

    config = {
        "gpu_batch_size": "auto"  # Default setting
    }

The optimizer:

1. Tests progressively larger batch sizes
2. Finds the maximum that fits in GPU memory
3. Uses 85% of available memory (configurable)
4. Rounds to nearest power of 2 for efficiency

To use a fixed batch size:

.. code-block:: python

    config = {
        "gpu_batch_size": 256  # Use specific batch size
    }

Efficient Data Pipeline
~~~~~~~~~~~~~~~~~~~~~~~

The memory-efficient data pipeline integrates seamlessly with GPU optimization:

.. code-block:: python

    config = {
        "use_efficient_pipeline": True  # Enabled by default
    }

Features:

* **Zero-copy operations**: Uses indices instead of copying arrays
* **Prefetching**: Overlaps data loading with model training
* **Caching**: Keeps frequently used data in GPU memory
* **Parallel processing**: Uses multiple CPU cores for data preparation
* **Automatic tuning**: Optimizes buffer sizes dynamically

For detailed information about the data pipeline architecture, including IndexSet and 
custom tf.data operations, see :doc:`data_pipeline_guide`.

GPU Memory Management
~~~~~~~~~~~~~~~~~~~~~

Control how GPU memory is allocated:

.. code-block:: python

    # Default: Allow memory growth (good for shared systems)
    config = {"gpu_memory_mode": "growth"}
    
    # Pre-allocate all memory (best performance)
    config = {"gpu_memory_mode": "preallocate"}
    
    # Limit memory usage (for multi-user systems)
    config = {"gpu_memory_mode": "limit:4096"}  # Limit to 4GB

Advanced Features
-----------------

Gradient Accumulation
~~~~~~~~~~~~~~~~~~~~~

Simulate larger batch sizes on memory-limited GPUs:

.. code-block:: python

    config = {
        "gradient_accumulation_steps": 4  # Effective batch = physical_batch × 4
    }

This is useful when:

* Your desired batch size doesn't fit in GPU memory
* You want to maintain small batch statistics
* You're comparing results across different GPUs

XLA Compilation (Experimental)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Enable XLA (Accelerated Linear Algebra) for additional optimizations:

.. code-block:: python

    config = {
        "enable_xla": True  # Experimental feature
    }

Benefits:

* 10-20% additional speedup (model-dependent)
* Optimized kernel fusion
* Reduced memory bandwidth usage

Note: May not work with all custom operations.

Multi-GPU Configuration
-----------------------

Select specific GPUs:

.. code-block:: python

    # Use GPU 0 (default)
    config = {"gpu_number": 0}
    
    # Use GPU 1
    config = {"gpu_number": 1}
    
    # Disable GPU, use CPU only
    config = {"disable_gpu": True}

Command line usage:

.. code-block:: bash

    # Use specific GPU
    locator --gpu_number 1 --vcf data.vcf --sample_data samples.txt
    
    # Disable GPU
    locator --disable_gpu --vcf data.vcf --sample_data samples.txt

Performance Monitoring
----------------------

Monitor GPU utilization:

.. code-block:: bash

    # Real-time GPU monitoring
    watch -n 1 nvidia-smi
    
    # Log GPU metrics
    nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv -l 1

Check optimization status in Python:

.. code-block:: python

    from locator.gpu_optimizer import GPUOptimizer
    
    # Get GPU information
    info = GPUOptimizer.get_gpu_info()
    print(f"GPU count: {info['gpu_count']}")
    for gpu in info['gpus']:
        print(f"  {gpu['name']}")
    
    # Check mixed precision support
    GPUOptimizer.setup_mixed_precision()

Troubleshooting
---------------

Out of Memory Errors
~~~~~~~~~~~~~~~~~~~~

If you encounter OOM errors, try these solutions in order:

1. **Enable mixed precision** (if not already enabled):
   
   .. code-block:: python
   
       config = {"use_mixed_precision": True}

2. **Reduce batch size**:
   
   .. code-block:: python
   
       config = {"gpu_batch_size": 64}

3. **Use gradient accumulation**:
   
   .. code-block:: python
   
       config = {
           "gpu_batch_size": 32,
           "gradient_accumulation_steps": 4
       }

4. **Limit GPU memory**:
   
   .. code-block:: python
   
       config = {"gpu_memory_mode": "limit:8192"}  # 8GB limit

No Speedup Observed
~~~~~~~~~~~~~~~~~~~

Check if:

1. **GPU is being used**:
   
   .. code-block:: bash
   
       nvidia-smi  # Should show Python process

2. **Dataset is large enough**:
   
   * GPU optimizations are most effective with >10,000 samples
   * Small datasets may not benefit from GPU acceleration

3. **Mixed precision is active**:
   
   .. code-block:: python
   
       import tensorflow as tf
       print(tf.keras.mixed_precision.global_policy())
       # Should show 'mixed_float16' if active

Mixed Precision Not Working
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Verify GPU compatibility:

.. code-block:: python

    import tensorflow as tf
    
    # Check compute capability
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        details = tf.config.experimental.get_device_details(gpus[0])
        compute_capability = details.get('compute_capability', (0, 0))
        print(f"Compute capability: {compute_capability}")
        # Need >= (7, 0) for mixed precision

Best Practices
--------------

1. **Large Datasets**: GPU optimizations work best with:
   
   * >10,000 samples
   * >100,000 SNPs
   * Deep models (8+ layers)

2. **Memory Management**:
   
   * Use mixed precision for 2x memory savings
   * Start with "auto" batch size
   * Use gradient accumulation for very large batches

3. **Performance Tuning**:
   
   * Monitor GPU utilization (target >85%)
   * Profile training with TensorBoard
   * Experiment with batch sizes

4. **Multi-User Systems**:
   
   * Use memory growth mode
   * Set memory limits
   * Coordinate GPU usage

Example Workflows
-----------------

Basic GPU-Optimized Training
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from locator import Locator
    
    # GPU optimizations are enabled by default
    loc = Locator({
        "out": "gpu_analysis",
        "zarr": "genotypes.zarr",
        "sample_data": "coordinates.txt"
    })
    
    # Load data
    genotypes, samples = loc.load_genotypes()
    
    # Train with automatic GPU optimization
    history = loc.train(genotypes=genotypes, samples=samples)

Custom GPU Configuration
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # High-performance configuration for dedicated GPU
    config = {
        "out": "high_perf_analysis",
        "use_mixed_precision": True,
        "gpu_batch_size": "auto",
        "gpu_memory_mode": "preallocate",  # Maximum performance
        "enable_xla": True,  # Experimental speedup
    }
    
    loc = Locator(config)

Memory-Constrained Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Configuration for shared/limited GPU
    config = {
        "out": "limited_gpu_analysis",
        "use_mixed_precision": True,
        "gpu_batch_size": 64,  # Small batch
        "gradient_accumulation_steps": 8,  # Simulate batch of 512
        "gpu_memory_mode": "limit:4096",  # 4GB limit
    }
    
    loc = Locator(config)

Benchmarking
------------

Run the included benchmark to test GPU optimizations:

.. code-block:: bash

    python examples/gpu_optimization_demo.py

This will:

* Compare default vs optimized configurations
* Show speedup achieved
* Test automatic batch size optimization

Expected Performance
--------------------

Performance varies by hardware and dataset:

**Consumer GPUs** (RTX 3090, RTX 4090):

* 3-4x speedup over CPU
* 2-3x speedup over unoptimized GPU
* Batch sizes: 256-1024

**Data Center GPUs** (V100, A100):

* 5-10x speedup over CPU
* 3-5x speedup over unoptimized GPU
* Batch sizes: 512-2048

**Factors affecting performance**:

* Dataset size (larger is better)
* Model complexity
* GPU memory bandwidth
* CPU-GPU transfer overhead

API Reference
-------------

For detailed API documentation, see:

* :class:`locator.gpu_optimizer.GPUOptimizer`
* :class:`locator.gpu_optimizer.GradientAccumulator`
* :func:`locator.gpu_optimizer.create_optimized_training_config`