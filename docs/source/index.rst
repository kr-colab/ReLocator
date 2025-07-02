Welcome to Locator's documentation!
====================================

Locator is a deep learning-based tool for predicting geographic coordinates from genotype matrices. It uses TensorFlow and Keras to build models that can accurately predict the geographic origin of samples based on their genetic data.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   usage
   data_pipeline_guide
   gpu_optimization_guide
   na_handling_guide
   api
   examples
   contributing

Quick Links
------------

* :doc:`installation` - Installation instructions
* :doc:`usage` - Basic and advanced usage guide
* :doc:`data_pipeline_guide` - Memory-efficient data pipeline
* :doc:`gpu_optimization_guide` - GPU optimization and performance guide
* :doc:`na_handling_guide` - Guide for handling missing coordinates
* :doc:`api` - Complete API reference
* :doc:`examples` - Example workflows
* :doc:`contributing` - Contribution guidelines

Key Features
-------------

* Deep learning-based coordinate prediction
* GPU acceleration with automatic optimization
* Mixed precision training for 2x speedup
* Memory-efficient data pipeline with zero-copy splits
* Custom loss functions for geographic constraints
* Species range mask integration
* Consistent handling of missing coordinates
* Efficient batch processing with tf.data pipeline
* Built-in data augmentation
* TensorFlow/Keras implementation
* Comprehensive evaluation metrics

Getting Started
----------------

1. :doc:`installation` - Install Locator
2. :doc:`usage` - Learn the basics
3. :doc:`api` - Explore the API
4. :doc:`examples` - See examples

For Developers
---------------

* :doc:`contributing` - How to contribute
* :doc:`api` - Detailed API documentation
* :doc:`installation` - Setting up for development

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search` 