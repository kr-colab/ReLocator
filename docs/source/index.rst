Welcome to Locator's documentation!
====================================

Locator is a deep learning-based tool for predicting geographic coordinates from genotype matrices. It uses TensorFlow and Keras to build models that can accurately predict the geographic origin of samples based on their genetic data.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   usage
   cli
   ensemble_guide
   pca_guide
   parallel_analysis_guide
   plotting_guide
   na_handling_guide
   api
   contributing

Quick Links
------------

* :doc:`installation` - Installation instructions
* :doc:`usage` - Basic and advanced usage guide
* :doc:`cli` - Command-line interface
* :doc:`ensemble_guide` - Ensemble models and k-fold cross-validation
* :doc:`pca_guide` - The PCA model for data with many SNPs
* :doc:`parallel_analysis_guide` - Multi-GPU parallel analysis guide
* :doc:`plotting_guide` - Visualization and plotting guide
* :doc:`na_handling_guide` - Guide for handling missing coordinates
* :doc:`api` - Complete API reference
* :doc:`contributing` - Contribution guidelines

Key Features
-------------

* Deep learning-based coordinate prediction
* GPU acceleration with automatic optimization
* Mixed precision training for 2x speedup
* Memory-efficient data pipeline with zero-copy splits
* Multi-GPU parallel analysis with Ray
* Species range mask integration
* Consistent handling of missing coordinates
* Built-in data augmentation

Getting Started
----------------

1. :doc:`installation` - Install Locator
2. :doc:`usage` - Learn the basics
3. :doc:`api` - Explore the API

For Developers
---------------

* :doc:`contributing` - How to contribute
* :doc:`api` - Detailed API documentation

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
