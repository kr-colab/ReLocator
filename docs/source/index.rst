Welcome to Locator's documentation!
================================

Locator is a deep learning-based tool for predicting geographic coordinates from genotype matrices. It uses TensorFlow and Keras to build models that can accurately predict the geographic origin of samples based on their genetic data.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   usage
   api
   examples
   contributing

Features
--------

* Deep learning-based coordinate prediction
* Custom loss functions for geographic constraints
* Species range mask integration
* Efficient batch processing
* TensorFlow/Keras implementation
* Comprehensive evaluation metrics

Installation
-----------

To install Locator, use pip:

.. code-block:: bash

   pip install locator

For development installation with documentation tools:

.. code-block:: bash

   pip install locator[dev,docs]

Quick Start
----------

Here's a simple example of using Locator:

.. code-block:: python

   from locator.models import LocatorModel
   from locator.data import load_genotype_data

   # Load your genotype data
   X_train, y_train = load_genotype_data('path/to/data')

   # Create and train the model
   model = LocatorModel()
   model.fit(X_train, y_train)

   # Make predictions
   predictions = model.predict(X_test)

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search` 