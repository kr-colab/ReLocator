Installation
============

Requirements
-------------

Locator requires Python 3.8 or higher. All dependencies are installed
automatically when you install the package with pip.

Basic Installation
-------------------

The simplest way to install Locator is using pip:

.. code-block:: bash

   pip install locator

Development Installation
-------------------------

For development, you may want to install additional dependencies for testing and documentation:

.. code-block:: bash

   pip install locator[dev,docs]

This will install:

* pytest for testing
* ruff for linting and formatting
* sphinx and related packages for documentation

For detailed API documentation useful during development, see :doc:`api`.

Optional Features
-----------------

Locator provides optional features that require additional dependencies:

Parallel Analysis
~~~~~~~~~~~~~~~~~

For multi-GPU parallel analysis using Ray:

.. code-block:: bash

   pip install locator[ray]

This enables:

* :doc:`parallel_analysis_guide` - Multi-GPU k-fold CV, holdouts, and windowed analysis
* Ray framework for distributed computing
* Automatic GPU load balancing

All Features
~~~~~~~~~~~~

To install Locator with all optional features:

.. code-block:: bash

   pip install locator[dev,docs,ray]

Installing from Source
-----------------------

To install from source:

1. Clone the repository:

   .. code-block:: bash

      git clone https://github.com/yourusername/locator.git
      cd locator

2. Install in development mode:

   .. code-block:: bash

      pip install -e .

Verifying Installation
-----------------------

To verify your installation, run:

.. code-block:: python

   import locator
   print(locator.__version__)

You should see the version number printed without any errors.

Next Steps
----------

* Read the :doc:`usage` guide to learn how to use Locator
* Check the :doc:`api` reference for detailed documentation
