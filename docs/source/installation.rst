Installation
============

Requirements
-------------

Locator requires Python 3.8 or higher. The following dependencies are automatically installed:

* TensorFlow >= 2.8.0
* NumPy >= 1.19.2
* Pandas >= 1.2.0
* Scikit-learn >= 0.24.0
* Matplotlib >= 3.3.0

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
* black for code formatting
* flake8 for linting
* sphinx and related packages for documentation

For detailed API documentation useful during development, see :doc:`api`.

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

Troubleshooting
---------------

Common Issues
~~~~~~~~~~~~~

1. TensorFlow GPU Support
   
   If you want to use GPU acceleration, make sure you have the appropriate CUDA and cuDNN versions installed for your TensorFlow version.

2. Memory Issues
   
   For large datasets, you may need to adjust your system's memory settings or use data generators.

Getting Help
~~~~~~~~~~~~

If you encounter any issues during installation:

* Check the troubleshooting section above
* Open an issue on GitHub
* Contact the development team

Next Steps
----------

* Read the :doc:`usage` guide to learn how to use Locator
* Check the :doc:`api` reference for detailed documentation
* See :doc:`examples` for example workflows 