Installation
============

Requirements
-------------

Locator requires Python 3.10 or higher and TensorFlow 2.16+.

Installing with pixi (recommended)
------------------------------------

`Pixi <https://pixi.sh>`_ manages all dependencies including TensorFlow, CUDA,
and the geospatial stack. This is the recommended approach:

.. code-block:: bash

   git clone https://github.com/kr-colab/ReLocator.git
   cd ReLocator
   pixi install              # GPU environment (default)

This installs TensorFlow with CUDA support, Ray for multi-GPU parallelism,
bio2zarr for fast VCF conversion, and all development tools.

Available environments:

.. code-block:: bash

   pixi install              # GPU + ray + fast-vcf + dev (default)
   pixi install -e cpu       # CPU + ray + fast-vcf + dev
   pixi install -e test      # CPU + dev (minimal for testing)
   pixi install -e docs      # CPU + sphinx
   pixi install -e minimal   # CPU only, no extras

Running tasks:

.. code-block:: bash

   pixi run test             # run tests
   pixi run lint             # check linting
   pixi run format           # auto-format code
   pixi run docs             # build documentation

Installing with pip
--------------------

Locator can also be installed with pip. Note that you must manage TensorFlow
and CUDA installation separately when using pip.

.. code-block:: bash

   pip install locator

For development:

.. code-block:: bash

   pip install locator[dev,docs,ray,fast-vcf]

Optional pip extras:

* ``locator[dev]`` — pytest, ruff, pre-commit
* ``locator[docs]`` — sphinx and related packages
* ``locator[ray]`` — Ray for multi-GPU parallel analysis
* ``locator[fast-vcf]`` — bio2zarr + cyvcf2 for fast VCF-to-Zarr conversion

Fast VCF-to-Zarr Conversion
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For large VCF files, we recommend converting to zarr format for fast loading.
With pixi, ``bio2zarr`` is included in the default environment. With pip,
install via ``pip install locator[fast-vcf]``.

.. code-block:: bash

   bcftools index -t genotypes.vcf.gz
   vcf2zarr convert -p 8 genotypes.vcf.gz genotypes.zarr

Locator supports zarr files from both ``bio2zarr`` and ``scikit-allel``.
See :doc:`cli` for usage with windowed analysis.

Installing from Source
-----------------------

.. code-block:: bash

   git clone https://github.com/kr-colab/ReLocator.git
   cd ReLocator

   # With pixi (recommended)
   pixi install

   # Or with pip
   pip install -e ".[dev]"

Verifying Installation
-----------------------

.. code-block:: bash

   # With pixi
   pixi run python -c "from locator import Locator; print('OK')"

   # Or directly
   python -c "from locator import Locator; print('OK')"

Next Steps
----------

* Read the :doc:`usage` guide to learn how to use Locator
* Check the :doc:`api` reference for detailed documentation
