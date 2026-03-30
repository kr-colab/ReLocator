Contributing
============

Thank you for your interest in contributing to Locator! This guide will help
you get started with development.

Development Setup
-----------------

1. **Clone the repository and install with pixi:**

   .. code-block:: bash

      git clone https://github.com/kr-colab/ReLocator.git
      cd ReLocator
      pixi install          # GPU environment (default)
      pixi install -e cpu   # or CPU-only

   This installs Locator in editable mode along with TensorFlow, CUDA,
   and all development dependencies (pytest, ruff, pre-commit, etc.).

   Alternatively, with pip (you must manage TF/CUDA separately):

   .. code-block:: bash

      pip install -e ".[dev]"

2. **Set up pre-commit hooks:**

   .. code-block:: bash

      pixi run pre-commit install

Code Style
----------

We use `ruff <https://docs.astral.sh/ruff/>`_ for both linting and formatting,
configured in ``pyproject.toml`` with a line length of 89 characters.

Pre-commit hooks run ruff automatically on each commit. You can also run
the tools manually:

.. code-block:: bash

   # Lint (with auto-fix)
   ruff check --fix locator/ tests/

   # Format
   ruff format locator/ tests/

   # Run all pre-commit hooks at once
   pre-commit run --all-files

Testing
-------

Run the test suite:

.. code-block:: bash

   # Via pixi
   pixi run test

   # Or directly
   pytest tests/ -v -m "not slow and not gpu"

   # Run a specific test file or test
   pytest tests/test_core.py
   pytest tests/test_core.py::test_name -v

   # Force CPU-only
   CUDA_VISIBLE_DEVICES=-1 pytest

Making Changes
--------------

1. Create a new branch for your feature or bugfix:

   .. code-block:: bash

      git checkout -b feature/your-feature-name

2. Make your changes and ensure tests pass.

3. Commit your changes (pre-commit hooks will run automatically):

   .. code-block:: bash

      git commit -m "feat: add new feature"

   If pre-commit hooks fail, they may have automatically fixed issues.
   Review the changes, stage them, and commit again.

4. Push your branch and create a pull request.

Commit Messages
---------------

We use conventional commits format. A few examples:

- ``feat: add windowed analysis support``
- ``fix: correct sample index alignment after exclusion``
- ``docs: update API reference for EnsembleMixin``
- ``test: add coverage for k-fold holdouts``
- ``refactor: simplify IndexSet creation logic``

Documentation
-------------

When adding new features, please:

- Update the API documentation with docstrings
- Add examples if appropriate
- Update the user guide if necessary

All public functions and classes should have docstrings following the
`NumPy style <https://numpydoc.readthedocs.io/en/latest/format.html>`_
convention.

Reporting Issues
----------------

When reporting issues, please include:

- A clear description of the problem
- Steps to reproduce
- Expected vs. actual behavior
- Environment details (OS, Python version, etc.)

License
-------

By contributing to Locator, you agree that your contributions will be
licensed under the project's license.
