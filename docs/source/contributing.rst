Contributing
============

Thank you for your interest in contributing to Locator! This guide will help
you get started with development.

Development Setup
-----------------

1. **Clone the repository and create a conda environment:**

   .. code-block:: bash

      git clone https://github.com/kr-colab/relocator.git
      cd relocator
      conda create -n relocator python=3.12
      conda activate relocator

2. **Install in development mode:**

   .. code-block:: bash

      pip install -e ".[dev]"

   This installs Locator in editable mode along with all development
   dependencies including ``pytest``, ``ruff``, ``pre-commit``, and
   ``codespell``.

3. **Set up pre-commit hooks:**

   .. code-block:: bash

      pre-commit install

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

Run the test suite with pytest:

.. code-block:: bash

   # Run all tests
   pytest

   # Run a specific test file or test
   pytest tests/test_core.py
   pytest tests/test_core.py::test_name -v

   # Skip slow or GPU tests
   pytest -m "not slow"
   pytest -m "not gpu"

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
