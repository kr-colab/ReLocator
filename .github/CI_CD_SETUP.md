# CI/CD Setup for Locator

This repository uses GitHub Actions for continuous integration and deployment.

## Workflows

### 1. Tests (`.github/workflows/test.yml`)
- **Triggers**: Push and pull requests to main/develop branches
- **Python versions**: 3.9, 3.10, 3.11
- **Features**:
  - Parallel test execution with pytest-xdist (`-n auto`)
  - Coverage reporting with pytest-cov
  - CPU-only mode (no GPU required)
  - Dependency caching for faster runs
  - Code linting with black, isort, and flake8

### 2. Documentation (`.github/workflows/docs.yml`)
- Builds Sphinx documentation
- Checks for documentation warnings
- Uploads built docs as artifacts

### 3. Publishing (`.github/workflows/publish.yml`)
- Triggered on GitHub releases
- Publishes to PyPI and Test PyPI
- Requires secrets: `PYPI_API_TOKEN`, `TEST_PYPI_API_TOKEN`

### 4. Manual Testing (`.github/workflows/manual-test.yml`)
- Allows manual workflow triggers
- Configurable Python version and test patterns

## Local Testing

Run tests locally with parallel execution:
```bash
# Run all tests in parallel
pytest -n auto

# Run with 4 workers
pytest -n 4

# Run without GPU (recommended)
CUDA_VISIBLE_DEVICES=-1 pytest -n auto

# Run specific test file
pytest tests/test_verbosity_control.py -n auto

# Run with coverage
pytest -n auto --cov=locator --cov-report=html
```

## Configuration

- **pytest configuration**: See `pyproject.toml`
- **Coverage settings**: See `pyproject.toml`
- **Dependabot**: See `.github/dependabot.yml`

## Required Secrets (for publishing)

Set these in your GitHub repository settings:
- `CODECOV_TOKEN` (optional, for private repos)
- `PYPI_API_TOKEN` (for PyPI publishing)
- `TEST_PYPI_API_TOKEN` (for Test PyPI publishing)

## Status Badges

Add these to your README.md:
```markdown
[![Tests](https://github.com/YOUR_USERNAME/relocator/actions/workflows/test.yml/badge.svg)](https://github.com/YOUR_USERNAME/relocator/actions/workflows/test.yml)
[![codecov](https://codecov.io/gh/YOUR_USERNAME/relocator/branch/main/graph/badge.svg)](https://codecov.io/gh/YOUR_USERNAME/relocator)
```
