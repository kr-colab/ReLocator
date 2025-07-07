# Contributing to Locator

Thank you for your interest in contributing to Locator! This guide will help you get started with development.

## Development Setup

### 1. Clone the repository

```bash
git clone https://github.com/kr-colab/locator.git
cd locator
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install the package in development mode

```bash
pip install -e ".[dev]"
```

This installs Locator in editable mode along with all development dependencies including:
- `pytest` for testing
- `black` for code formatting
- `isort` for import sorting
- `flake8` for linting
- `pre-commit` for git hooks

### 4. Set up pre-commit hooks

Pre-commit hooks ensure code quality by automatically running formatters and linters before each commit.

```bash
python scripts/setup_pre_commit.py
```

Or manually:

```bash
pre-commit install
```

## Code Style

We use the following tools to maintain consistent code style:

- **Black**: Code formatter with a line length of 89 characters
- **isort**: Sorts and organizes imports
- **flake8**: Linting for code quality

These tools run automatically via pre-commit hooks, but you can also run them manually:

```bash
# Format all Python files
black locator/ tests/ scripts/

# Sort imports
isort locator/ tests/ scripts/

# Run linting
flake8 locator/ tests/ scripts/

# Or run all pre-commit hooks
pre-commit run --all-files
```

## Testing

Run the test suite with pytest:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=locator

# Run specific test file
pytest tests/test_core.py

# Run tests in parallel (requires pytest-xdist)
pytest -n auto
```

## Making Changes

1. Create a new branch for your feature or bugfix:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes and ensure tests pass

3. Commit your changes (pre-commit hooks will run automatically):
   ```bash
   git add .
   git commit -m "feat: add new feature"
   ```

   If pre-commit hooks fail, they may have automatically fixed issues. Review the changes and commit again.

4. Push your branch and create a pull request

## Commit Messages

We follow conventional commits format:

- `feat:` for new features
- `fix:` for bug fixes
- `docs:` for documentation changes
- `test:` for test additions/changes
- `refactor:` for code refactoring
- `perf:` for performance improvements
- `chore:` for maintenance tasks

## Pre-commit Hook Details

Our pre-commit configuration includes:

1. **File fixes**:
   - Remove trailing whitespace
   - Ensure files end with a newline
   - Check YAML syntax
   - Prevent large files (>1MB) from being committed
   - Check for merge conflicts
   - Fix line endings (LF)

2. **Python formatting**:
   - Black (89 character line limit)
   - isort (compatible with Black)

3. **Linting**:
   - flake8 with plugins for docstrings, bugbear, comprehensions, and simplify

## Troubleshooting

### Pre-commit hooks failing

If pre-commit hooks fail, they often fix issues automatically. Simply:

1. Review the changes made by the hooks
2. Add the modified files: `git add .`
3. Commit again

### Skipping hooks temporarily

If you need to skip hooks for a specific commit:

```bash
git commit --no-verify -m "your message"
```

However, please ensure your code passes all checks before creating a pull request.

## Questions?

If you have questions or need help, please:
- Check existing issues on GitHub
- Create a new issue for bugs or feature requests
- Reach out to the maintainers

Thank you for contributing to Locator!