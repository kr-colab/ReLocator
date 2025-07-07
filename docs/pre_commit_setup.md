# Pre-commit Setup for Locator

This document explains how to set up and use pre-commit hooks for the Locator project.

## Quick Start

1. **Install development dependencies**:
   ```bash
   pip install -e ".[dev]"
   ```

2. **Install pre-commit hooks**:
   ```bash
   python scripts/setup_pre_commit.py
   ```

That's it! Pre-commit hooks will now run automatically when you commit changes.

## What Pre-commit Does

Pre-commit runs the following checks before each commit:

1. **File Fixes**:
   - Removes trailing whitespace
   - Ensures files end with a newline
   - Checks YAML/TOML syntax
   - Prevents large files (>1MB) from being committed
   - Checks for merge conflict markers
   - Ensures consistent line endings (LF)

2. **Code Formatting**:
   - **Black**: Formats Python code with 89-character line limit
   - **isort**: Sorts and organizes imports

3. **Code Quality**:
   - **flake8**: Checks for code style issues, with plugins for:
     - Docstring conventions
     - Common bugs (bugbear)
     - Comprehension improvements
     - Code simplification suggestions

## Manual Usage

### Format all files
```bash
python scripts/format_all.py
```

### Run pre-commit on all files
```bash
pre-commit run --all-files
```

### Run pre-commit on staged files only
```bash
pre-commit run
```

### Skip hooks for one commit
```bash
git commit --no-verify -m "your message"
```

## Configuration Files

- `.pre-commit-config.yaml`: Pre-commit hook configuration
- `pyproject.toml`: Black and isort settings
- `.flake8`: Flake8 linting rules

## Troubleshooting

### Pre-commit not installed
If you get an error about pre-commit not being installed:
```bash
pip install pre-commit
# or
pip install -e ".[dev]"
```

### Hooks failing on first run
This is normal! The hooks often fix issues automatically. Just:
1. Review the changes
2. Stage them: `git add .`
3. Commit again

### Black and isort conflicts
The configurations are set to be compatible. If you still see conflicts:
- Black line length: 89 characters
- isort profile: "black"
- Both tools respect the same settings

## IDE Integration

### VS Code
Add to `.vscode/settings.json`:
```json
{
    "python.formatting.provider": "black",
    "python.formatting.blackArgs": ["--line-length=89"],
    "python.sortImports.args": ["--profile", "black", "--line-length", "89"],
    "editor.formatOnSave": true,
    "python.linting.flake8Enabled": true
}
```

### PyCharm
1. Go to Settings → Tools → File Watchers
2. Add watchers for Black and isort
3. Configure with the same arguments as in our config files

## Benefits

- **Consistent code style** across all contributors
- **Automatic formatting** reduces review friction
- **Early error detection** before CI/CD
- **Clean git history** without formatting commits