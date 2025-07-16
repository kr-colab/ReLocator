#!/usr/bin/env python
"""
Script to set up pre-commit hooks for the relocator project.

Usage:
    python scripts/setup_pre_commit.py
"""

import subprocess
import sys
from pathlib import Path


def main():
    """Install pre-commit hooks."""
    project_root = Path(__file__).parent.parent

    print("Setting up pre-commit hooks for relocator project...")

    # Check if pre-commit is installed
    try:
        subprocess.run(["pre-commit", "--version"], check=True, capture_output=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Error: pre-commit is not installed.")
        print("Please install it with: pip install pre-commit")
        print("Or install all dev dependencies with: pip install -e '.[dev]'")
        sys.exit(1)

    # Install the pre-commit hooks
    try:
        subprocess.run(["pre-commit", "install"], cwd=project_root, check=True)
        print("✓ Pre-commit hooks installed successfully!")

        # Run pre-commit on all files to check current status
        print("\nRunning pre-commit checks on all files (this may take a moment)...")
        result = subprocess.run(
            ["pre-commit", "run", "--all-files"],
            cwd=project_root,
            capture_output=True,
            text=True,
        )

        if result.returncode == 0:
            print("✓ All pre-commit checks passed!")
        else:
            print("Some pre-commit checks failed. This is normal for the first run.")
            print("The hooks will automatically fix many issues on commit.")
            print("\nTo manually run all hooks now:")
            print("  pre-commit run --all-files")

    except subprocess.CalledProcessError as e:
        print(f"Error installing pre-commit hooks: {e}")
        sys.exit(1)

    print("\nPre-commit is now set up! Hooks will run automatically on git commit.")
    print("\nUseful commands:")
    print("  pre-commit run --all-files  # Run on all files")
    print("  pre-commit run             # Run on staged files")
    print("  git commit --no-verify      # Skip hooks for one commit")


if __name__ == "__main__":
    main()
