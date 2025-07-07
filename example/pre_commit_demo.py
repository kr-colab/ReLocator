"""
Demo file to show pre-commit in action.

This file intentionally has formatting issues that will be fixed by pre-commit:
- Long lines
- Unsorted imports
- Trailing whitespace
- Missing final newline
"""

from typing import Dict, List

import numpy as np


# Long line (black will wrap)
def example_function_with_very_long_name_that_exceeds_line_limit(
    parameter1: str, parameter2: int, parameter3: float
) -> Dict[str, any]:
    """This is an example function with a very long line that will be wrapped by black."""
    return {"param1": parameter1, "param2": parameter2, "param3": parameter3}


# Trailing whitespace (pre-commit will remove)
class ExampleClass:
    """Example class with formatting issues."""

    def __init__(self):
        """Initialize with empty data list."""
        self.data = []

    def process_data(self, input_data: List[float]) -> np.ndarray:
        """Process some data."""
        # Complex expression (black will format)
        result = np.array(
            [x**2 + 2 * x + 1 if x > 0 else x**2 - 2 * x + 1 for x in input_data]
        )
        return result


# Missing final newline (pre-commit will add)
