"""Locator: A tool for predicting geographic location from genetic variation"""

from .core import Locator, EnsembleLocator
from .plotting import plot_predictions, plot_error_summary, plot_sample_weights
from .models import create_network, euclidean_distance_loss

__version__ = "0.1.0"

# Make the package namespace clean and complete
__all__ = [
    # Main class
    "Locator",
    "EnsembleLocator",
    # Plotting functions
    "plot_predictions",
    "plot_error_summary",
    "plot_sample_weights",
    # Model functions
    "create_network",
    "euclidean_distance_loss",
]
