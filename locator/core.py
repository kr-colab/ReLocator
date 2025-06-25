"""Core functionality for locator - Refactored version"""

import numpy as np
import pandas as pd
import sys
import warnings
from tensorflow import keras
import matplotlib.pyplot as plt
import copy
from tqdm import tqdm
from pathlib import Path
import tensorflow as tf
from typing import List, Optional

from .models import create_network
from .utils import normalize_locs, filter_snps, weight_samples

# Import all the mixins
from .loaders import DataLoaderMixin
from .training import TrainingMixin
from .prediction import PredictionMixin
from .analysis import AnalysisMixin
from .visualization import VisualizationMixin


def setup_gpu(gpu_number=None):
    """Configure GPU settings for optimal usage.

    Args:
        gpu_number (int or str, optional): GPU index to use (0-based). If None, the first available GPU is used.

    Returns:
        bool: True if a GPU is available and successfully configured, otherwise False.
    """
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        print("No GPU devices available. Running on CPU.")
        return False

    try:
        if gpu_number is not None:
            # Convert to int if string
            gpu_number = int(gpu_number)
            if gpu_number < 0 or gpu_number >= len(gpus):
                raise ValueError(
                    f"GPU {gpu_number} not available. Found {len(gpus)} GPUs."
                )
            # Set visible devices to only the specified GPU
            tf.config.set_visible_devices(gpus[gpu_number], "GPU")
            print(f"Using GPU {gpu_number}: {gpus[gpu_number].name}")
        else:
            # Use first GPU by default
            tf.config.set_visible_devices(gpus[0], "GPU")
            print(f"Using GPU 0: {gpus[0].name}")

        # Enable memory growth for all visible GPUs
        for gpu in tf.config.get_visible_devices("GPU"):
            tf.config.experimental.set_memory_growth(gpu, True)

        return True
    except RuntimeError as e:
        print(f"GPU configuration error: {e}")
        print("Falling back to CPU.")
        return False
    except ValueError as e:
        print(f"GPU selection error: {e}")
        print("Falling back to CPU.")
        return False


class Locator(DataLoaderMixin, TrainingMixin, PredictionMixin, AnalysisMixin, VisualizationMixin):
    """A class for predicting geographic locations from genetic data.

    This class implements a neural network approach to predict sample locations from
    genetic data. It can handle various input formats including:

    - Genotype data:
        * VCF or VCF.gz files
        * Zarr format
        * Pandas DataFrame with samples as index, SNP positions as columns

    - Sample location data:
        * Tab-delimited file
        * Pandas DataFrame

    The model can be configured through a dictionary of parameters passed during
    initialization. Sample location data can be provided either as a file path or
    as a pandas DataFrame.

    Attributes:
        config (dict): Configuration dictionary containing model parameters
        model (keras.Model): The neural network model (created during training)
        history (keras.callbacks.History): Training history (available after training)
        samples (numpy.ndarray): Sample IDs from genotype data
        meanlong (float): Mean longitude for normalization
        sdlong (float): Standard deviation of longitude for normalization
        meanlat (float): Mean latitude for normalization
        sdlat (float): Standard deviation of latitude for normalization

    Example:
        >>> # Using a file path for sample data
        >>> locator = Locator({
        ...     "out": "analysis_1",
        ...     "sample_data": "samples.txt",
        ...     "zarr": "genotypes.zarr"
        ... })

        >>> # Using a DataFrame for sample data
        >>> locator = Locator({
        ...     "out": "analysis_1",
        ...     "sample_data": sample_df,  # pandas DataFrame
        ...     "zarr": "genotypes.zarr"
        ... })

        >>> # Using DataFrames for both inputs
        >>> # Coordinate DataFrame must have columns: sampleID, x, y
        >>> coords_df = pd.DataFrame({
        ...     "sampleID": ["sample1", "sample2"],
        ...     "x": [longitude1, longitude2],
        ...     "y": [latitude1, latitude2]
        ... })
        >>>
        >>> # Genotype DataFrame has samples as index, SNP positions as columns
        >>> geno_df = pd.DataFrame({
        ...     1001: [0, 1],    # SNP position 1001
        ...     2001: [1, 2],    # SNP position 2001
        ... }, index=["sample1", "sample2"])
        >>>
        >>> locator = Locator({
        ...     "out": "analysis_1",
        ...     "sample_data": coords_df,
        ...     "genotype_data": geno_df
        ... })
    """

    def __init__(self, config=None):
        """
        Initialize Locator with configuration parameters.

        :param config: Configuration dictionary that can include the following keys:
        :type config: dict, optional

        **Top-level keys:**

        - **sample_data** (*str or pandas.DataFrame*): Path to sample data file or a DataFrame with columns 'sampleID', 'x', 'y'.
        - **genotype_data** (*pandas.DataFrame*): DataFrame with samples as index, SNP positions as columns, and genotype counts (0, 1, 2) as values.
        - **zarr** (*str*): Path to Zarr format genotype data.
        - **vcf** (*str*): Path to VCF format genotype data.
        - **out** (*str*): Output root name for all output files.
        - **train_split** (*float*): Proportion of data to use for training.
        - **batch_size** (*int*): Batch size for training.
        - **max_epochs** (*int*): Maximum number of training epochs.
        - **patience** (*int*): Patience for early stopping.
        - **min_mac** (*int*): Minimum minor allele count for SNP filtering.
        - **max_SNPs** (*int*): Maximum number of SNPs to use.
        - **width** (*int*): Width of neural network layers.
        - **nlayers** (*int*): Number of neural network layers.
        - **dropout_prop** (*float*): Dropout proportion.
        - **keras_verbose** (*int*): Verbosity level for Keras training.
        - **impute_missing** (*bool*): Whether to impute missing genotypes.
        - **validation_split** (*float*): Proportion of data to use for validation.
        - **learning_rate** (*float*): Learning rate for the optimizer.
        - **min_epochs** (*int*): Minimum number of epochs to train.
        - **patience** (*int*): Number of epochs with no improvement to wait before stopping.
        - **min_delta** (*float*): Minimum change in validation loss to qualify as an improvement.
        - **restore_best_weights** (*bool*): Whether to restore model weights from the epoch with the best validation loss.
        - **prediction_frequency** (*int*): Frequency (in epochs) of making predictions during training.
        - **optimizer_algo** (*str*): Optimizer algorithm to use ("adam" or "adamw").
        - **weight_decay** (*float*): Weight decay coefficient for AdamW optimizer.
        - **augmentation** (*dict*): Dictionary of augmentation parameters:
            - **enabled** (*bool*): Whether data augmentation is enabled.
            - **flip_rate** (*float*): Rate at which to randomly flip genotypes during augmentation.
        - **weight_samples** (*dict*): Dictionary of sample weighting parameters:
            - **enabled** (*bool*): Whether to weight samples by distance.
            - **method** (*str*): Method for weighting samples ("KD", "histogram", "df").
            - **xbins** (*int*): Number of bins for histogram.
            - **ybins** (*int*): Number of bins for histogram.
            - **lam** (*float*): Exponent for weights.
            - **bandwidth** (*float*): Bandwidth for KDE.
            - **weightdf** (*pandas.DataFrame*): DataFrame containing sample weights.
        - **use_range_penalty** (*bool*): Whether to apply a range penalty in the loss function.
        - **penalty_weight** (*float*): Weight assigned to the range penalty term.
        - **species_range_geom** (*shapely.geometry*): Shapely geometry object defining the valid species range.
        """
        # Set default configuration
        self.config = {
            # Data parameters
            "train_split": 0.9,
            "batch_size": 32,
            "min_mac": 2,
            "max_SNPs": None,
            "impute_missing": False,
            # Network architecture
            "width": 256,
            "nlayers": 8,
            "dropout_prop": 0.25,
            # Training parameters
            "max_epochs": 5000,
            "patience": 100,
            "learning_rate": 0.001,
            "min_epochs": 10,
            "min_delta": 1e-4,
            "restore_best_weights": True,
            # Optimizer parameters
            "optimizer_algo": "adam",
            "weight_decay": 0.004,
            # Output control
            "keras_verbose": 1,
            "prediction_frequency": 1,
            # Validation
            "validation_split": 0.1,
            # Data augmentation parameters
            "augmentation": {
                "enabled": False,  # Whether to use data augmentation
                "flip_rate": 0.05,  # Rate at which to flip genotypes
            },
            "weight_samples": {
                "enabled": False,  # Whether to weight samples by distance
                "method": "KD",     # Method for weighting samples ("KD", "histogram", "df")
                "xbins": 10,       # Number of bins for histogram
                "ybins": 10,       # Number of bins for histogram
                "lam": 1.0,       # Exponent for weights
                "bandwidth": None, # Bandwidth for KDE
                "weightdf": None,  # DataFrame containing sample weights
                },
            # Range penalty parameters
            "use_range_penalty": False,
            "species_range_shapefile": None,
            "resolution": 0.05,
            "penalty_weight": 1.0,
            "out": "locator",
        }

        # Update with user config
        if config is not None:
            self.config.update(config)

        # If using range penalty and a species_range_geom is provided, set it in models
        if (
            self.config.get("use_range_penalty")
            and self.config.get("species_range_geom") is not None
        ):
            from .models import set_species_range_geom

            set_species_range_geom(self.config["species_range_geom"])

        # Handle sample_data DataFrame input
        if isinstance(self.config.get("sample_data"), pd.DataFrame):
            sample_df = self.config["sample_data"]
            required_cols = ["sampleID", "x", "y"]
            if not all(col in sample_df.columns for col in required_cols):
                raise ValueError(
                    f"sample_data DataFrame must contain columns: {required_cols}"
                )
            self._sample_data_df = sample_df.copy()

        # Handle genotype_data DataFrame input
        if isinstance(self.config.get("genotype_data"), pd.DataFrame):
            geno_df = self.config["genotype_data"]
            # Validate genotype values are 0,1,2
            unique_values = np.unique(geno_df.values)
            if not all(x in [0, 1, 2] for x in unique_values):
                raise ValueError("Genotype values must be 0, 1, or 2")
            # Store positions for windowed analysis
            try:
                self.positions = geno_df.columns.astype(float).values
            except ValueError:
                raise ValueError(
                    "Column names must be convertible to integers (SNP positions)"
                )
            # Store DataFrame
            self._genotype_df = geno_df.copy()

        # Initialize attributes that will be set during training
        self.model = None
        self.history = None
        self.samples = None
        self.meanlong = None
        self.sdlong = None
        self.meanlat = None
        self.sdlat = None
        if not hasattr(self, "positions"):
            self.positions = None  # For windowed analysis
        self.unnormedlocs = None # For calculating sample weights
        self.sample_weights = None

        # Setup GPU if not explicitly disabled
        if not self.config.get("disable_gpu", False):
            gpu_number = self.config.get("gpu_number")
            if gpu_number is not None:
                # Convert to int if string
                try:
                    gpu_number = int(gpu_number)
                except ValueError:
                    print(f"Invalid GPU number: {gpu_number}. Using default GPU.")
                    gpu_number = None
            setup_gpu(gpu_number)
        else:
            print("GPU usage disabled by configuration.")

        # Set memory growth for better GPU memory management
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                print(f"GPU memory growth setting failed: {e}")

    @property
    def sample_data(self) -> pd.DataFrame:
        """
        Returns the sample data as a pandas DataFrame.

        Returns:
            pd.DataFrame: The sample data DataFrame with columns ['sampleID', 'x', 'y', ...].

        Raises:
            ValueError: If sample data is not available.

        Example:
            >>> locator = Locator({"sample_data": coords_df})
            >>> df = locator.sample_data
        """
        if hasattr(self, "_sample_data_df"):
            return self._sample_data_df
        elif "sample_data" in self.config:
            # Try to load from file
            try:
                sample_df = pd.read_csv(self.config["sample_data"], sep="\t")
                self._sample_data_df = sample_df
                return sample_df
            except Exception as e:
                raise ValueError(f"Could not load sample data: {e}")
        else:
            raise ValueError("No sample data available")


# Import EnsembleLocator from ensemble.py
from .ensemble import EnsembleLocator