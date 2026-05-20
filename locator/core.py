"""Core functionality for locator - Refactored version"""

import warnings

import numpy as np
import pandas as pd
import tensorflow as tf

from .analysis import AnalysisMixin
from .ensemble_mixin import EnsembleMixin
from .gpu_optimizer import GPUOptimizer

# Import all the mixins
from .loaders import DataLoaderMixin
from .plotting import PlottingMixin
from .prediction import PredictionMixin
from .training import TrainingMixin


def setup_gpu(gpu_number=None):
    """Configure GPU settings for optimal usage.

    Args:
        gpu_number (int or str, optional): GPU index to use (0-based). If None, the first available GPU is used.

    Returns
    -------
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
    except RuntimeError as e:
        print(f"GPU configuration error: {e}")
        print("Falling back to CPU.")
        return False
    except ValueError as e:
        print(f"GPU selection error: {e}")
        print("Falling back to CPU.")
        return False

    # Memory growth is best-effort. When a Ray actor has already installed
    # a hard memory cap via ``set_logical_device_configuration`` before
    # Locator init runs, this call raises ValueError — that's fine, the
    # GPU is still usable under the existing cap.
    for gpu in tf.config.get_visible_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except (RuntimeError, ValueError):
            pass

    return True


class Locator(
    DataLoaderMixin,
    TrainingMixin,
    PredictionMixin,
    AnalysisMixin,
    EnsembleMixin,
    PlottingMixin,
):
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

    Attributes
    ----------
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

    def __init__(self, config=None):  # noqa: C901
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
        - **pca_components** (*int or "auto"*): If set, prepend a PCA-initialized linear projection of this width as the first layer and fine-tune it. Use ``"auto"`` to pick the width from the genotype-PCA scree elbow. Recommended when n_SNPs >> n_samples. Default None (disabled).
        - **pca_finetune** (*bool*): Whether to unfreeze the PCA projection for a low-learning-rate fine-tuning phase. Default True. False keeps the projection frozen at its PCA initialization.
        - **pca_finetune_lr** (*float*): Learning rate for the PCA fine-tuning phase. Default 1e-4.
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
        - **na_action** (*str*): How to handle samples without coordinates. Options:
            - 'separate' (default): Include all samples, train on known, predict unknown.
            - 'exclude': Only use samples with known coordinates.
            - 'fail': Raise error if any samples lack coordinates.
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
            # PCA-initialized projection (for n_SNPs >> n_samples)
            "pca_components": None,
            "pca_finetune": True,
            "pca_finetune_lr": 1e-4,
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
                "method": "KD",  # Method for weighting samples ("KD", "histogram", "df")
                "xbins": 10,  # Number of bins for histogram
                "ybins": 10,  # Number of bins for histogram
                "lam": 1.0,  # Exponent for weights
                "bandwidth": None,  # Bandwidth for KDE
                "weightdf": None,  # DataFrame containing sample weights
            },
            # Range penalty parameters
            "use_range_penalty": False,
            "species_range_shapefile": None,
            "resolution": 0.05,
            "penalty_weight": 1.0,
            "out": "locator",
            # NA handling
            "na_action": "separate",  # How to handle samples without coordinates
            # GPU optimization parameters
            "use_mixed_precision": True,  # Enable mixed precision training
            "gpu_batch_size": "auto",  # 'auto' or specific number
            "gradient_accumulation_steps": 1,  # For simulating larger batches
            "gpu_memory_mode": "growth",  # 'growth', 'preallocate', or 'limit:MB'
            "enable_xla": False,  # Experimental XLA compilation
            # Performance optimization
            "optimize_tf_parallelism": True,  # Reduce TF parallelism to prevent forking
            "holdout_no_intermediate_saves": True,  # Skip intermediate model saves in k-fold CV
            "save_fold_models": True,  # Save model checkpoints during training
            # Verbosity control
            "verbose_splits": False,  # Show train/val/test split sizes
            "verbose_batch_size": False,  # Show batch size optimization details
        }

        # Update with user config
        if config is not None:
            self.config.update(config)

        # Handle deprecated use_efficient_pipeline option
        if "use_efficient_pipeline" in self.config:
            warnings.warn(
                "The 'use_efficient_pipeline' option is deprecated and will be ignored. "
                "Locator now always uses the efficient tf.data pipeline.",
                DeprecationWarning,
                stacklevel=2,
            )
            # Remove from config to avoid confusion
            del self.config["use_efficient_pipeline"]

        # Validate na_action parameter
        valid_na_actions = ["separate", "exclude", "fail"]
        if self.config["na_action"] not in valid_na_actions:
            raise ValueError(
                f"Invalid na_action '{self.config['na_action']}'. "
                f"Must be one of: {valid_na_actions}"
            )

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
        self.unnormedlocs = None  # For calculating sample weights
        self.sample_weights = None
        # GPU-resident genotype table (see IndexedGenotypeModel) built lazily
        # and reused across folds; _genotype_table_src records which
        # filtered_genotypes array it was built from so it can be rebuilt when
        # the underlying data changes (e.g. per window in windowed analysis).
        self._genotype_table = None
        self._genotype_table_src = None

        # Store na_action as instance attribute for convenience
        self.na_action = self.config["na_action"]

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

            # Apply GPU optimizations
            # 1. Mixed precision training
            if self.config.get("use_mixed_precision", False):
                if GPUOptimizer.setup_mixed_precision():
                    self.config["use_mixed_precision"] = True
                else:
                    self.config["use_mixed_precision"] = False

            # 2. GPU memory configuration
            memory_mode = self.config.get("gpu_memory_mode", "growth")
            if memory_mode.startswith("limit:"):
                limit_mb = int(memory_mode.split(":")[1])
                GPUOptimizer.optimize_gpu_memory("limit", limit_mb)
            else:
                GPUOptimizer.optimize_gpu_memory(memory_mode)

            # 3. Enable XLA if requested
            if self.config.get("enable_xla", False):
                try:
                    GPUOptimizer.enable_xla_compilation()
                except Exception as e:
                    print(f"XLA compilation failed: {e}")
                    self.config["enable_xla"] = False

        else:
            print("GPU usage disabled by configuration.")
            self.config["use_mixed_precision"] = False

        # Configure TensorFlow for optimal performance
        self._configure_tensorflow_optimization()

    def _configure_tensorflow_optimization(self):
        """Configure TensorFlow to minimize process forking and optimize performance."""
        # Reduce inter-op parallelism to prevent excessive forking
        if self.config.get("optimize_tf_parallelism", True):
            # Set to 1 to prevent process forking, use threads within ops instead
            tf.config.threading.set_inter_op_parallelism_threads(1)
            # Keep intra-op threads reasonable for parallel operations
            tf.config.threading.set_intra_op_parallelism_threads(4)

            # Also set environment variables for consistency
            import os

            os.environ["TF_NUM_INTEROP_THREADS"] = "1"
            os.environ["TF_NUM_INTRAOP_THREADS"] = "4"

            # Disable tf.data autotune to prevent excessive parallelism
            os.environ["TF_DATA_EXPERIMENTAL_SLACK"] = "false"

            if self.config.get("keras_verbose", 1) >= 1:
                print("TensorFlow threading optimized to reduce process forking")

    @property
    def sample_data(self) -> pd.DataFrame:
        """
        Returns the sample data as a pandas DataFrame.

        Returns
        -------
            pd.DataFrame: The sample data DataFrame with columns ['sampleID', 'x', 'y', ...].

        Raises
        ------
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

    def get_sample_status(self, samples, sample_data=None):
        """
        Analyze sample coordinate status.

        This method identifies which samples have known geographic coordinates and which have
        missing (NA) coordinates. This is useful for understanding your data and for methods
        that need to handle samples with and without coordinates differently.

        Args:
            samples (numpy.ndarray): Array of sample IDs from genotype data
            sample_data (pandas.DataFrame, optional): DataFrame with columns 'sampleID', 'x', 'y'.
                If not provided, uses the stored sample data or loads from config.

        Returns
        -------
            dict: A dictionary containing:
                - 'known_indices' (numpy.ndarray): Array indices of samples with coordinates
                - 'na_indices' (numpy.ndarray): Array indices of samples without coordinates
                - 'known_samples' (numpy.ndarray): Sample IDs with coordinates
                - 'na_samples' (numpy.ndarray): Sample IDs without coordinates
                - 'n_known' (int): Count of samples with known coordinates
                - 'n_na' (int): Count of samples with NA coordinates
                - 'total' (int): Total number of samples

        Example:
            >>> locator = Locator(config)
            >>> status = locator.get_sample_status(samples)
            >>> print(f"Found {status['n_known']} samples with coordinates")
            >>> print(f"Found {status['n_na']} samples without coordinates")
        """
        # Get sample data and locations
        if sample_data is None:
            sample_data, locs = self.sort_samples(samples)
        else:
            # Validate provided DataFrame
            required_cols = ["sampleID", "x", "y"]
            if not all(col in sample_data.columns for col in required_cols):
                raise ValueError(f"sample_data must contain columns: {required_cols}")
            locs = sample_data[["x", "y"]].values

        # Find indices with known and NA coordinates
        # A sample has known coordinates if both x and y are not NaN
        known_mask = ~(np.isnan(locs[:, 0]) | np.isnan(locs[:, 1]))
        known_idx = np.where(known_mask)[0]
        na_idx = np.where(~known_mask)[0]

        # Get sample IDs for each group
        known_samples = samples[known_idx] if len(known_idx) > 0 else np.array([])
        na_samples = samples[na_idx] if len(na_idx) > 0 else np.array([])

        return {
            "known_indices": known_idx,
            "na_indices": na_idx,
            "known_samples": known_samples,
            "na_samples": na_samples,
            "n_known": len(known_idx),
            "n_na": len(na_idx),
            "total": len(samples),
        }

    def check_data(self, genotypes, samples, verbose=True):
        """
        Check data quality and report statistics.

        This is a convenience method to help users understand their data before running
        analyses. It reports the number of samples, SNPs, and identifies samples with
        missing coordinates.

        Args:
            genotypes (numpy.ndarray or allel.GenotypeArray): Genotype data
            samples (numpy.ndarray): Array of sample IDs
            verbose (bool): If True, print detailed statistics. Default: True

        Returns
        -------
            dict: Sample status dictionary from get_sample_status()

        Example::

            >>> locator = Locator(config)
            >>> genotypes, samples = locator.load_genotypes()
            >>> status = locator.check_data(genotypes, samples)
            Data Summary
            ==================================================
            Total samples: 231
            Samples with coordinates: 211
            Samples without coordinates: 20
            Total SNPs: 1000

            Current NA handling mode: separate
            - Will train on samples with known locations
            - Can predict on samples without locations

            Samples without coordinates (first 10):
              - sample_001
              - sample_002
              ...
        """
        # Get sample status
        status = self.get_sample_status(samples)

        if verbose:
            print("Data Summary")
            print("=" * 50)
            print(f"Total samples: {status['total']}")
            print(f"Samples with coordinates: {status['n_known']}")
            print(f"Samples without coordinates: {status['n_na']}")

            # Report SNP count
            if hasattr(genotypes, "shape"):
                n_snps = genotypes.shape[0]
                print(f"Total SNPs: {n_snps}")

            # Report NA handling mode
            print(f"\nCurrent NA handling mode: {self.na_action}")
            if self.na_action == "separate":
                print("- Will train on samples with known locations")
                print("- Can predict on samples without locations")
            elif self.na_action == "exclude":
                print("- Will only use samples with known locations")
                print("- Samples without locations will be excluded from all analyses")
            elif self.na_action == "fail":
                print("- Will raise an error if any samples lack coordinates")

            # Show samples without coordinates
            if status["n_na"] > 0:
                print("\nSamples without coordinates (first 10):")
                for i, sample_id in enumerate(status["na_samples"][:10]):
                    print(f"  - {sample_id}")
                if status["n_na"] > 10:
                    print(f"  ... and {status['n_na'] - 10} more")

                # Provide guidance based on na_action
                if self.na_action == "fail":
                    print(
                        "\n⚠️  WARNING: Your current na_action='fail' setting will cause"
                    )
                    print("   methods to fail with these NA samples. Consider using")
                    print("   na_action='separate' or 'exclude' instead.")

        return status


# Import EnsembleLocator from ensemble.py
from .ensemble import EnsembleLocator  # noqa: E402, F401
