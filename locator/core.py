"""Core functionality for locator - Refactored version"""

import os
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
            "save_fold_models": False,  # Skip saving individual fold models and histories
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
        self.unnormedlocs = None  # For calculating sample weights
        self.sample_weights = None

        # Store na_action as instance attribute for convenience
        self.na_action = self.config["na_action"]

        # Initialize sample exclusion attributes
        self._excluded_sample_ids = set()  # Set of excluded sample IDs
        self._exclusion_source = {}  # Track why each sample was excluded

        # Handle exclude_samples parameter if provided
        if "exclude_samples" in self.config:
            self._load_excluded_samples(self.config["exclude_samples"])

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

    def _load_excluded_samples(self, exclude_source):
        """Load excluded samples from file or list.

        Args:
            exclude_source: Can be:
                - str: Path to text file with one sample ID per line
                - list: List of sample IDs to exclude
                - set: Set of sample IDs to exclude

        File Format:
            - One sample ID per line
            - Lines starting with # are treated as comments
            - Empty lines are ignored

        Example file content:
            # Outlier samples identified in QC
            sample_001
            sample_005

            # Low quality samples
            sample_023
            sample_045
        """
        if isinstance(exclude_source, str):
            # Load from file
            if not os.path.exists(exclude_source):
                raise FileNotFoundError(f"Exclusion file not found: {exclude_source}")

            with open(exclude_source, "r") as f:
                for line in f:
                    sample_id = line.strip()
                    if sample_id and not sample_id.startswith("#"):  # Allow comments
                        self._excluded_sample_ids.add(sample_id)
                        self._exclusion_source[sample_id] = f"file:{exclude_source}"

        elif isinstance(exclude_source, (list, set)):
            # Load from collection
            for sample_id in exclude_source:
                self._excluded_sample_ids.add(str(sample_id))
                self._exclusion_source[str(sample_id)] = "config"
        else:
            raise ValueError(f"Invalid exclude_samples type: {type(exclude_source)}")

        if self.config.get("verbose", 0) > 0:
            print(f"Loaded {len(self._excluded_sample_ids)} samples for exclusion")

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

        Returns:
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

        # Get sample IDs for each group - use sample_data since it's already filtered
        sample_ids_in_data = sample_data["sampleID"].values
        known_samples = (
            sample_ids_in_data[known_idx] if len(known_idx) > 0 else np.array([])
        )
        na_samples = sample_ids_in_data[na_idx] if len(na_idx) > 0 else np.array([])

        # Add exclusion information
        status = {
            "known_indices": known_idx,
            "na_indices": na_idx,
            "known_samples": known_samples,
            "na_samples": na_samples,
            "n_known": len(known_idx),
            "n_na": len(na_idx),
            "total": len(samples),  # Original sample count
            "n_excluded": len(self._excluded_sample_ids),
        }

        # Calculate excluded samples that had coordinates
        if self._excluded_sample_ids:
            # Check exclusions against original samples array
            excluded_in_original = np.isin(samples, list(self._excluded_sample_ids))

            # For excluded samples, check if they would have had coordinates
            n_excluded_with_coords = 0
            for i, sample_id in enumerate(samples):
                if excluded_in_original[i] and sample_id in self._excluded_sample_ids:
                    # Check if this sample exists in the original data with coordinates
                    if hasattr(self, "_sample_data_df"):
                        sample_row = self._sample_data_df[
                            self._sample_data_df["sampleID"] == sample_id
                        ]
                        if (
                            not sample_row.empty
                            and not sample_row[["x", "y"]].isna().any(axis=1).iloc[0]
                        ):
                            n_excluded_with_coords += 1

            status["n_excluded_with_coords"] = n_excluded_with_coords
            status["n_available"] = status["n_known"]  # Known samples after exclusion
        else:
            status["n_excluded_with_coords"] = 0
            status["n_available"] = status["n_known"]

        return status

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

        Returns:
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

        if not verbose:
            return status

        # Print basic summary
        self._print_data_summary(status)

        # Report exclusions if any
        if status["n_excluded"] > 0:
            self._print_exclusion_summary(status)

        # Report SNP count
        if hasattr(genotypes, "shape"):
            print(f"Total SNPs: {genotypes.shape[0]}")

        # Report NA handling mode
        self._print_na_handling_mode()

        # Show samples without coordinates
        if status["n_na"] > 0:
            self._print_na_samples(status)

        return status

    def _print_data_summary(self, status):
        """Print basic data summary."""
        print("Data Summary")
        print("=" * 50)
        print(f"Total samples: {status['total']}")
        print(f"Samples with coordinates: {status['n_known']}")
        print(f"Samples without coordinates: {status['n_na']}")

    def _print_exclusion_summary(self, status):
        """Print exclusion summary."""
        print(f"Excluded samples: {status['n_excluded']}")
        print(
            f"  - Excluded samples with coordinates: {status['n_excluded_with_coords']}"
        )
        print(f"Available samples for training: {status['n_available']}")

    def _print_na_handling_mode(self):
        """Print NA handling mode information."""
        print(f"\nCurrent NA handling mode: {self.na_action}")

        na_mode_messages = {
            "separate": [
                "- Will train on samples with known locations",
                "- Can predict on samples without locations",
            ],
            "exclude": [
                "- Will only use samples with known locations",
                "- Samples without locations will be excluded from all analyses",
            ],
            "fail": ["- Will raise an error if any samples lack coordinates"],
        }

        for message in na_mode_messages.get(self.na_action, []):
            print(message)

    def _print_na_samples(self, status):
        """Print samples without coordinates."""
        print("\nSamples without coordinates (first 10):")
        for i, sample_id in enumerate(status["na_samples"][:10]):
            print(f"  - {sample_id}")
        if status["n_na"] > 10:
            print(f"  ... and {status['n_na'] - 10} more")

        # Provide guidance based on na_action
        if self.na_action == "fail":
            print("\n⚠️  WARNING: Your current na_action='fail' setting will cause")
            print("   methods to fail with these NA samples. Consider using")
            print("   na_action='separate' or 'exclude' instead.")

    def exclude_samples(self, sample_ids, reason="manual"):
        """Exclude additional samples from analysis.

        Args:
            sample_ids: Sample ID or list of sample IDs to exclude
            reason: String describing why samples were excluded

        Note:
            This modifies the exclusion list for future operations.
            Already completed analyses are not affected.

        Example:
            >>> # Exclude a single sample
            >>> locator.exclude_samples("sample_001", reason="outlier")

            >>> # Exclude multiple samples
            >>> locator.exclude_samples(["sample_002", "sample_003"], reason="low_quality")
        """
        if isinstance(sample_ids, str):
            sample_ids = [sample_ids]

        for sample_id in sample_ids:
            self._excluded_sample_ids.add(str(sample_id))
            self._exclusion_source[str(sample_id)] = reason

        print(f"Excluded {len(sample_ids)} samples (reason: {reason})")
        print(f"Total excluded samples: {len(self._excluded_sample_ids)}")

    def include_samples(self, sample_ids):
        """Remove samples from the exclusion list.

        Args:
            sample_ids: Sample ID or list of sample IDs to include back

        Returns:
            int: Number of samples actually removed from exclusion list

        Example:
            >>> # Include previously excluded samples back
            >>> n_included = locator.include_samples(["sample_001", "sample_002"])
            >>> print(f"Included {n_included} samples back into analysis")
        """
        if isinstance(sample_ids, str):
            sample_ids = [sample_ids]

        n_removed = 0
        for sample_id in sample_ids:
            if str(sample_id) in self._excluded_sample_ids:
                self._excluded_sample_ids.remove(str(sample_id))
                self._exclusion_source.pop(str(sample_id), None)
                n_removed += 1

        print(f"Removed {n_removed} samples from exclusion list")
        print(f"Total excluded samples: {len(self._excluded_sample_ids)}")
        return n_removed

    def get_excluded_samples(self):
        """Get information about excluded samples.

        Returns:
            pandas.DataFrame: DataFrame with columns:
                - sampleID: Excluded sample ID
                - reason: Reason for exclusion

        Example:
            >>> # View all excluded samples
            >>> excluded_df = locator.get_excluded_samples()
            >>> print(excluded_df)
                 sampleID        reason
            0  sample_001      outlier
            1  sample_002   low_quality
            2  sample_003   low_quality
        """
        if not self._excluded_sample_ids:
            return pd.DataFrame(columns=["sampleID", "reason"])

        data = []
        for sample_id in sorted(self._excluded_sample_ids):
            data.append(
                {
                    "sampleID": sample_id,
                    "reason": self._exclusion_source.get(sample_id, "unknown"),
                }
            )

        return pd.DataFrame(data)

    def clear_exclusions(self):
        """Remove all sample exclusions.

        Example:
            >>> # Clear all exclusions and start fresh
            >>> locator.clear_exclusions()
            Cleared 3 sample exclusions
        """
        n_cleared = len(self._excluded_sample_ids)
        self._excluded_sample_ids.clear()
        self._exclusion_source.clear()
        print(f"Cleared {n_cleared} sample exclusions")

    def exclude_samples_by_condition(
        self, condition_func, sample_df=None, reason="condition"
    ):
        """Exclude samples based on a condition function.

        Args:
            condition_func: Function that takes a DataFrame and returns boolean Series
                           True values indicate samples to exclude
            sample_df: DataFrame to evaluate. If None, uses self.sample_data
            reason: Reason string for exclusion tracking

        Example:
            >>> # Exclude samples with high prediction error
            >>> locator.exclude_samples_by_condition(
            ...     lambda df: df['error'] > 100,
            ...     sample_df=error_results,
            ...     reason="high_error"
            ... )
            Excluded 5 samples (reason: high_error)
            Total excluded samples: 5

            >>> # Exclude samples with low genotype rate
            >>> locator.exclude_samples_by_condition(
            ...     lambda df: df['genotype_rate'] < 0.8,
            ...     reason="low_genotype_rate"
            ... )
        """
        if sample_df is None:
            if not hasattr(self, "sample_data"):
                raise ValueError("No sample data available. Load data first.")
            sample_df = self.sample_data

        # Apply condition
        mask = condition_func(sample_df)
        samples_to_exclude = sample_df.loc[mask, "sampleID"].tolist()

        if samples_to_exclude:
            self.exclude_samples(samples_to_exclude, reason=reason)
        else:
            print("No samples matched the exclusion condition")


# Import EnsembleLocator from ensemble.py
from .ensemble import EnsembleLocator  # noqa: E402, F401
