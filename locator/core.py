"""Core functionality for locator"""

import numpy as np
import pandas as pd
import allel
import zarr
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


class Locator:
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
#            "weight_samples": False,
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

    def _load_from_zarr(self, zarr_path):
        """Load genotypes from zarr file.

        Args:
            zarr_path: Path to zarr file containing genotype data

        Returns:
            tuple: (genotypes, samples) where:
                - genotypes is an allel.GenotypeArray containing genetic data
                - samples is a numpy array of sample IDs
        """
        print("reading zarr")
        callset = zarr.open_group(zarr_path, mode="r")
        gt = callset["calldata/GT"]
        genotypes = allel.GenotypeArray(gt[:])
        samples = callset["samples"][:]
        return genotypes, samples

    def _load_from_vcf(self, vcf_path):
        """Load genotypes from VCF file.

        Args:
            vcf_path: Path to VCF file containing genotype data

        Returns:
            tuple: (genotypes, samples) where:
                - genotypes is an allel.GenotypeArray containing genetic data
                - samples is a numpy array of sample IDs

        Raises:
            ValueError: If VCF file cannot be read
        """
        print("reading VCF")
        vcf = allel.read_vcf(vcf_path)
        if vcf is None:
            raise ValueError(f"Could not read VCF file: {vcf_path}")
        genotypes = allel.GenotypeArray(vcf["calldata/GT"])
        samples = vcf["samples"]
        return genotypes, samples

    def _load_from_matrix(self, matrix_path):
        """Load genotypes from matrix file.

        Args:
            matrix_path: Path to tab-delimited matrix file containing genotype data.
                File should have a header row with 'sampleID' as first column,
                followed by variant columns. Each row contains genotype counts (0,1,2)
                for one sample.

        Returns:
            tuple: (genotypes, samples) where:
                - genotypes is an allel.GenotypeArray containing genetic data
                - samples is a numpy array of sample IDs
        """
        gmat = pd.read_csv(matrix_path, sep="\t")
        samples = np.array(gmat["sampleID"])
        gmat = gmat.drop(labels="sampleID", axis=1)
        if not np.all(np.isin(gmat, [0, 1, 2])):
            raise ValueError("Genotype values must be 0, 1, or 2")
        gmat = np.array(gmat, dtype="int8")

        # Convert to haplotype format
        hmat = None
        for i in range(gmat.shape[0]):
            h1 = []
            h2 = []
            for j in range(gmat.shape[1]):
                count = gmat[i, j]
                if count == 0:
                    h1.append(0)
                    h2.append(0)
                elif count == 1:
                    h1.append(1)
                    h2.append(0)
                elif count == 2:
                    h1.append(1)
                    h2.append(1)
            if i == 0:
                hmat = h1
                hmat = np.vstack((hmat, h2))
            else:
                hmat = np.vstack((hmat, h1))
                hmat = np.vstack((hmat, h2))

        genotypes = allel.HaplotypeArray(np.transpose(hmat)).to_genotypes(ploidy=2)
        return genotypes, samples

    def load_genotypes(self, vcf=None, zarr=None, matrix=None):
        """Load genotype data from various input sources.

        This method can load genotype data from:
        1. A stored DataFrame provided during initialization
        2. A VCF file
        3. A zarr file
        4. A tab-delimited matrix file

        For windowed analysis, SNP positions must be available either from:
        - Column names in the genotype DataFrame
        - The zarr file's variants/POS array

        Args:
            vcf (str, optional): Path to VCF format genotype data
            zarr (str, optional): Path to zarr format genotype data
            matrix (str, optional): Path to tab-delimited matrix file

        Returns:
            tuple: (genotypes, samples) where:
                - genotypes is an allel.GenotypeArray of shape (n_sites, n_samples, 2)
                - samples is a numpy array of sample IDs

        Examples:
            >>> # Using stored DataFrame from initialization
            >>> locator = Locator({
            ...     "genotype_data": geno_df,  # DataFrame with genotypes
            ...     "sample_data": coords_df   # DataFrame with coordinates
            ... })
            >>> genotypes, samples = locator.load_genotypes()

            >>> # Using zarr file (recommended for windowed analysis)
            >>> locator = Locator({"sample_data": coords_df})
            >>> genotypes, samples = locator.load_genotypes(zarr="path/to/geno.zarr")

            >>> # Using VCF file
            >>> genotypes, samples = locator.load_genotypes(vcf="path/to/geno.vcf")

            >>> # Using matrix file
            >>> genotypes, samples = locator.load_genotypes(matrix="path/to/geno.txt")

        Raises:
            ValueError: If no input source is provided or if input format is invalid
        """
        # First load sample data if not already loaded
        if not hasattr(self, "_sample_data_df") and "sample_data" in self.config:
            sample_df = pd.read_csv(self.config["sample_data"], sep="\t")
            required_cols = ["sampleID", "x", "y"]
            if not all(col in sample_df.columns for col in required_cols):
                raise ValueError(f"sample_data must contain columns: {required_cols}")
            self._sample_data_df = sample_df

        # Use stored DataFrame if available
        if hasattr(self, "_genotype_df"):
            print("using stored genotype DataFrame")
            geno_df = self._genotype_df
            # Convert samples to Python's native str type
            samples = np.array([str(x) for x in geno_df.index], dtype=object)
            # Store positions for windowed analysis if not already set
            if self.positions is None:
                try:
                    self.positions = geno_df.columns.astype(float).values
                except ValueError:
                    raise ValueError(
                        "Column names must be convertible to integers (SNP positions)"
                    )

            # Convert DataFrame values to genotype array format
            # Shape needs to be (n_sites, n_samples, 2) for compatibility
            genotypes = np.zeros((geno_df.shape[1], geno_df.shape[0], 2), dtype=int)

            # Convert each genotype count to allele counts
            # e.g., 0 -> [0,0], 1 -> [1,0], 2 -> [1,1]
            for i, count in enumerate([0, 1, 2]):
                mask = geno_df.values.T == count
                if count == 0:
                    continue  # already zeros
                elif count == 1:
                    genotypes[mask, 0] = 1
                else:  # count == 2
                    genotypes[mask] = 1

            return allel.GenotypeArray(genotypes), samples

        # Load from zarr
        elif zarr is not None:
            return self._load_from_zarr(zarr)

        # Load from VCF
        elif vcf is not None:
            print("reading VCF")
            vcf_data = allel.read_vcf(vcf, log=sys.stderr)
            if vcf_data is None:
                raise ValueError(f"Could not read VCF file: {vcf}")
            genotypes = allel.GenotypeArray(vcf_data["calldata/GT"])
            samples = vcf_data["samples"]
            return genotypes, samples

        # Load from matrix
        elif matrix is not None:
            print("reading matrix")
            gmat = pd.read_csv(matrix, sep="\t")
            samples = np.array(gmat["sampleID"])
            gmat = gmat.drop(labels="sampleID", axis=1)
            if not np.all(np.isin(gmat, [0, 1, 2])):
                raise ValueError("Genotype values must be 0, 1, or 2")
            gmat = np.array(gmat, dtype="int8")

            # Convert to haplotype format
            hmat = None
            for i in range(gmat.shape[0]):
                h1 = []
                h2 = []
                for j in range(gmat.shape[1]):
                    count = gmat[i, j]
                    if count == 0:
                        h1.append(0)
                        h2.append(0)
                    elif count == 1:
                        h1.append(1)
                        h2.append(0)
                    elif count == 2:
                        h1.append(1)
                        h2.append(1)
                if i == 0:
                    hmat = h1
                    hmat = np.vstack((hmat, h2))
                else:
                    hmat = np.vstack((hmat, h1))
                    hmat = np.vstack((hmat, h2))

            genotypes = allel.HaplotypeArray(np.transpose(hmat)).to_genotypes(ploidy=2)
            return genotypes, samples

        else:
            raise ValueError(
                "No genotype data provided. Either initialize with genotype_data DataFrame "
                "or provide vcf/zarr/matrix path."
            )

    def _split_train_test(self, genotypes, locations, train_split=0.9):
        """Split genotype and location data into training and test sets.

        Args:
            genotypes: GenotypeArray containing genetic data for all samples
            locations: Array of geographic coordinates (x,y) for each sample,
                      with NaN values for samples with unknown locations
            train_split: Proportion of samples to use for training (default: 0.9)

        Returns:
            tuple: (train_idx, test_idx, train_gen, test_gen, train_locs, test_locs, pred_idx, pred_gen)
                train_idx: Indices of training samples
                test_idx: Indices of test samples
                train_gen: Genotype data for training samples
                test_gen: Genotype data for test samples
                train_locs: Location data for training samples
                test_locs: Location data for test samples
                pred_idx: Indices of samples with unknown locations
                pred_gen: Genotype data for samples with unknown locations
        """
        # Get indices of samples with known locations
        train = np.argwhere(~np.isnan(locations[:, 0]))
        train = np.array([x[0] for x in train])
        # Get indices of samples with unknown locations
        pred = np.array([x for x in range(len(locations)) if x not in train])

        # Split known locations into train/test
        test = np.random.choice(
            train, round((1 - train_split) * len(train)), replace=False
        )
        train = np.array([x for x in train if x not in test])

        # Prepare data arrays
        traingen = np.transpose(genotypes[:, train])
        testgen = np.transpose(genotypes[:, test])
        trainlocs = locations[train]
        testlocs = locations[test]
        predgen = np.transpose(genotypes[:, pred])

        return train, test, traingen, testgen, trainlocs, testlocs, pred, predgen

    def _create_callbacks(self, boot=0):
        """Create Keras callbacks for training.

        Args:
            boot: Bootstrap replicate number (default: 0)

        Returns:
            list: List of Keras callbacks [ModelCheckpoint, EarlyStopping, ReduceLROnPlateau]
        """
        filepath = (
            f"{self.config['out']}_boot{boot}.weights.h5"
            if self.config.get("bootstrap", False)
            else f"{self.config['out']}.weights.h5"
        )

        checkpointer = keras.callbacks.ModelCheckpoint(
            filepath=filepath,
            verbose=self.config.get("keras_verbose", 1),
            save_best_only=True,
            save_weights_only=True,
            monitor="val_loss",
            save_freq="epoch",
        )

        earlystop = keras.callbacks.EarlyStopping(
            monitor="val_loss",
            min_delta=0,
            patience=self.config.get("patience", 100),
        )

        reducelr = keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=self.config.get("patience", 100) // 6,
            verbose=self.config.get("keras_verbose", 1),
            mode="auto",
            min_delta=0,
            cooldown=0,
            min_lr=0,
        )

        return [checkpointer, earlystop, reducelr]

    def set_sample_weights(self, wdict):
        """Set sample weights for training.
        Args:
            wdict (dict): Dictionary returned by utils.weight_samples() containing sample weights.
        """
        self.sample_weights = wdict
        self.config["weight_samples"]["enabled"] = True
        for key, value in wdict.items():
                self.config["weight_samples"][key] = value


    def train(
        self,
        *,  # Force keyword arguments
        genotypes,
        samples,
        sample_data_file=None,
        boot=None,

        train_gen=None,
        test_gen=None,
        pred_gen=None,
        train_locs=None,
        test_locs=None,
        setup_only=False,
        weight_samples=False,
        weight_method=None,
    ):
        """Train the Locator model on genotype and location data.

        This method trains the neural network model to predict geographic locations from genetic data.
        It supports both standard training and advanced workflows such as bootstrapping, by accepting
        pre-processed genotype and location arrays. The model is configured using the parameters
        provided at initialization.

        Args:
            genotypes (allel.GenotypeArray or np.ndarray): Genotype data for all samples. Should be of shape (n_sites, n_samples, ploidy).
            samples (np.ndarray): Array of sample IDs corresponding to the genotype data.
            sample_data_file (str, optional): Path to a tab-delimited file with columns 'sampleID', 'x', 'y' for sample locations. Used if not provided in config or as a DataFrame.
            boot (int, optional): Bootstrap replicate number. Used for bootstrapping analyses. Defaults to None.
            train_gen (np.ndarray, optional): Pre-processed training genotype data. Used for bootstrapping. If None, will be generated from `genotypes`. Defaults to None.
            test_gen (np.ndarray, optional): Pre-processed test genotype data. Used for bootstrapping. If None, will be generated from `genotypes`. Defaults to None.
            pred_gen (np.ndarray, optional): Pre-processed prediction genotype data. Used for bootstrapping. If None, will be generated from `genotypes`. Defaults to None.
            train_locs (np.ndarray, optional): Pre-processed training locations. Used for bootstrapping. If None, will be generated from sample data. Defaults to None.
            test_locs (np.ndarray, optional): Pre-processed test locations. Used for bootstrapping. If None, will be generated from sample data. Defaults to None.
            setup_only (bool, optional): If True, only sets up the model and data without training. Defaults to False.

        Returns:
            keras.callbacks.History or None: The Keras training history object if training is performed, or None if `setup_only` is True.

        Raises:
            ValueError: If required sample data is missing or improperly formatted.

        Example:
            >>> # Standard training
            >>> loc = Locator({"out": "analysis", "sample_data": "samples.txt", "zarr": "genotypes.zarr"})
            >>> genotypes, samples = loc.load_genotypes(zarr="genotypes.zarr")
            >>> history = loc.train(genotypes=genotypes, samples=samples)

            >>> # Bootstrapping with pre-processed data
            >>> history = loc.train(
            ...     genotypes=None,
            ...     samples=samples,
            ...     boot=1,
            ...     train_gen=boot_train_gen,
            ...     test_gen=boot_test_gen,
            ...     pred_gen=boot_pred_gen,
            ...     train_locs=boot_train_locs,
            ...     test_locs=boot_test_locs
            ... )
        """
        # Store samples
        self.samples = samples

        # Get sorted sample data and locations
        if hasattr(self, "_sample_data_df"):
            # Use stored DataFrame
            sample_data, locs = self.sort_samples(samples)
        else:
            # Use file path
            sample_data_path = sample_data_file or self.config.get("sample_data")
            if not isinstance(sample_data_path, str):
                raise ValueError(
                    "sample_data file path must be provided in config or as argument "
                    "when not using DataFrame input"
                )
            sample_data, locs = self.sort_samples(samples, sample_data_file)

        # Normalize locations
        self.meanlong, self.sdlong, self.meanlat, self.sdlat, self.unnormedlocs, normalized_locs = (
            normalize_locs(locs)
        )

        # Filter SNPs if not using pre-processed data
        if train_gen is None:
            filtered_genotypes = filter_snps(
                genotypes,
                min_mac=self.config.get("min_mac", 2),
                max_snps=self.config.get("max_SNPs"),
                impute=self.config.get("impute_missing", False),
            )

            # Split data
            (
                train,
                test,
                self.traingen,
                self.testgen,
                trainlocs,
                testlocs,
                pred,
                self.predgen,
            ) = self._split_train_test(
                filtered_genotypes,
                normalized_locs,
                train_split=self.config.get("train_split", 0.9),
            )

            # Apply sample weighting only if enabled in config
            if self.config.get("weight_samples", {}).get("enabled", False):
                if self.sample_weights is not None:
                    raise ValueError(
                        "Sample weights already calculated. "
                        "Set weight_samples to False in config to disable."
                    )
                wmethod = self.config.get("weight_samples", {}).get("method")
                self.sample_weights = weight_samples(wmethod,
                                                    trainlocs=self.unnormedlocs,
                                                    trainsamps=self.samples[train_idx_final],
                                                    weightdf=self.config.get("weight_samples", {}).get("dataframe"),
                                                    xbins=self.config.get("weight_samples", {}).get("xbins"),
                                                    ybins=self.config.get("weight_samples", {}).get("ybins"),
                                                    lam=self.config.get("weight_samples", {}).get("lam"),
                                                    bandwidth=self.config.get("weight_samples", {}).get("bandwidth"),
                                                    )
            # Store prediction indices
            self.pred_indices = pred
        else:
            # Use pre-processed data (for bootstrapping)
            self.traingen = train_gen
            self.testgen = test_gen
            self.predgen = pred_gen
            # Use provided locations if available
            if train_locs is not None and test_locs is not None:
                trainlocs = train_locs
                testlocs = test_locs
            else:
                # Get train/test indices and locations from original split
                train = np.where(~np.isnan(normalized_locs[:, 0]))[0]
                test = np.random.choice(
                    train,
                    round((1 - self.config.get("train_split", 0.9)) * len(train)),
                    replace=False,
                )
                train = np.array([x for x in train if x not in test])
                trainlocs = normalized_locs[train]
                testlocs = normalized_locs[test]

        # Store both training and test locations
        self.trainlocs = trainlocs
        self.testlocs = testlocs

        # Create and train model if not already created
        if self.model is None:
            # Decide which loss function to use based on the config
            loss_fn = None
            if self.config.get("use_range_penalty"):
                from .models import loss_with_range_penalty

                assert (
                    self.config.get("species_range_shapefile") is not None
                ), "species_range_shapefile must be provided if use_range_penalty is True"
                assert (
                    self.config.get("resolution") is not None
                ), "resolution must be provided if use_range_penalty is True"
                # Rasterize the species range from the provided shapefile.
                mask_tensor, mask_transform = rasterize_species_range(
                    self.config["species_range_shapefile"],
                    resolution=self.config.get("raster_resolution", 0.1),
                )

                loss_fn = lambda y_true, y_pred: loss_with_range_penalty(
                    y_true,
                    y_pred,
                    mask_tensor=mask_tensor,
                    transform=mask_transform,
                    resolution=self.config.get("resolution", 0.05),
                    penalty_weight=self.config.get("penalty_weight", 1.0),
                )

            self.model = create_network(
                input_shape=self.traingen.shape[1],
                width=self.config.get("width", 256),
                n_layers=self.config.get("nlayers", 8),
                dropout_prop=self.config.get("dropout_prop", 0.25),
                optimizer_config={
                    "algo": self.config.get("optimizer_algo", "adam"),
                    "learning_rate": self.config.get("learning_rate", 0.001),
                    "weight_decay": self.config.get("weight_decay", 0.004),
                },
                loss_fn=loss_fn,
            )

        # Return early if setup_only
        if setup_only:
            return None

        callbacks = self._create_callbacks(boot=boot)

        self.history = self.model.fit(
            self.traingen,
            trainlocs,
            epochs=self.config.get("max_epochs", 5000),
            batch_size=self.config.get("batch_size", 32),
            shuffle=True,
            verbose=self.config.get("keras_verbose", 1),
            validation_data=(self.testgen, testlocs),
            callbacks=callbacks,
            sample_weights = None if self.sample_weights is None else self.sample_weights['sample_weights'],
        )

        # Save training history
        hist_df = pd.DataFrame(self.history.history)
        hist_df.to_csv(f"{self.config['out']}_history.txt", sep="\t", index=False)

        return self.history

    def predict(
        self,
        boot=0,
        verbose=True,
        prediction_genotypes=None,
        return_df=False,
        save_preds_to_disk=True,
    ):
        """Make predictions for samples with unknown locations.

        Args:
            boot (int, optional): Bootstrap replicate number. Defaults to 0.
            verbose (bool, optional): Whether to print validation metrics. Defaults to True.
            prediction_genotypes (numpy.ndarray, optional): Override default prediction genotypes.
                Used for jacknife resampling. Defaults to None.
            return_df (bool, optional): Whether to return predictions as pandas DataFrame.
                Defaults to False.
            save_preds_to_disk (bool, optional): Whether to save predictions to disk.
                Defaults to True.
        Returns:
            numpy.ndarray or pandas.DataFrame: Array of predicted coordinates or DataFrame with
                x,y coordinates and sampleID columns
        """
        if self.model is None:
            raise ValueError("Model must be trained before prediction")

        # Use provided prediction genotypes if available, otherwise use stored ones
        predgen = (
            prediction_genotypes if prediction_genotypes is not None else self.predgen
        )

        # Get predictions
        predictions = self.model.predict(predgen)

        # Denormalize predictions
        predictions = np.array(
            [
                [x[0] * self.sdlong + self.meanlong, x[1] * self.sdlat + self.meanlat]
                for x in predictions
            ]
        )

        # Create DataFrame
        pred_df = pd.DataFrame(predictions, columns=["x", "y"])
        if hasattr(self, "samples") and hasattr(self, "pred_indices"):
            pred_df.insert(0, "sampleID", self.samples[self.pred_indices])

        # Save predictions to file
        outfile = (
            f"{self.config['out']}_boot{boot}_predlocs.txt"
            if self.config.get("bootstrap", False) or self.config.get("jacknife", False)
            else f"{self.config['out']}_predlocs.txt"
        )
        if save_preds_to_disk:
            pred_df.to_csv(outfile, index=False)

        if return_df:
            return pred_df

        return predictions

    def sort_samples(self, samples=None, sample_data_file=None):
        """Sort samples and match with location data.

        This method matches samples with their location data and ensures consistent ordering
        between genotype and location data. It can use either a stored DataFrame from
        initialization or a provided sample data file.

        Args:
            samples (numpy.ndarray): Array of sample IDs from the genotype data
            sample_data_file (str, optional): Override path to tab-delimited file with
                columns 'sampleID', 'x', 'y'. If not provided, uses stored sample data.

        Returns:
            tuple: A tuple containing:
                - sample_data (pandas.DataFrame): DataFrame with sample metadata and coordinates
                - locs (numpy.ndarray): Array of x,y coordinates for each sample

        Raises:
            ValueError: If samples not provided or if no sample data available
            ValueError: If sample IDs don't match between genotype and sample data
        """
        if samples is None:
            raise ValueError("samples must be provided")

        # Use stored DataFrame if available
        if hasattr(self, "_sample_data_df"):
            sample_data = self._sample_data_df.copy()
        else:
            # Get sample data file path
            sample_data_path = sample_data_file or self.config.get("sample_data")
            if not sample_data_path:
                raise ValueError(
                    "sample_data must be provided in config or as argument"
                )
            # Read sample data file
            sample_data = pd.read_csv(sample_data_path, sep="\t")

        # Ensure sampleID column exists
        if "sampleID" not in sample_data.columns:
            raise ValueError("sample_data must contain 'sampleID' column")

        # Ensure consistent string type for comparison
        sample_data = self._sample_data_df.copy()
        # Convert the sampleID column to match the type of samples
        sample_data["sampleID"] = sample_data["sampleID"].astype(str)

        # Verify sample order matches using the correct column name
        if not all(
            sample_data["sampleID"].iloc[x] == samples[x] for x in range(len(samples))
        ):
            raise ValueError(
                "Sample ordering failed! Check that sample IDs match the genotype data."
            )

        # Extract location data
        locs = np.array(sample_data[["x", "y"]])

        return sample_data, locs

    def plot_history(self, history):
        """Plot training history and prediction error.

        Creates a figure with two subplots showing the validation loss and training loss
        over epochs. Saves the plot to a PDF file using the output prefix specified in config.

        Args:
            history: keras.callbacks.History object containing training history
        """
        if self.config.get("plot_history", False):
            plt.switch_backend("agg")
            fig = plt.figure(figsize=(4, 1.5), dpi=200)
            plt.rcParams.update({"font.size": 7})
            ax1 = fig.add_axes([0, 0, 0.4, 1])
            ax1.plot(history.history["val_loss"][3:], "-", color="black", lw=0.5)
            ax1.set_xlabel("Validation Loss")
            ax2 = fig.add_axes([0.55, 0, 0.4, 1])
            ax2.plot(history.history["loss"][3:], "-", color="black", lw=0.5)
            ax2.set_xlabel("Training Loss")
            fig.savefig(self.config["out"] + "_fitplot.pdf", bbox_inches="tight")

    def run_windows(
        self,
        genotypes,
        samples,
        window_start=0,
        window_size=5e5,
        window_stop=None,
        return_df=False,
        save_full_pred_matrix=True,
    ):
        """Run windowed prediction analysis.

        Args:
            genotypes: GenotypeArray containing genetic data
            samples: Array of sample IDs
            window_start: Start position for windows (default: 0)
            window_size: Size of windows in base pairs (default: 500kb)
            window_stop: Stop position for windows (default: None)
            ...
        """
        # Store samples
        self.samples = samples

        # Get positions if not already stored
        if not hasattr(self, "positions"):
            if hasattr(self, "_genotype_df"):
                # Use positions from DataFrame columns
                self.positions = np.array(self._genotype_df.columns, dtype=int)
            elif self.config.get("zarr"):
                # Get positions from zarr file
                callset = zarr.open_group(self.config["zarr"], mode="r")
                self.positions = callset["variants/POS"][:]
            else:
                raise ValueError(
                    "SNP positions required for windowed analysis. Use zarr input or "
                    "genotype DataFrame with position-labeled columns."
                )

        if window_stop is None:
            window_stop = max(self.positions)

        windows = range(int(window_start), int(window_stop), int(window_size))

        # Initial training to set up model and data
        first_window = (self.positions >= int(window_start)) & (
            self.positions < int(window_start + window_size)
        )
        if sum(first_window) > 0:
            window_genos = genotypes[first_window, :, :]
            self.train(genotypes=window_genos, samples=samples)

        # Create lists to store predictions
        pred_dfs = []

        print("starting window analysis")
        for start in tqdm(windows):
            stop = start + int(window_size)
            in_window = (self.positions >= start) & (self.positions < stop)

            if sum(in_window) > 0:
                # Get genotypes for this window
                window_genos = genotypes[in_window, :, :]

                # Clear existing model
                self.model = None

                # Train on window data
                self.train(genotypes=window_genos, samples=samples)

                # Get predictions using self.predgen which is already properly formatted
                preds = self.predict(
                    return_df=True, save_preds_to_disk=not save_full_pred_matrix
                )

                if return_df:
                    # Rename columns to include window start
                    boot_preds = preds[["x", "y"]].copy()
                    boot_preds.columns = [f"x_win{start}", f"y_win{start}"]
                    pred_dfs.append(boot_preds)

                # Clear keras session
                keras.backend.clear_session()

        if return_df:
            # Concatenate all predictions and add sampleIDs
            all_predictions = pd.concat([preds[["sampleID"]], *pred_dfs], axis=1)

            if save_full_pred_matrix:
                all_predictions.to_csv(
                    f"{self.config['out']}_windows_predlocs.csv", index=False
                )
            return all_predictions

        return None

    def run_jacknife(
        self,
        genotypes,
        samples,
        prop=0.05,
        return_df=False,
        save_full_pred_matrix=True,
    ):
        """Run jacknife analysis by dropping SNPs.

        Args:
            genotypes: Array of genotype data
            samples: Sample IDs corresponding to genotypes
            prop (float, optional): Proportion of SNPs to drop in each replicate.
                Defaults to 0.05.
            return_df (bool, optional): Whether to return DataFrame of all predictions.
                Defaults to False.
            save_full_pred_matrix (bool, optional): Whether to save the full prediction matrix.
                Defaults to True.

        Returns:
            pandas.DataFrame or None: If return_df=True, returns DataFrame containing
                all predictions, with columns named 'x_0', 'y_0', 'x_1', 'y_1', etc.
                for each jacknife replicate. Row index contains sample IDs.
        """
        # Store samples
        self.samples = samples

        # Set jacknife flag in config
        self.config["jacknife"] = True

        # Set up prediction indices if not already done
        if not hasattr(self, "pred_indices"):
            # Get sample data
            if isinstance(self.config["sample_data"], pd.DataFrame):
                sample_data = self.config["sample_data"]
            else:
                sample_data = pd.read_csv(self.config["sample_data"], sep="\t")
            # Find samples without locations (NA in x or y)
            pred = sample_data.index[sample_data.x.isna() | sample_data.y.isna()].values
            # Convert to indices in the samples array
            self.pred_indices = np.where(
                np.isin(np.array(samples), sample_data.index[pred])
            )[0]

        # Create lists to store predictions
        pred_dfs = []
        preds = None

        # Initial training to set up model (but don't output predictions)
        self.train(genotypes=genotypes, samples=samples)

        print("starting jacknife resampling")
        af = []
        # Convert genotypes to allele counts first
        ac = genotypes.to_allele_counts()[:, :, 1]  # Get counts of alternate allele

        # Calculate allele frequencies
        for i in range(ac.shape[0]):
            freq = np.sum(ac[i, :]) / (ac.shape[1] * 2)
            af.append(freq)
        af = np.array(af)

        for boot in tqdm(range(self.config.get("nboots", 50))):
            callbacks = self._create_callbacks(boot)
            pg = copy.deepcopy(self.predgen)

            sites_to_remove = np.random.choice(
                pg.shape[1], int(pg.shape[1] * prop), replace=False
            )

            for i in sites_to_remove:
                pg[:, i] = np.random.binomial(2, af[i], size=pg.shape[0])

            # Get predictions
            preds = self.predict(
                boot=boot,
                verbose=False,
                prediction_genotypes=pg,
                return_df=True,
                save_preds_to_disk=not save_full_pred_matrix,
            )

            # Rename columns to include boot number
            boot_preds = preds[["x", "y"]].copy()
            boot_preds.columns = [f"x_{boot}", f"y_{boot}"]
            pred_dfs.append(boot_preds)

        if return_df:
            # Concatenate all predictions and add sampleIDs
            all_predictions = pd.concat([preds[["sampleID"]], *pred_dfs], axis=1)

            if save_full_pred_matrix:
                all_predictions.to_csv(
                    f"{self.config['out']}_jacknife_predlocs.csv", index=False
                )
            return all_predictions

        return None

    def run_bootstraps(
        self,
        genotypes,
        samples,
        n_bootstraps=50,
        return_df=False,
        save_full_pred_matrix=True,
    ):
        # Store samples
        self.samples = samples

        # Set bootstrap flag in config
        self.config["bootstrap"] = True
        self.config["nboots"] = n_bootstraps

        # Initial training to set up model and data
        self.train(genotypes=genotypes, samples=samples)

        # Store original locations
        original_trainlocs = self.trainlocs
        original_testlocs = self.testlocs

        # Create lists to store predictions
        pred_dfs = []

        print("starting bootstrap resampling")

        for boot in tqdm(range(n_bootstraps)):
            # Set random seed
            np.random.seed(np.random.choice(range(int(1e6)), 1))

            # Create copies of data
            traingen2 = copy.deepcopy(self.traingen)
            testgen2 = copy.deepcopy(self.testgen)
            predgen2 = copy.deepcopy(self.predgen)

            # Resample sites with replacement
            site_order = np.random.choice(
                traingen2.shape[1], traingen2.shape[1], replace=True
            )

            # Reorder sites in all datasets
            traingen2 = traingen2[:, site_order]
            testgen2 = testgen2[:, site_order]
            predgen2 = predgen2[:, site_order]

            # Clear existing model
            self.model = None

            # Train on bootstrapped data with original locations
            self.train(
                genotypes=None,
                samples=samples,
                boot=boot,
                train_gen=traingen2,
                test_gen=testgen2,
                pred_gen=predgen2,
                train_locs=original_trainlocs,
                test_locs=original_testlocs,
            )

            # Get predictions
            preds = self.predict(
                boot=boot,
                verbose=False,
                prediction_genotypes=predgen2,
                return_df=True,
                save_preds_to_disk=not save_full_pred_matrix,
            )

            if return_df:
                # Rename columns to include boot number
                boot_preds = preds[["x", "y"]].copy()
                boot_preds.columns = [f"x_{boot}", f"y_{boot}"]
                pred_dfs.append(boot_preds)

            # Clear keras session
            keras.backend.clear_session()

        if return_df:
            # Concatenate all predictions and add sampleIDs
            all_predictions = pd.concat([preds[["sampleID"]], *pred_dfs], axis=1)

            if save_full_pred_matrix:
                all_predictions.to_csv(
                    f"{self.config['out']}_bootstrap_predlocs.csv", index=False
                )
            return all_predictions

        return None

    def train_holdout(
        self,
        genotypes,
        samples,
        k=10,
        holdout_indices=None,
    ):
        """Train the model while holding out samples with known locations.

        Args:
            genotypes: Array of genotype data
            samples: Sample IDs corresponding to genotypes
            k: Number of samples to hold out (ignored if holdout_indices provided)
            holdout_indices: Optional specific indices of samples to hold out

        Returns:
            keras.callbacks.History object from model training
        """
        # Store samples
        self.samples = samples

        # Get sample data and locations
        if hasattr(self, "_sample_data_df"):
            # Use stored DataFrame
            sample_data, locs = self.sort_samples(samples)
        else:
            # Use file path
            sample_data_path = self.config.get("sample_data")
            if not sample_data_path:
                raise ValueError("sample_data file path must be provided in config")
            sample_data, locs = self.sort_samples(samples, sample_data_path)

        # Get indices of samples with known locations
        known_idx = np.argwhere(~np.isnan(locs[:, 0]))
        known_idx = np.array([x[0] for x in known_idx])

        # Set holdout indices
        if holdout_indices is not None:
            # Verify provided indices are valid
            if not all(idx in known_idx for idx in holdout_indices):
                raise ValueError(
                    "All holdout_indices must be indices of samples with known locations"
                )
            holdout_idx = np.array(holdout_indices)
        else:
            # Random selection
            if k >= len(known_idx):
                raise ValueError(
                    f"k ({k}) must be less than number of samples with known locations ({len(known_idx)})"
                )
            holdout_idx = np.random.choice(known_idx, k, replace=False)

        # Create mask for non-holdout samples
        mask = np.ones(len(locs), dtype=bool)
        mask[holdout_idx] = False
        train_idx = known_idx[~np.isin(known_idx, holdout_idx)]

        # Filter SNPs
        filtered_genotypes = filter_snps(
            genotypes,
            min_mac=self.config.get("min_mac", 2),
            max_snps=self.config.get("max_SNPs"),
            impute=self.config.get("impute_missing", False),
        )

        # Split remaining samples into train/test
        test_size = round((1 - self.config.get("train_split", 0.9)) * len(train_idx))
        test_idx = np.random.choice(train_idx, test_size, replace=False)
        train_idx_final = np.array([x for x in train_idx if x not in test_idx])

        # Prepare training data arrays
        self.traingen = np.transpose(filtered_genotypes[:, train_idx_final])
        self.testgen = np.transpose(filtered_genotypes[:, test_idx])

        # Now normalize locations using only training data
        train_locs = locs[train_idx_final]
        self.trainIDs = samples[train_idx_final]
        self.meanlong, self.sdlong, self.meanlat, self.sdlat, self.unnormedlocs, normalized_train_locs = (
            normalize_locs(train_locs)
        )

        # Apply sample weighting only if enabled in config
        if self.config.get("weight_samples", {}).get("enabled", False):
            if self.sample_weights is not None:
                warnings.warn(
                    """Sample weights already calculated. 
                    Set locator.sample_weights to None in config to disable."""
                )
            else:
                wmethod = self.config.get("weight_samples", {}).get("method")
                self.sample_weights = weight_samples(wmethod,
                                                    trainlocs=self.unnormedlocs,
                                                    trainsamps=self.samples[train_idx_final],
                                                    weightdf=self.config.get("weight_samples", {}).get("dataframe"),
                                                    xbins=self.config.get("weight_samples", {}).get("xbins"),
                                                    ybins=self.config.get("weight_samples", {}).get("ybins"),
                                                    lam=self.config.get("weight_samples", {}).get("lam"),
                                                    bandwidth=self.config.get("weight_samples", {}).get("bandwidth"),
                                                    )


        # Normalize test and holdout locations using same parameters
        test_locs = locs[test_idx]
        normalized_test_locs = np.array(
            [
                [
                    (x[0] - self.meanlong) / self.sdlong,
                    (x[1] - self.meanlat) / self.sdlat,
                ]
                for x in test_locs
            ]
        )

        holdout_locs = locs[holdout_idx]
        normalized_holdout_locs = np.array(
            [
                [
                    (x[0] - self.meanlong) / self.sdlong,
                    (x[1] - self.meanlat) / self.sdlat,
                ]
                for x in holdout_locs
            ]
        )

        # Store training and test data
        self.trainlocs = normalized_train_locs
        self.testlocs = normalized_test_locs

        # Store holdout data
        self.holdout_idx = holdout_idx
        self.holdout_gen = np.transpose(filtered_genotypes[:, holdout_idx])
        self.holdout_locs = normalized_holdout_locs

        # Create new model (force recreation)
        loss_fn = None
        if self.config.get("use_range_penalty"):
            from .models import loss_with_range_penalty, rasterize_species_range

            assert (
                self.config.get("species_range_shapefile") is not None
            ), "species_range_shapefile must be provided if use_range_penalty is True"
            assert (
                self.config.get("resolution") is not None
            ), "resolution must be provided if use_range_penalty is True"
            mask_tensor, mask_transform = rasterize_species_range(
                self.config["species_range_shapefile"],
                resolution=self.config.get("raster_resolution", 0.1),
            )
            loss_fn = lambda y_true, y_pred: loss_with_range_penalty(
                y_true,
                y_pred,
                mask_tensor=mask_tensor,
                transform=mask_transform,
                resolution=self.config.get("resolution", 0.05),
                penalty_weight=self.config.get("penalty_weight", 1.0),
            )
        self.model = create_network(
            input_shape=self.traingen.shape[1],
            width=self.config.get("width", 256),
            n_layers=self.config.get("nlayers", 8),
            dropout_prop=self.config.get("dropout_prop", 0.25),
            optimizer_config={
                "algo": self.config.get("optimizer_algo", "adam"),
                "learning_rate": self.config.get("learning_rate", 0.001),
                "weight_decay": self.config.get("weight_decay", 0.004),
            },
            loss_fn=loss_fn,
        )

        callbacks = self._create_callbacks()

        def flip_genotypes(genotypes, locations, mask_rate=0.05):
            """Randomly flip genotype values with probability mask_rate"""
            mask = tf.random.uniform(tf.shape(genotypes)) < mask_rate
            return tf.where(mask, 1 - genotypes, genotypes), locations

        train_dataset = tf.data.Dataset.from_tensor_slices(
            (self.traingen, self.trainlocs, None if self.sample_weights is None else self.sample_weights['sample_weights'])
        )
        train_dataset = train_dataset.cache()
        train_dataset = train_dataset.shuffle(buffer_size=1000)

        # Apply augmentation only if enabled in config
        if self.config.get("augmentation", {}).get("enabled", False):
            flip_rate = self.config.get("augmentation", {}).get("flip_rate", 0.05)
            train_dataset = train_dataset.map(
                lambda x, y: flip_genotypes(x, y, mask_rate=flip_rate),
                num_parallel_calls=tf.data.AUTOTUNE,
            )

        train_dataset = train_dataset.batch(self.config.get("batch_size", 32))
        train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)

        validation_dataset = tf.data.Dataset.from_tensor_slices(
            (self.testgen, self.testlocs)
        )
        validation_dataset = validation_dataset.batch(self.config.get("batch_size", 32))
        validation_dataset = validation_dataset.prefetch(tf.data.AUTOTUNE)

        self.history = self.model.fit(
            train_dataset,
            epochs=self.config.get("max_epochs", 5000),
            verbose=self.config.get("keras_verbose", 0),
            validation_data=validation_dataset,
            callbacks=callbacks,
            #sample_weight=self.sample_weights,

        )

        # Save training history
        hist_df = pd.DataFrame(self.history.history)
        hist_df.to_csv(f"{self.config['out']}_history.txt", sep="\t", index=False)

        return self.history

    def predict_holdout(
        self,
        verbose=True,
        return_df=False,
        save_preds_to_disk=True,
        plot_summary=True,
        plot_map=True,
    ):
        """Predict locations for held out samples.

        Args:
            verbose: Print progress and metrics
            return_df: Return predictions as pandas DataFrame
            save_preds_to_disk: Save predictions to disk
            plot_summary: Display error summary plot in notebook (only if return_df=True)
            plot_map: Display map of predictions (only if plot_summary=True)

        Returns:
            If return_df is True, returns pandas DataFrame with predictions
            Otherwise returns None
        """
        if not hasattr(self, "holdout_gen") or not hasattr(self, "holdout_locs"):
            raise ValueError("No holdout data found. Run train_holdout() first.")

        if verbose:
            print("Predicting locations for holdout samples...")

        # Get predictions
        predictions = self.model.predict(self.holdout_gen, verbose=verbose)

        # Create output dataframe
        pred_df = pd.DataFrame(predictions, columns=["x", "y"])
        pred_df["sampleID"] = self.samples[self.holdout_idx]

        # Denormalize predictions
        pred_df["x"] = pred_df["x"] * self.sdlong + self.meanlong
        pred_df["y"] = pred_df["y"] * self.sdlat + self.meanlat
        pred_df["x_pred"] = pred_df["x"]
        pred_df["y_pred"] = pred_df["y"]

        if save_preds_to_disk:
            pred_df.to_csv(f"{self.config['out']}_holdout_predlocs.csv", index=False)

        if return_df:
            # If we're in a notebook and plot_summary is True, display the error plot
            try:
                from IPython.display import display
                import matplotlib.pyplot as plt
                from .plotting import plot_error_summary

                if plot_summary:
                    # Get sample data
                    if hasattr(self, "_sample_data_df"):
                        sample_data = self._sample_data_df
                    else:
                        sample_data = pd.read_csv(self.config["sample_data"], sep="\t")

                    # Create and display plot
                    plot_error_summary(
                        predictions=pred_df,
                        sample_data=sample_data,
                        plot_map=plot_map,
                        width=15,
                        height=5,
                        out_prefix=self.config.get("out"),
                    )

            except ImportError:
                # Not in a notebook, skip plotting
                pass

            return pred_df

        return predictions

    def run_holdouts(
        self,
        genotypes,
        samples,
        k=10,
        return_df=False,
        save_full_pred_matrix=True,
    ):
        """Run systematic holdouts across all samples with known locations.

        This function iteratively holds out groups of samples, trains models without them,
        and predicts their locations. This provides unbiased predictions for all samples
        with known locations since each prediction is made without using that sample in training.

        Args:
            genotypes: Array of genotype data
            samples: Sample IDs corresponding to genotypes
            k: Number of samples to hold out in each iteration
            return_df: Whether to return DataFrame of all predictions
            save_full_pred_matrix: Whether to save the full prediction matrix

        Returns:
            pandas.DataFrame or None: If return_df=True, returns DataFrame containing
                all predictions. Row index contains sample IDs.
        """
        # Store samples
        self.samples = samples

        # Get sample data and locations
        if hasattr(self, "_sample_data_df"):
            # Use stored DataFrame
            sample_data, locs = self.sort_samples(samples)
        else:
            # Use file path
            sample_data_path = self.config.get("sample_data")
            if not sample_data_path:
                raise ValueError("sample_data file path must be provided in config")
            sample_data, locs = self.sort_samples(samples, sample_data_path)

        # Get indices of samples with known locations
        known_idx = np.argwhere(~np.isnan(locs[:, 0]))
        known_idx = np.array([x[0] for x in known_idx])

        # Create list to store prediction DataFrames
        pred_dfs = []

        # Calculate number of iterations needed
        n_iterations = len(known_idx) // k
        if len(known_idx) % k != 0:
            n_iterations += 1

        print(f"Running {n_iterations} iterations, holding out {k} samples at a time")

        # Iterate through samples in groups of size k
        for i in tqdm(range(n_iterations)):
            start_idx = i * k
            end_idx = min(start_idx + k, len(known_idx))
            holdout_indices = known_idx[start_idx:end_idx]

            # Train model without holdout samples
            self.train_holdout(
                genotypes=genotypes, samples=samples, holdout_indices=holdout_indices
            )

            # Get predictions for holdout samples
            preds = self.predict_holdout(
                return_df=True,
                save_preds_to_disk=not save_full_pred_matrix,
                plot_summary=False,
                verbose=self.config.get("keras_verbose", 1),
            )

            if return_df:
                pred_dfs.append(preds)

            # Clear keras session to free memory
            keras.backend.clear_session()

        if return_df:
            # Concatenate all predictions
            all_predictions = pd.concat(pred_dfs, axis=0)

            if save_full_pred_matrix:
                all_predictions.to_csv(
                    f"{self.config['out']}_allholdouts_predlocs.csv", index=False
                )

            return all_predictions

        return None

    def run_jacknife_holdouts(
        self,
        genotypes,
        samples,
        k=10,
        holdout_indices=None,
        jacknife_prop=0.05,
        n_replicates=100,
        return_df=True,
    ):
        """Run jacknife analysis on predictions for held out samples.

        Args:
            genotypes: Array of genotype data
            samples: Sample IDs corresponding to genotypes
            k: Number of samples to hold out (ignored if holdout_indices provided)
            holdout_indices: Optional specific indices of samples to hold out
            jacknife_prop: Proportion of sites to mask in each jacknife replicate
            n_replicates: Number of jacknife replicates to run
            return_df: Whether to return results as DataFrame

        Returns:
            pandas DataFrame with columns:
                - sampleID: Sample identifier
                - x_0...x_n: Longitude predictions for n jacknife replicates
                - y_0...y_n: Latitude predictions for n jacknife replicates
        """
        # First train model without holdout samples
        self.train_holdout(
            genotypes=genotypes, samples=samples, k=k, holdout_indices=holdout_indices
        )

        # Calculate allele frequencies from training data
        af = []
        for i in range(self.holdout_gen.shape[1]):
            af.append(np.sum(self.traingen[:, i]) / (self.traingen.shape[0] * 2))
        af = np.array(af)

        # Store predictions for each replicate
        pred_dfs = []
        preds = None

        # Run jacknife replicates
        for rep in tqdm(range(n_replicates), desc="Running jacknife replicates"):
            # Copy holdout genotypes
            masked_gen = self.holdout_gen.copy()

            # Randomly mask sites
            sites_to_mask = np.random.choice(
                masked_gen.shape[1],
                size=int(masked_gen.shape[1] * jacknife_prop),
                replace=False,
            )

            # Replace masked sites with random draws from allele frequencies
            for site in sites_to_mask:
                masked_gen[:, site] = np.random.binomial(
                    2, af[site], size=masked_gen.shape[0]
                )

            # Get predictions for masked data
            predictions = self.model.predict(masked_gen, verbose=False)

            # Create prediction dataframe
            pred_df = pd.DataFrame(predictions, columns=["x", "y"])
            pred_df["sampleID"] = self.samples[self.holdout_idx]

            # Denormalize predictions
            pred_df["x"] = pred_df["x"] * self.sdlong + self.meanlong
            pred_df["y"] = pred_df["y"] * self.sdlat + self.meanlat

            # Rename columns to include replicate number
            boot_preds = pred_df[["x", "y"]].copy()
            boot_preds.columns = [f"x_rep{rep}", f"y_rep{rep}"]
            pred_dfs.append(boot_preds)

            # Store last predictions
            preds = pred_df

        if return_df:
            # Concatenate all predictions and add sampleIDs
            all_predictions = pd.concat([preds[["sampleID"]], *pred_dfs], axis=1)
            return all_predictions

        return preds

    def run_windows_holdouts(
        self,
        genotypes: allel.GenotypeArray,
        samples: np.ndarray,
        k: int = 10,
        holdout_indices: Optional[List[int]] = None,
        window_start: int = 0,
        window_size: int = 500000,
        window_step: Optional[int] = None,
        window_stop: Optional[int] = None,
        return_df: bool = True,
        save_full_pred_matrix: bool = True,
    ) -> pd.DataFrame:
        """Runs windowed analysis predicting locations for held-out samples.

        This method combines windowed SNP analysis with holdout cross-validation.
        For each sliding window defined by physical genomic coordinates:
        1. A specified set of samples (`k` or `holdout_indices`) is excluded
        from the training data for that window.
        2. A separate neural network model is trained using only the SNPs
        within the current window and the remaining training samples.
        3. The trained window-specific model is used to predict the geographic
        locations of the samples that were held out for that window.

        This process iterates through all defined windows across the genome.

        Args:
            genotypes (allel.GenotypeArray): Genotype data for all samples,
                shape (n_sites, n_samples, ploidy). SNP positions must be
                available either via `self._genotype_df.columns` or
                `self.config['zarr']`.
            samples (np.ndarray): Array of sample IDs corresponding to the
                genotype data.
            k (int, optional): The number of samples with known locations to
                hold out in each training iteration. Ignored if
                `holdout_indices` is provided. Defaults to 10.
            holdout_indices (Optional[List[int]], optional): A specific list
                of sample indices (relative to the `samples` array) to hold
                out. If provided, `k` is ignored. Defaults to None.
            window_start (int, optional): The starting genomic position (bp)
                for the first window. Defaults to 0.
            window_size (int, optional): The size of each genomic window in
                base pairs. Defaults to 500,000 (5e5).
            window_step (Optional[int], optional): The step size (bp) to move
                the window start position for each subsequent window. If None,
                defaults to `window_size` (non-overlapping windows).
            window_stop (Optional[int], optional): The genomic position (bp)
                at which to stop creating windows. If None, defaults to the
                maximum SNP position found.
            return_df (bool, optional): If True, returns a pandas DataFrame
                containing the predictions for the held-out samples from each
                window. Defaults to True.
            save_full_pred_matrix (bool, optional): If True and `return_df` is
                True, saves the consolidated prediction DataFrame (containing
                results from all windows) to a CSV file named
                '{out}_allwindows_predlocs.csv'. If False and `return_df` is
                True, individual window prediction files may be saved depending
                on internal logic. Defaults to True.

        Returns:
            pd.DataFrame: A DataFrame containing the location predictions for
                the held-out samples. It includes a 'sampleID' column and
                pairs of prediction columns for each window, named like
                'x_pos{start_pos}' and 'y_pos{start_pos}', where {start_pos}
                is the starting base pair position of the window.

        Raises:
            ValueError: If SNP positions required for windowing cannot be
                determined (e.g., not using zarr input or a genotype DataFrame
                with position-based columns).
            ValueError: If `k` is greater than or equal to the number of
                samples with known locations.
            ValueError: If `holdout_indices` contains invalid indices or
                indices for samples without known locations.
            ValueError: If `sample_data` information is missing.
        """
        # ... existing code ...

    def _repr_html_(self):
        """Return HTML representation of Locator instance for Jupyter notebooks."""
        html = [
            "<div style='font-family: monospace'>",
            "<h3>Locator Model</h3>",
            "<table>",
            "<tr><th style='text-align:left; padding:5px'>Configuration</th><th style='text-align:left; padding:5px'>Value</th></tr>",
        ]

        # Add key configuration parameters
        key_params = [
            "train_split",
            "batch_size",
            "min_mac",
            "max_SNPs",
            "width",
            "nlayers",
            "dropout_prop",
            "max_epochs",
            "optimizer_algo",
            "learning_rate",
            "weight_decay",
            "use_range_penalty",
            "species_range_shapefile",
            "resolution",
            "penalty_weight",
            "species_range_geom",
        ]

        for param in key_params:
            if param in self.config:
                html.append(
                    f"<tr><td style='padding:5px'>{param}</td>"
                    f"<td style='padding:5px'>{self.config[param]}</td></tr>"
                )
        # add weight samples to end, deal with weird dictionary thing
        if self.config.get("weight_samples", {}).get("enabled", False):
            html.append(
                    f"<tr><td style='padding:5px'>{'weight_samples'}</td>"
                    f"<td style='padding:5px'>{'True'}</td></tr>"
                ) 
            for k in ['method', 'xbins', 'ybins', 'lam', 'bandwidth']:
                if k in self.config['weight_samples'].keys():
                    if self.config['weight_samples'][k] is not None:
                        html.append(
                            f"<tr><td style='padding:5px'>weight_samples {'weight_samples '+k}</td>"
                            f"<td style='padding:5px'>{self.config['weight_samples'][k]}</td></tr>"
                        )

        html.append("</table>")

        # Add model status
        html.append("<h4>Status:</h4>")
        html.append("<ul>")

        # Model trained status and training history
        if self.model is not None:
            html.append("<li>Model: Trained ✓</li>")
            if hasattr(self, "traingen"):
                html.append(f"<li>Training samples: {self.traingen.shape[0]}</li>")
                html.append(f"<li>Features: {self.traingen.shape[1]}</li>")

            # Add training history if available
            if hasattr(self, "history") and self.history is not None:
                # Create figure
                fig, ax = plt.subplots(figsize=(8, 4))
                hist = self.history.history

                # Plot training and validation loss
                ax.plot(hist["loss"], label="Training Loss", color="blue")
                axV = ax.twinx()
                axV.plot(hist["val_loss"], label="Validation Loss", color="orange")
                ax.set_xlabel("Epoch")
                ax.set_ylabel("Training Loss")
                axV.set_ylabel("Validation Loss")
                ax.legend()
                axV.legend(loc='upper center')

                # Get final validation loss
                final_val_loss = hist["val_loss"][-1]

                # Convert plot to base64 string
                import io
                import base64

                buf = io.BytesIO()
                fig.savefig(buf, format="png", bbox_inches="tight")
                buf.seek(0)
                plot_data = base64.b64encode(buf.getvalue()).decode("utf-8")
                plt.close(fig)

                # Add plot and metrics to HTML
                html.append("</ul>")  # Close the status list
                html.append("<h4>Training History:</h4>")
                html.append(f"<p>Final validation loss: {final_val_loss:.4f}</p>")
                html.append(
                    f'<img src="data:image/png;base64,{plot_data}" style="max-width:100%">'
                )
                html.append("<ul>")  # Reopen list for remaining items
        else:
            html.append("<li>Model: Not trained</li>")

        # Location normalization status
        if all(
            x is not None
            for x in [self.meanlong, self.sdlong, self.meanlat, self.sdlat]
        ):
            html.append("<li>Location normalization: Computed ✓</li>")
        else:
            html.append("<li>Location normalization: Not computed</li>")

        # Sample data status
        if hasattr(self, "_sample_data_df"):
            html.append(
                f"<li>Sample data loaded: {len(self._sample_data_df)} samples</li>"
            )
        elif "sample_data" in self.config:
            html.append("<li>Sample data: Path provided</li>")
        else:
            html.append("<li>Sample data: Not provided</li>")

        # Genotype data status
        if hasattr(self, "_genotype_df"):
            html.append(
                f"<li>Genotype data loaded: {self._genotype_df.shape[1]} SNPs</li>"
            )
        elif any(x in self.config for x in ["zarr", "vcf", "genotype_data"]):
            html.append("<li>Genotype data: Path provided</li>")
        else:
            html.append("<li>Genotype data: Not provided</li>")

        if hasattr(self, "sample_weights"):
            html.append(
                f"<li>Samples weighted using {self.config['weight_samples'].get('method')}</li>"
            )

        # Add holdout information
        if hasattr(self, "holdout_idx") and self.samples is not None:
            n_holdout = len(self.holdout_idx)
            html.append(f"<li>Holdout samples: {n_holdout} samples held out</li>")
            # Add collapsible list of held out sample IDs
            if n_holdout > 0:
                sample_list = self.samples[self.holdout_idx]
                html.append("<li>Held out samples: <details>")
                html.append("<summary>Click to show/hide</summary>")
                html.append("<ul style='max-height:200px;overflow-y:auto'>")
                for sample in sample_list:
                    html.append(f"<li>{sample}</li>")
                html.append("</ul></details></li>")
        elif hasattr(self, "pred_indices") and self.samples is not None:
            n_holdout = len(self.pred_indices)
            html.append(f"<li>Prediction samples: {n_holdout} samples held out</li>")
            # Add collapsible list of held out sample IDs
            if n_holdout > 0:
                sample_list = self.samples[self.pred_indices]
                html.append("<li>Held out samples: <details>")
                html.append("<summary>Click to show/hide</summary>")
                html.append("<ul style='max-height:200px;overflow-y:auto'>")
                for sample in sample_list:
                    html.append(f"<li>{sample}</li>")
                html.append("</ul></details></li>")

        html.append("</ul>")
        html.append("</div>")

        return "".join(html)

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
        elif "sample_data" in self.config and isinstance(
            self.config["sample_data"], str
        ):
            # Load from file if not already loaded
            sample_df = pd.read_csv(self.config["sample_data"], sep="\t")
            required_cols = ["sampleID", "x", "y"]
            if not all(col in sample_df.columns for col in required_cols):
                raise ValueError(f"sample_data must contain columns: {required_cols}")
            self._sample_data_df = sample_df
            return self._sample_data_df
        else:
            raise ValueError("Sample data is not available in this Locator instance.")


class EnsembleLocator:
    """A class for managing an ensemble of Locator models."""

    def __init__(self, base_config=None, k_folds=5, training_set_indices=None):
        """Initialize EnsembleLocator with configuration parameters.

        Args:
            base_config (dict, optional): Base configuration shared by all models.
                Each model will get a copy of this config.
            k_folds (int, optional): Number of folds for cross-validation.
                Defaults to 5.
            training_set_indices (array-like, optional): Indices of samples to use for
                training and validation. If provided, only these samples will be used
                to create k-folds, while others will automatically be assigned to
                prediction set.
        """
        self.base_config = base_config or {}
        self.k_folds = k_folds
        self.training_set_indices = (
            np.array(training_set_indices) if training_set_indices is not None else None
        )
        self.models = []
        self.fold_indices = {}

        # Initialize attributes that will be set during training
        self.samples = None
        self.meanlong = None
        self.sdlong = None
        self.meanlat = None
        self.sdlat = None

    def create_folds(self, genotypes, samples, locations, training_set_indices=None):
        """Create k-fold splits of the data.

        Args:
            genotypes: GenotypeArray containing genetic data
            samples: Array of sample IDs
            locations: Array of geographic coordinates (x,y)
            training_set_indices: Optional list/array of indices to use for training+validation.
                If provided, only these samples will be used to create the k-folds.
                If None, all samples will be considered for training/validation.

        Returns:
            dict: Dictionary with fold indices
        """
        # First verify dimensions match
        if (
            len(samples) != genotypes.shape[1]
        ):  # Assuming genotypes is (n_snps, n_samples, ploidy)
            raise ValueError(
                f"Number of samples ({len(samples)}) does not match genotype data dimension ({genotypes.shape[1]})"
            )

        # If training_set_indices provided, verify they are valid
        if training_set_indices is not None:
            training_set_indices = np.array(training_set_indices)
            if not np.all(np.isin(training_set_indices, range(len(samples)))):
                raise ValueError("training_set_indices contains invalid indices")

            # Subset the relevant arrays to only include training set samples
            subset_samples = samples[training_set_indices]
            subset_locations = locations[training_set_indices]
        else:
            # Use all samples
            subset_samples = samples
            subset_locations = locations
            training_set_indices = np.arange(len(samples))

        # Get indices of samples with known locations within the subset
        known_idx = np.argwhere(~np.isnan(subset_locations[:, 0]))
        known_idx = np.array([x[0] for x in known_idx])

        # Get indices of samples with unknown locations (from the full dataset)
        # These are samples not in training_set_indices OR samples with unknown locations
        all_indices = set(range(len(samples)))
        training_set = set(training_set_indices)
        pred_idx = np.array(
            list(all_indices - training_set)
        )  # Samples not in training set

        # Also add samples with unknown locations from the training set
        unknown_in_training = training_set_indices[np.isnan(subset_locations[:, 0])]
        pred_idx = np.concatenate([pred_idx, unknown_in_training])
        pred_idx.sort()

        # Randomly shuffle known indices
        np.random.shuffle(known_idx)

        # Create k folds
        fold_size = len(known_idx) // self.k_folds
        self.fold_indices = {}  # Initialize fold_indices dictionary

        for fold in range(self.k_folds):
            start_idx = fold * fold_size
            end_idx = (
                start_idx + fold_size if fold < self.k_folds - 1 else len(known_idx)
            )

            # Get validation indices for this fold
            # Convert back to original sample indices
            val_idx = training_set_indices[known_idx[start_idx:end_idx]]

            # Get training indices (all other known samples from training set)
            train_subset = known_idx[
                np.concatenate(
                    [np.arange(0, start_idx), np.arange(end_idx, len(known_idx))]
                )
            ]
            train_idx = training_set_indices[train_subset]

            self.fold_indices[fold] = {
                "train": train_idx,
                "val": val_idx,
                "pred": pred_idx,
            }

        return self.fold_indices

    def train(self, genotypes, samples, sample_data_file=None):
        """Train k models on different folds of the data."""
        self.samples = samples

        # Get sample data and locations
        locator = Locator(self.base_config)
        if hasattr(locator, "_sample_data_df"):
            sample_data, locs = locator.sort_samples(samples)
        else:
            sample_data_path = sample_data_file or self.base_config.get("sample_data")
            if not sample_data_path:
                raise ValueError("sample_data file path must be provided")
            sample_data, locs = locator.sort_samples(samples, sample_data_path)

        # Create folds if not already done
        if not self.fold_indices:
            self.create_folds(
                genotypes, samples, locs, training_set_indices=self.training_set_indices
            )

        # Filter SNPs once before creating folds
        filtered_genotypes = filter_snps(
            genotypes,
            min_mac=self.base_config.get("min_mac", 2),
            max_snps=self.base_config.get("max_SNPs"),
            impute=self.base_config.get("impute_missing", False),
        )

        # Initialize lists to store normalization parameters
        all_meanlongs = []
        all_sdlongs = []
        all_meanlats = []
        all_sdlats = []

        # Train a model for each fold
        histories = []
        for fold in range(self.k_folds):
            print(f"\nTraining fold {fold + 1}/{self.k_folds}")

            # Create new model for this fold
            fold_config = self.base_config.copy()
            fold_config["out"] = f"{self.base_config['out']}_fold{fold}"
            model = Locator(fold_config)

            # Get indices for this fold
            fold_indices = self.fold_indices[fold]
            train_idx = fold_indices["train"]
            val_idx = fold_indices["val"]
            pred_idx = fold_indices["pred"]

            # Store samples and prediction indices
            model.samples = samples
            model.pred_indices = pred_idx

            # Prepare data for this fold using pre-filtered genotypes
            model.traingen = np.transpose(filtered_genotypes[:, train_idx])
            model.testgen = np.transpose(filtered_genotypes[:, val_idx])
            model.predgen = np.transpose(filtered_genotypes[:, pred_idx])

            # Normalize locations using only training data
            train_locs = locs[train_idx]
            (
                model.meanlong,
                model.sdlong,
                model.meanlat,
                model.sdlat,
                model.unnormedlocs,
                normalized_train_locs,
            ) = normalize_locs(train_locs)

            # Store normalization parameters
            all_meanlongs.append(model.meanlong)
            all_sdlongs.append(model.sdlong)
            all_meanlats.append(model.meanlat)
            all_sdlats.append(model.sdlat)

            # Normalize validation locations
            val_locs = locs[val_idx]
            normalized_val_locs = np.array(
                [
                    [
                        (x[0] - model.meanlong) / model.sdlong,
                        (x[1] - model.meanlat) / model.sdlat,
                    ]
                    for x in val_locs
                ]
            )

            # Store locations
            model.trainlocs = normalized_train_locs
            model.testlocs = normalized_val_locs

            # Create TensorFlow datasets with caching
            train_dataset = tf.data.Dataset.from_tensor_slices(
                (model.traingen, normalized_train_locs)
            )
            train_dataset = train_dataset.cache()
            train_dataset = train_dataset.shuffle(buffer_size=len(train_idx))

            # Apply augmentation if enabled
            if model.config.get("augmentation", {}).get("enabled", False):
                flip_rate = model.config.get("augmentation", {}).get("flip_rate", 0.05)
                train_dataset = train_dataset.map(
                    lambda x, y: flip_genotypes(x, y, mask_rate=flip_rate),
                    num_parallel_calls=tf.data.AUTOTUNE,
                )

            train_dataset = train_dataset.batch(model.config.get("batch_size", 32))
            train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)

            # Create validation dataset
            validation_dataset = tf.data.Dataset.from_tensor_slices(
                (model.testgen, normalized_val_locs)
            )
            validation_dataset = validation_dataset.cache()
            validation_dataset = validation_dataset.batch(
                model.config.get("batch_size", 32)
            )
            validation_dataset = validation_dataset.prefetch(tf.data.AUTOTUNE)

            # Set up model and train
            model.model = create_network(
                input_shape=model.traingen.shape[1],
                width=model.config.get("width", 256),
                n_layers=model.config.get("nlayers", 8),
                dropout_prop=model.config.get("dropout_prop", 0.25),
                optimizer_config={
                    "algo": model.config.get("optimizer_algo", "adam"),
                    "learning_rate": model.config.get("learning_rate", 0.001),
                    "weight_decay": model.config.get("weight_decay", 0.004),
                },
            )

            # Create callbacks
            callbacks = [
                keras.callbacks.ModelCheckpoint(
                    filepath=f"{fold_config['out']}.weights.h5",
                    verbose=model.config.get("keras_verbose", 1),
                    save_best_only=True,
                    save_weights_only=True,
                    monitor="val_loss",
                    save_freq="epoch",
                ),
                keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    min_delta=0,
                    patience=model.config.get("patience", 100),
                ),
                keras.callbacks.ReduceLROnPlateau(
                    monitor="val_loss",
                    factor=0.5,
                    patience=model.config.get("patience", 100) // 6,
                    verbose=model.config.get("keras_verbose", 1),
                    mode="auto",
                    min_delta=0,
                    cooldown=0,
                    min_lr=0,
                ),
            ]

            # Train model
            history = model.model.fit(
                train_dataset,
                epochs=model.config.get("max_epochs", 5000),
                verbose=model.config.get("keras_verbose", 1),
                validation_data=validation_dataset,
                callbacks=callbacks,
            )

            histories.append(history)
            self.models.append(model)

            # Clear session to free memory
            keras.backend.clear_session()

        # Store average normalization parameters
        self.meanlong = np.mean(all_meanlongs)
        self.sdlong = np.mean(all_sdlongs)
        self.meanlat = np.mean(all_meanlats)
        self.sdlat = np.mean(all_sdlats)

        return histories

    def predict(
        self, return_df=True, save_preds_to_disk=True, include_val_predictions=True
    ):
        """Make predictions using the ensemble of models."""
        if not self.models:
            raise ValueError("No trained models in ensemble")

        # Initialize dictionary to store predictions
        all_predictions = {}

        # Get predictions for unknown locations from all models
        pred_predictions = []
        sample_ids = None

        for model in self.models:
            # Ensure samples and pred_indices are set
            if not hasattr(model, "samples"):
                model.samples = self.samples
            if not hasattr(model, "pred_indices"):
                model.pred_indices = self.fold_indices[0][
                    "pred"
                ]  # Use first fold's pred indices

            preds = model.predict(return_df=True, save_preds_to_disk=False)
            pred_predictions.append(preds[["x", "y"]].values)

            # Store sample IDs from first model (they should be the same for all models)
            if sample_ids is None and "sampleID" in preds.columns:
                sample_ids = preds["sampleID"].values

        # Average predictions across models
        mean_predictions = np.mean(pred_predictions, axis=0)
        # rename columns to x_pred and y_pred
        mean_predictions = pd.DataFrame(mean_predictions, columns=["x_pred", "y_pred"])
        # Create DataFrame with predictions
        pred_df = pd.DataFrame(mean_predictions, columns=["x_pred", "y_pred"])
        if sample_ids is not None:
            pred_df.insert(0, "sampleID", sample_ids)

        if not include_val_predictions:
            # Return only prediction set results
            if save_preds_to_disk:
                pred_df.to_csv(
                    f"{self.base_config['out']}_ensemble_predlocs_pred_only.csv",
                    index=False,
                )
            return pred_df if return_df else pred_df.values[:, 1:]

        # Get validation predictions for each fold
        val_predictions = {}
        for fold, model in enumerate(self.models):
            val_idx = self.fold_indices[fold]["val"]

            # Make predictions on validation set
            val_preds = model.model.predict(model.testgen)

            # Denormalize predictions
            val_preds = np.array(
                [
                    [
                        x[0] * model.sdlong + model.meanlong,
                        x[1] * model.sdlat + model.meanlat,
                    ]
                    for x in val_preds
                ]
            )

            # Store predictions with sample IDs
            for idx, pred in zip(val_idx, val_preds):
                sample_id = self.samples[idx]
                val_predictions[sample_id] = pred

        # Create DataFrame with validation predictions
        val_df = pd.DataFrame.from_dict(
            val_predictions, orient="index", columns=["x", "y"]
        )
        val_df.index.name = "sampleID"
        val_df.reset_index(inplace=True)

        # Combine predictions
        all_predictions = pd.concat([pred_df, val_df], ignore_index=True)

        # Save predictions if requested
        if save_preds_to_disk:
            all_predictions.to_csv(
                f"{self.base_config['out']}_ensemble_predlocs.csv", index=False
            )

        if return_df:
            return all_predictions

        return all_predictions.values[:, 1:]  # Return just x,y coordinates

    def _repr_html_(self):
        """Return HTML representation for Jupyter notebooks."""
        html = [
            "<div style='font-family: monospace'>",
            "<h3>EnsembleLocator</h3>",
            "<table>",
            "<tr><th style='text-align:left; padding:5px'>Configuration</th><th style='text-align:left; padding:5px'>Value</th></tr>",
        ]

        # Add number of folds and models
        html.append(
            f"<tr><td style='padding:5px'>Number of folds</td><td style='padding:5px'>{self.k_folds}</td></tr>"
        )
        html.append(
            f"<tr><td style='padding:5px'>Trained models</td><td style='padding:5px'>{len(self.models)}</td></tr>"
        )

        # Add key configuration parameters from base config
        key_params = [
            "train_split",
            "training_set_indices",
            "batch_size",
            "min_mac",
            "max_SNPs",
            "width",
            "nlayers",
            "dropout_prop",
            "max_epochs",
            "weight_samples",
        ]

        for param in key_params:
            if param in self.base_config:
                html.append(
                    f"<tr><td style='padding:5px'>{param}</td>"
                    f"<td style='padding:5px'>{self.base_config[param]}</td></tr>"
                )

        html.append("</table>")

        # Add model status
        html.append("<h4>Status:</h4>")
        html.append("<ul>")

        if self.models:
            html.append(f"<li>Models trained: {len(self.models)} / {self.k_folds}</li>")
            if hasattr(self.models[0], "traingen"):
                html.append(
                    f"<li>Training samples per fold: ~{self.models[0].traingen.shape[0]}</li>"
                )
                html.append(f"<li>Features: {self.models[0].traingen.shape[1]}</li>")

            # Add validation metrics if available
            val_losses = []
            for model in self.models:
                if hasattr(model, "history") and model.history is not None:
                    val_losses.append(model.history.history["val_loss"][-1])

            if val_losses:
                mean_val_loss = np.mean(val_losses)
                std_val_loss = np.std(val_losses)
                html.append(
                    f"<li>Mean validation loss: {mean_val_loss:.4f} ± {std_val_loss:.4f}</li>"
                )
        else:
            html.append("<li>Models: Not trained</li>")

        html.append("</ul>")
        html.append("</div>")

        return "".join(html)
