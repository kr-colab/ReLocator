"""Training functionality for locator"""

import numpy as np
import pandas as pd
import warnings
from tensorflow import keras
import tensorflow as tf

from .models import create_network, loss_with_range_penalty, rasterize_species_range
from .utils import weight_samples
from .data import normalize_locs, filter_snps_legacy as filter_snps, IndexSet, make_tf_dataset
from .gpu_optimizer import GPUOptimizer
import h5py
import json
from datetime import datetime


class TrainingMixin:
    """Mixin class providing training functionality for Locator."""
    
    def _split_train_test(self, genotypes, locations, train_split=0.9, na_action='separate'):
        """Split genotype and location data into training and test sets.

        Args:
            genotypes: GenotypeArray containing genetic data for all samples
            locations: Array of geographic coordinates (x,y) for each sample,
                      with NaN values for samples with unknown locations
            train_split: Proportion of samples to use for training (default: 0.9)
            na_action: How to handle NA samples ('separate', 'exclude', 'fail')

        Returns:
            tuple: (index_set, train_gen, test_gen, train_locs, test_locs, pred_gen)
                index_set: IndexSet containing train/test/predict indices
                train_gen: Genotype data for training samples
                test_gen: Genotype data for test samples
                train_locs: Location data for training samples
                test_locs: Location data for test samples
                pred_gen: Genotype data for prediction samples (all samples in 'separate' mode)
        """
        # Create NA mask
        na_mask = np.isnan(locations[:, 0])
        n_samples = len(locations)
        
        # Create IndexSet with custom splits for train/test
        splits = {"train": train_split, "test": 1.0 - train_split}
        index_set = IndexSet.random_split(
            n=n_samples,
            splits=splits,
            na_mask=na_mask,
            na_action=na_action
        )
        
        # Get indices
        train_idx = index_set.train
        test_idx = index_set.test
        
        # For 'separate' mode, prediction set should include ALL samples
        if na_action == 'separate':
            pred_idx = np.arange(n_samples)
        else:
            pred_idx = index_set.get_split('predict') if 'predict' in index_set.indices else np.array([], dtype=int)
        
        # Prepare data arrays (still need to return these for backward compatibility)
        traingen = np.transpose(genotypes[:, train_idx])
        testgen = np.transpose(genotypes[:, test_idx])
        trainlocs = locations[train_idx]
        testlocs = locations[test_idx]
        
        # Handle case when there are no samples to predict
        if len(pred_idx) > 0:
            predgen = np.transpose(genotypes[:, pred_idx])
        else:
            # Create empty array with correct shape
            predgen = np.empty((0, genotypes.shape[0]), dtype=genotypes.dtype)

        # Return both IndexSet and data arrays for gradual migration
        return index_set, train_idx, test_idx, traingen, testgen, trainlocs, testlocs, pred_idx, predgen

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
        na_action=None,
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
            na_action (str, optional): How to handle NA samples ('separate', 'exclude', 'fail'). 
                If None, uses self.na_action. Defaults to None.

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

        # Use instance default if na_action not specified
        if na_action is None:
            na_action = self.na_action
            
        # Get sample status
        status = self.get_sample_status(samples)
        
        # Report status
        print(f"Training data: {status['n_known']} samples with coordinates, {status['n_na']} without")
        if status['n_na'] > 0:
            print(f"NA handling mode: {na_action}")
        
        # Apply NA action
        if na_action == 'fail' and status['n_na'] > 0:
            raise ValueError(
                f"Found {status['n_na']} samples without coordinates. "
                f"Set na_action='separate' or 'exclude' to proceed."
            )
        
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
            
        # Apply 'exclude' mode if needed
        if na_action == 'exclude' and status['n_na'] > 0:
            print(f"Excluding {status['n_na']} samples without coordinates")
            # Filter to only known samples
            mask = status['known_indices']
            genotypes = genotypes[:, mask]
            samples = samples[mask]
            locs = locs[mask]
            # Update sample data to match
            sample_data = sample_data.iloc[mask]

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
                self.index_set,
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
                na_action=na_action,
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
                                                    trainlocs=self.unnormedlocs[train],
                                                    trainsamps=self.samples[train],
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

        # Determine batch size
        batch_size = self.config.get("batch_size", 32)
        if self.config.get("gpu_batch_size") == "auto" and not self.config.get("disable_gpu", False):
            # Try to determine optimal batch size
            try:
                optimal_batch = GPUOptimizer.get_optimal_batch_size(
                    self.model, 
                    input_shape=(self.traingen.shape[1],),
                    target_memory_usage=0.85,
                    dataset_size=self.traingen.shape[0]
                )
                print(f"Using optimized batch size: {optimal_batch}")
                batch_size = optimal_batch
            except Exception as e:
                print(f"Failed to optimize batch size: {e}. Using default: {batch_size}")
        elif isinstance(self.config.get("gpu_batch_size"), int):
            batch_size = self.config["gpu_batch_size"]

        # Use efficient data pipeline if enabled
        if self.config.get("use_efficient_pipeline", True) and not self.config.get("disable_gpu", False):
            # Prepare sample weights if available
            sample_weights_array = None
            if self.sample_weights is not None:
                sample_weights_array = self.sample_weights['sample_weights']
            
            # Create datasets using the new unified function
            # Note: we need to work with the full genotype array and use IndexSet
            # First, we need to combine the data back since it was split
            if hasattr(self, 'index_set') and self.index_set is not None:
                # We have an IndexSet - use it directly with the original filtered genotypes
                # This requires access to the full genotype array
                # For now, reconstruct from the split data
                all_genotypes = np.hstack([
                    self.traingen.T,  # Transpose back to (n_snps, n_samples)
                    self.testgen.T,
                    self.predgen.T if self.predgen.shape[0] > 0 else np.empty((self.traingen.shape[1], 0))
                ])
                all_coords = np.vstack([
                    trainlocs,
                    testlocs,
                    np.full((self.predgen.shape[0], 2), np.nan) if self.predgen.shape[0] > 0 else np.empty((0, 2))
                ])
                
                # Create training dataset
                train_dataset = make_tf_dataset(
                    genotypes=all_genotypes,
                    coordinates=all_coords,
                    index_set=self.index_set,
                    split="train",
                    batch_size=batch_size,
                    sample_weights=sample_weights_array,
                    training=True,
                    cache=True
                )
                
                # Create validation dataset
                val_dataset = make_tf_dataset(
                    genotypes=all_genotypes,
                    coordinates=all_coords,
                    index_set=self.index_set,
                    split="test",
                    batch_size=batch_size,
                    training=False,
                    cache=True
                )
            else:
                # Fallback: use the old GPUOptimizer for backward compatibility
                train_dataset = GPUOptimizer.create_efficient_dataset(
                    self.traingen, 
                    trainlocs,
                    batch_size=batch_size,
                    training=True,
                    cache=True
                )
                
                val_dataset = GPUOptimizer.create_efficient_dataset(
                    self.testgen,
                    testlocs,
                    batch_size=batch_size,
                    training=False,
                    cache=True
                )
                
                # Apply sample weights if available
                if sample_weights_array is not None:
                    weights_dataset = tf.data.Dataset.from_tensor_slices(sample_weights_array)
                    train_dataset = tf.data.Dataset.zip((train_dataset, weights_dataset))
                    train_dataset = train_dataset.map(
                        lambda data_tuple, weight: (data_tuple[0], data_tuple[1], weight),
                        num_parallel_calls=tf.data.AUTOTUNE
                    )
            
            self.history = self.model.fit(
                train_dataset,
                epochs=self.config.get("max_epochs", 5000),
                verbose=self.config.get("keras_verbose", 1),
                validation_data=val_dataset,
                callbacks=callbacks,
            )
        else:
            # Use standard fit (legacy mode)
            self.history = self.model.fit(
                self.traingen,
                trainlocs,
                epochs=self.config.get("max_epochs", 5000),
                batch_size=batch_size,
                shuffle=True,
                verbose=self.config.get("keras_verbose", 1),
                validation_data=(self.testgen, testlocs),
                callbacks=callbacks,
                sample_weight = None if self.sample_weights is None else self.sample_weights['sample_weights'],
            )

        # Save training history
        hist_df = pd.DataFrame(self.history.history)
        hist_df.to_csv(f"{self.config['out']}_history.txt", sep="\t", index=False)

        # Save model metadata including normalization parameters
        self._save_model_metadata(boot=boot)

        return self.history

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

        # Determine batch size
        batch_size = self.config.get("batch_size", 32)
        if self.config.get("gpu_batch_size") == "auto" and not self.config.get("disable_gpu", False):
            # Try to determine optimal batch size
            try:
                optimal_batch = GPUOptimizer.get_optimal_batch_size(
                    self.model, 
                    input_shape=(self.traingen.shape[1],),
                    target_memory_usage=0.85,
                    dataset_size=self.traingen.shape[0]
                )
                print(f"Using optimized batch size: {optimal_batch}")
                batch_size = optimal_batch
            except Exception as e:
                print(f"Failed to optimize batch size: {e}. Using default: {batch_size}")
        elif isinstance(self.config.get("gpu_batch_size"), int):
            batch_size = self.config["gpu_batch_size"]

        # Use GPU optimizer for efficient datasets
        if self.config.get("use_efficient_pipeline", True) and not self.config.get("disable_gpu", False):
            # Prepare sample weights if available
            sample_weights = None if self.sample_weights is None else self.sample_weights['sample_weights']
            
            # Create training dataset with GPU optimization
            train_dataset = GPUOptimizer.create_efficient_dataset(
                self.traingen,
                self.trainlocs,
                batch_size=batch_size,
                training=True,
                cache=True
            )
            
            if sample_weights is not None:
                # Add weights to the dataset
                weights_dataset = tf.data.Dataset.from_tensor_slices(sample_weights)
                weights_dataset = weights_dataset.batch(batch_size, drop_remainder=True)
                train_dataset = tf.data.Dataset.zip((train_dataset, weights_dataset))
                train_dataset = train_dataset.map(
                    lambda data_tuple, weights: (data_tuple[0], data_tuple[1], weights),
                    num_parallel_calls=tf.data.AUTOTUNE
                )
            
            # Apply augmentation if enabled
            if self.config.get("augmentation", {}).get("enabled", False):
                flip_rate = self.config.get("augmentation", {}).get("flip_rate", 0.05)
                
                def flip_genotypes(genotypes, locations, mask_rate=0.05):
                    """Randomly flip genotype values with probability mask_rate"""
                    mask = tf.random.uniform(tf.shape(genotypes)) < mask_rate
                    return tf.where(mask, 1 - genotypes, genotypes), locations
                
                if sample_weights is not None:
                    train_dataset = train_dataset.map(
                        lambda x, y, w: (*flip_genotypes(x, y, mask_rate=flip_rate), w),
                        num_parallel_calls=tf.data.AUTOTUNE
                    )
                else:
                    train_dataset = train_dataset.map(
                        lambda x, y: flip_genotypes(x, y, mask_rate=flip_rate),
                        num_parallel_calls=tf.data.AUTOTUNE
                    )
            
            # Create validation dataset
            validation_dataset = GPUOptimizer.create_efficient_dataset(
                self.testgen,
                self.testlocs,
                batch_size=batch_size,
                training=False,
                cache=True
            )
        else:
            # Fallback to original implementation
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
                    lambda x, y, w: (*flip_genotypes(x, y, mask_rate=flip_rate), w) if w is not None else flip_genotypes(x, y, mask_rate=flip_rate),
                    num_parallel_calls=tf.data.AUTOTUNE,
                )

            train_dataset = train_dataset.batch(batch_size)
            train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)

            validation_dataset = tf.data.Dataset.from_tensor_slices(
                (self.testgen, self.testlocs)
            )
            validation_dataset = validation_dataset.batch(batch_size)
            validation_dataset = validation_dataset.prefetch(tf.data.AUTOTUNE)

        self.history = self.model.fit(
            train_dataset,
            epochs=self.config.get("max_epochs", 5000),
            verbose=self.config.get("keras_verbose", 0),
            validation_data=validation_dataset,
            callbacks=callbacks,
        )

        # Save training history
        hist_df = pd.DataFrame(self.history.history)
        hist_df.to_csv(f"{self.config['out']}_history.txt", sep="\t", index=False)

        # Save model metadata including normalization parameters
        self._save_model_metadata()

        return self.history
    
    def _save_model_metadata(self, boot=0):
        """Save model metadata including normalization parameters to HDF5 file.
        
        This method saves essential preprocessing parameters as HDF5 attributes
        so the model can be properly used for predictions in a new session.
        
        Args:
            boot: Bootstrap iteration number (default: 0)
        """
        # Determine the weights file path
        if self.config.get("bootstrap", False):
            filepath = f"{self.config['out']}_boot{boot}.weights.h5"
        else:
            filepath = f"{self.config['out']}.weights.h5"
        
        # Wait a moment to ensure the weights file is written
        import time
        time.sleep(0.5)
        
        # Open the HDF5 file and add metadata as attributes
        try:
            with h5py.File(filepath, 'a') as f:
                # Save normalization parameters
                f.attrs['coord_meanlong'] = self.meanlong if self.meanlong is not None else 0.0
                f.attrs['coord_sdlong'] = self.sdlong if self.sdlong is not None else 1.0
                f.attrs['coord_meanlat'] = self.meanlat if self.meanlat is not None else 0.0
                f.attrs['coord_sdlat'] = self.sdlat if self.sdlat is not None else 1.0
                
                # Save preprocessing parameters
                f.attrs['min_mac'] = self.config.get('min_mac', 2)
                f.attrs['max_SNPs'] = self.config.get('max_SNPs', None) if self.config.get('max_SNPs') is not None else -1
                f.attrs['impute_missing'] = self.config.get('impute_missing', False)
                f.attrs['n_samples'] = len(self.samples) if self.samples is not None else 0
                f.attrs['n_snps'] = self.traingen.shape[1] if hasattr(self, 'traingen') and self.traingen is not None else 0
                
                # Save metadata version for future compatibility
                f.attrs['metadata_version'] = '1.0'
                f.attrs['locator_version'] = '0.1.0'  # Should get from package version
                f.attrs['save_date'] = datetime.now().isoformat()
                
                # Save config as JSON string for full reproducibility
                config_to_save = self.config.copy()
                # Remove non-serializable items
                non_serializable_keys = ['genotypes', 'sample_data', 'genotype_data', 'species_range_geom']
                for key in non_serializable_keys:
                    config_to_save.pop(key, None)
                    
                # Also remove any DataFrame values in nested dicts
                if 'weight_samples' in config_to_save and isinstance(config_to_save['weight_samples'], dict):
                    config_to_save['weight_samples'] = config_to_save['weight_samples'].copy()
                    config_to_save['weight_samples'].pop('weightdf', None)
                    
                f.attrs['config_json'] = json.dumps(config_to_save)
                
                print(f"Model metadata saved to {filepath}")
                
        except Exception as e:
            warnings.warn(f"Failed to save model metadata: {e}")
            # Don't fail training if metadata save fails