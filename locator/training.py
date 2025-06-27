"""Training functionality for locator"""

import numpy as np
import pandas as pd
import warnings
from tensorflow import keras
import tensorflow as tf

from .models import create_network, loss_with_range_penalty, rasterize_species_range
from .utils import normalize_locs, filter_snps, weight_samples


class TrainingMixin:
    """Mixin class providing training functionality for Locator."""
    
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
        
        # Handle case when there are no samples to predict (e.g., after exclude mode)
        if len(pred) > 0:
            predgen = np.transpose(genotypes[:, pred])
        else:
            # Create empty array with correct shape
            predgen = np.empty((0, genotypes.shape[0]), dtype=genotypes.dtype)

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

        self.history = self.model.fit(
            self.traingen,
            trainlocs,
            epochs=self.config.get("max_epochs", 5000),
            batch_size=self.config.get("batch_size", 32),
            shuffle=True,
            verbose=self.config.get("keras_verbose", 1),
            validation_data=(self.testgen, testlocs),
            callbacks=callbacks,
            sample_weight = None if self.sample_weights is None else self.sample_weights['sample_weights'],
        )

        # Save training history
        hist_df = pd.DataFrame(self.history.history)
        hist_df.to_csv(f"{self.config['out']}_history.txt", sep="\t", index=False)

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
        )

        # Save training history
        hist_df = pd.DataFrame(self.history.history)
        hist_df.to_csv(f"{self.config['out']}_history.txt", sep="\t", index=False)

        return self.history