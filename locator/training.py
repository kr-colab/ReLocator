"""Training functionality for locator"""

import json
import warnings
from datetime import datetime

import h5py
import numpy as np
import pandas as pd
from tensorflow import keras

from .data import IndexSet, make_tf_dataset, normalize_locs
from .data import filter_snps_legacy as filter_snps
from .gpu_optimizer import GPUOptimizer
from .models import create_network, loss_with_range_penalty, rasterize_species_range
from .utils import weight_samples


class TrainingMixin:
    """Mixin class providing training functionality for Locator."""

    def _split_train_test(
        self, genotypes, locations, train_split=0.9, na_action="separate"
    ):
        """Split genotype and location data into training and test sets.

        This method creates an IndexSet for efficient data splitting without creating
        full genotype arrays. The actual data loading is handled by tf.data pipeline.

        Args:
            genotypes: GenotypeArray containing genetic data for all samples
            locations: Array of geographic coordinates (x,y) for each sample,
                      with NaN values for samples with unknown locations
            train_split: Proportion of samples to use for training (default: 0.9)
            na_action: How to handle NA samples ('separate', 'exclude', 'fail')

        Returns
        -------
            tuple: (index_set, train_idx, test_idx, train_locs, test_locs, pred_idx)
                index_set: IndexSet containing train/test/predict indices
                train_idx: Training sample indices
                test_idx: Test sample indices
                train_locs: Location data for training samples
                test_locs: Location data for test samples
                pred_idx: Prediction sample indices
        """
        # Create NA mask
        na_mask = np.isnan(locations[:, 0])
        n_samples = len(locations)

        # Create IndexSet with custom splits for train/test
        splits = {"train": train_split, "test": 1.0 - train_split}
        index_set = IndexSet.random_split(
            n=n_samples, splits=splits, na_mask=na_mask, na_action=na_action
        )

        # Get indices
        train_idx = index_set.train
        test_idx = index_set.test

        # For 'separate' mode, prediction set should include ALL samples
        if na_action == "separate":
            pred_idx = np.arange(n_samples)
        else:
            pred_idx = (
                index_set.get_split("predict")
                if "predict" in index_set.indices
                else np.array([], dtype=int)
            )

        # Prepare location arrays (always needed)
        trainlocs = locations[train_idx]
        testlocs = locations[test_idx]

        # Return IndexSet and indices only - no arrays created
        return index_set, train_idx, test_idx, trainlocs, testlocs, pred_idx

    def _create_callbacks(self, boot=0):
        """Create Keras callbacks for training.

        Args:
            boot: Bootstrap replicate number (default: 0)

        Returns
        -------
            list: List of Keras callbacks
        """
        callbacks = []

        # Check if we should save fold models
        should_save = self.config.get("save_fold_models", True)

        if should_save:
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
                mode="min",  # Explicitly set mode for clarity
            )
            callbacks.append(checkpointer)

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

        callbacks.extend([earlystop, reducelr])
        return callbacks

    def set_sample_weights(self, wdict):
        """Set sample weights for training.
        Args:
            wdict (dict): Dictionary returned by utils.weight_samples() containing sample weights.
        """
        self.sample_weights = wdict
        self.config["weight_samples"]["enabled"] = True
        for key, value in wdict.items():
            self.config["weight_samples"][key] = value

    def train(  # noqa: C901
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
        na_action=None,
        site_order=None,
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
            site_order (np.ndarray, optional): Array of SNP indices for bootstrap resampling.
                If provided, SNPs will be reordered according to these indices during training.
                Used for bootstrap analyses to resample SNPs with replacement.

        Returns
        -------
            keras.callbacks.History or None: The Keras training history object if training is performed, or None if `setup_only` is True.

        Raises
        ------
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
        # Store samples and site_order
        self.samples = samples
        self.site_order = site_order

        # Use instance default if na_action not specified
        if na_action is None:
            na_action = self.na_action

        # Get sample status
        status = self.get_sample_status(samples)

        # Report status
        print(
            f"Training data: {status['n_known']} samples with coordinates, {status['n_na']} without"
        )
        if status["n_na"] > 0:
            print(f"NA handling mode: {na_action}")

        # Apply NA action
        if na_action == "fail" and status["n_na"] > 0:
            raise ValueError(
                f"Found {status['n_na']} samples without coordinates. "
                f"Set na_action='separate' or 'exclude' to proceed."
            )

        sample_data, locs = self._resolve_locations(samples, sample_data_file)

        # Apply 'exclude' mode if needed
        if na_action == "exclude" and status["n_na"] > 0:
            print(f"Excluding {status['n_na']} samples without coordinates")
            # Filter to only known samples
            mask = status["known_indices"]
            genotypes = genotypes[:, mask]
            samples = samples[mask]
            locs = locs[mask]
            # Update sample data to match
            sample_data = sample_data.iloc[mask]

        # Filter SNPs if not using pre-processed data
        if train_gen is None:
            self._filter_genotypes(genotypes)

            # Split data using IndexSet approach (no arrays created)
            (
                self.index_set,
                train,
                test,
                trainlocs,
                testlocs,
                pred,
            ) = self._split_train_test(
                self.filtered_genotypes,
                locs,  # Use unnormalized locations for split
                train_split=self.config.get("train_split", 0.9),
                na_action=na_action,
            )

            # Set array attributes to None for compatibility
            self.traingen = None
            self.testgen = None

            # For 'separate' mode, create predgen for backward compatibility
            if na_action == "separate" and len(pred) > 0:
                self.predgen = np.transpose(self.filtered_genotypes[:, pred])
            elif len(pred) == 0:
                # Create empty array with correct shape
                self.predgen = np.zeros(
                    (0, self.filtered_genotypes.shape[0]),
                    dtype=self.filtered_genotypes.dtype,
                )
            else:
                self.predgen = None

            # Normalize locations and store for each split using helper method
            normalized_locs = self._normalize_and_store_locations(
                locs, samples, train, test
            )

            # Calculate sample weights using helper method
            # Pass unnormalized training locations
            train_locs_unnormed = locs[train]
            self._calculate_sample_weights(train, train_locs=train_locs_unnormed)

            # Store prediction indices
            self.pred_indices = pred

            splits = {"Training": train, "Validation": test}
            if len(pred) > 0:
                splits["Prediction (no coords)"] = pred
            self._report_split_summary(
                splits, len(samples), self.filtered_genotypes.shape[0]
            )
        else:
            # Use pre-processed data (for bootstrapping)
            self.traingen = train_gen
            self.testgen = test_gen
            self.predgen = pred_gen

            # For pre-processed data, we still need to normalize locations to get the normalization parameters
            (
                self.meanlong,
                self.sdlong,
                self.meanlat,
                self.sdlat,
                self.unnormedlocs,
                normalized_locs,
            ) = normalize_locs(locs)

            # Use provided locations if available
            if train_locs is not None and test_locs is not None:
                self.trainlocs = train_locs
                self.testlocs = test_locs
            else:
                # Get train/test indices and locations from original split
                train = np.where(~np.isnan(normalized_locs[:, 0]))[0]
                test = np.random.choice(
                    train,
                    round((1 - self.config.get("train_split", 0.9)) * len(train)),
                    replace=False,
                )
                train = np.setdiff1d(train, test)
                self.trainlocs = normalized_locs[train]
                self.testlocs = normalized_locs[test]

        # Create model if not already created
        if self.model is None:
            # Determine input shape
            if self.traingen is not None:
                input_shape = self.traingen.shape[1]
            else:
                # When using efficient pipeline, get shape from filtered_genotypes
                # If site_order is provided, use its length (for jacknife/bootstrap)
                if site_order is not None:
                    input_shape = len(site_order)
                else:
                    input_shape = self.filtered_genotypes.shape[0]

            self.model = self._create_model(input_shape=input_shape)

        # Return early if setup_only
        if setup_only:
            return None

        callbacks = self._create_callbacks(boot=boot)
        self._build_datasets_and_fit(normalized_locs, callbacks, site_order=site_order)
        self._save_training_artifacts(boot=boot)
        return self.history

    def train_holdout(  # noqa: C901
        self,
        genotypes=None,
        samples=None,
        k=10,
        holdout_indices=None,
        filtered_genotypes=None,
    ):
        """Train the model while holding out samples with known locations.

        Args:
            genotypes: Array of genotype data. Required unless
                filtered_genotypes is provided.
            samples: Sample IDs corresponding to genotypes
            k: Number of samples to hold out (ignored if holdout_indices provided)
            holdout_indices: Optional specific indices of samples to hold out
            filtered_genotypes: Pre-filtered allele count array. If provided,
                skips internal filter_snps call and avoids loading the full
                genotype array. Used by parallel dispatch to share one
                filtered copy across all workers.

        Returns
        -------
            keras.callbacks.History object from model training
        """
        # Store samples
        self.samples = samples

        _, locs = self._resolve_locations(samples)

        # Get indices of samples with known locations
        known_idx = np.where(~np.isnan(locs[:, 0]))[0]

        # Determine holdout indices
        if holdout_indices is not None:
            holdout_idx = np.array(holdout_indices)
            if not all(idx in known_idx for idx in holdout_idx):
                raise ValueError(
                    "All holdout_indices must be indices of samples with known locations"
                )
        else:
            if k >= len(known_idx):
                raise ValueError(
                    f"k ({k}) must be less than number of samples with known locations ({len(known_idx)})"
                )
            holdout_idx = np.random.choice(known_idx, k, replace=False)

        self._filter_genotypes(genotypes, filtered_genotypes)

        # Get available samples for training (exclude holdout and NA samples)
        available_indices = np.setdiff1d(known_idx, holdout_idx)
        n_available = len(available_indices)

        if n_available == 0:
            raise ValueError("No samples available for training after holdout")

        # Split available samples into train/test
        train_split = self.config.get("train_split", 0.9)
        n_train = int(n_available * train_split)

        np.random.shuffle(available_indices)
        train_indices = available_indices[:n_train]
        test_indices = available_indices[n_train:]

        # Create IndexSet for efficient data handling
        n_samples = len(locs)
        self.index_set = IndexSet(
            indices={
                "train": train_indices,
                "test": test_indices,
                "holdout": holdout_idx,
            },
            total_samples=n_samples,
            na_mask=np.isnan(locs[:, 0]),
        )

        # Normalize locations and store for each split
        normalized_locs = self._normalize_and_store_locations(
            locs, samples, train_indices, test_indices
        )

        self._store_holdout_state(holdout_idx, normalized_locs)

        self._report_split_summary(
            {
                "Training": train_indices,
                "Validation": test_indices,
                "Holdout": holdout_idx,
            },
            len(samples),
            self.filtered_genotypes.shape[0],
        )

        # Handle sample weights if enabled
        self._calculate_sample_weights(train_indices)

        # Create model
        self.model = self._create_model(input_shape=self.filtered_genotypes.shape[0])

        # Create callbacks
        # For train_holdout, we might want to skip saving intermediate models
        # to reduce file I/O overhead during k-fold cross-validation
        if self.config.get("holdout_no_intermediate_saves", False):
            # Minimal callbacks without file saves
            callbacks = [
                keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    min_delta=0,
                    patience=self.config.get("patience", 100),
                    restore_best_weights=True,
                ),
                keras.callbacks.ReduceLROnPlateau(
                    monitor="val_loss",
                    factor=0.5,
                    patience=self.config.get("patience", 100) // 6,
                    verbose=self.config.get("keras_verbose", 1),
                    mode="auto",
                    min_delta=0,
                    min_lr=1e-5,
                ),
            ]
        else:
            callbacks = self._create_callbacks()

        self._build_datasets_and_fit(normalized_locs, callbacks, keras_verbose=0)
        self._save_training_artifacts(save=self.config.get("save_fold_models", True))
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

        # Open the HDF5 file and add metadata as attributes
        try:
            with h5py.File(filepath, "a") as f:
                # Save normalization parameters
                f.attrs["coord_meanlong"] = (
                    self.meanlong if self.meanlong is not None else 0.0
                )
                f.attrs["coord_sdlong"] = self.sdlong if self.sdlong is not None else 1.0
                f.attrs["coord_meanlat"] = (
                    self.meanlat if self.meanlat is not None else 0.0
                )
                f.attrs["coord_sdlat"] = self.sdlat if self.sdlat is not None else 1.0

                # Save preprocessing parameters
                f.attrs["min_mac"] = self.config.get("min_mac", 2)
                f.attrs["max_SNPs"] = (
                    self.config.get("max_SNPs", None)
                    if self.config.get("max_SNPs") is not None
                    else -1
                )
                f.attrs["impute_missing"] = self.config.get("impute_missing", False)
                f.attrs["n_samples"] = (
                    len(self.samples) if self.samples is not None else 0
                )
                f.attrs["n_snps"] = (
                    self.filtered_genotypes.shape[0]
                    if hasattr(self, "filtered_genotypes")
                    and self.filtered_genotypes is not None
                    else 0
                )

                # Save metadata version for future compatibility
                f.attrs["metadata_version"] = "1.0"
                f.attrs["locator_version"] = "0.1.0"  # Should get from package version
                f.attrs["save_date"] = datetime.now().isoformat()

                # Save config as JSON string for full reproducibility
                config_to_save = self.config.copy()
                # Remove non-serializable items
                non_serializable_keys = [
                    "genotypes",
                    "sample_data",
                    "genotype_data",
                    "species_range_geom",
                ]
                for key in non_serializable_keys:
                    config_to_save.pop(key, None)

                # Also remove any DataFrame values in nested dicts
                if "weight_samples" in config_to_save and isinstance(
                    config_to_save["weight_samples"], dict
                ):
                    config_to_save["weight_samples"] = config_to_save[
                        "weight_samples"
                    ].copy()
                    config_to_save["weight_samples"].pop("weightdf", None)

                f.attrs["config_json"] = json.dumps(config_to_save)

                print(f"Model metadata saved to {filepath}")

        except Exception as e:
            warnings.warn(f"Failed to save model metadata: {e}")
            # Don't fail training if metadata save fails

    def _create_model(self, input_shape):
        """Create neural network model. Extracted to avoid duplication."""
        loss_fn = None
        if self.config.get("use_range_penalty"):
            if self.config.get("species_range_shapefile") is None:
                raise ValueError(
                    "species_range_shapefile must be provided "
                    "if use_range_penalty is True"
                )
            if self.config.get("resolution") is None:
                raise ValueError(
                    "resolution must be provided if use_range_penalty is True"
                )

            mask_tensor, mask_transform = rasterize_species_range(
                self.config["species_range_shapefile"],
                resolution=self.config.get("resolution", 0.05),
            )

            def loss_fn(y_true, y_pred):  # noqa: F811
                return loss_with_range_penalty(
                    y_true,
                    y_pred,
                    mask_tensor=mask_tensor,
                    transform=mask_transform,
                    resolution=self.config.get("resolution", 0.05),
                    penalty_weight=self.config.get("penalty_weight", 1.0),
                )

        return create_network(
            input_shape=input_shape,
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

    def train_window(
        self,
        genotypes,
        samples,
        window_snp_indices,
        index_set,
        normalized_locs,
    ):
        """Train the model for a specific genomic window using efficient tf.data pipeline.

        This is an internal method used by run_windows_holdouts to train models
        on specific genomic windows without creating intermediate arrays.

        Args:
            genotypes: Full genotype array (not filtered)
            samples: Sample IDs
            window_snp_indices: Indices of SNPs in this window
            index_set: Pre-computed IndexSet with train/test/holdout splits
            normalized_locs: Pre-normalized location coordinates

        Returns
        -------
            keras.callbacks.History object from model training
        """
        # Store samples and index set
        self.samples = samples
        self.index_set = index_set

        window_genotypes = genotypes[window_snp_indices, :, :]
        self._filter_genotypes(window_genotypes)

        # Store filtered data shape
        n_snps_filtered = self.filtered_genotypes.shape[0]

        # Calculate sample weights if enabled
        self._calculate_sample_weights(index_set.train)

        # Create model for this window
        self.model = self._create_model(input_shape=n_snps_filtered)

        # Create callbacks
        callbacks = self._create_callbacks()

        # Store necessary data for prediction
        # In window analysis, 'test' split contains the holdout samples
        self._store_holdout_state(index_set.get_split("test"), normalized_locs)

        # For window analysis, we need to split the train indices into train/val
        train_indices = index_set.get_split("train")
        train_split = self.config.get("train_split", 0.9)
        n_train = int(len(train_indices) * train_split)

        # Shuffle and split
        np.random.shuffle(train_indices)
        actual_train = train_indices[:n_train]
        actual_val = train_indices[n_train:]

        self.trainlocs = normalized_locs[actual_train]
        self.testlocs = normalized_locs[actual_val]

        # Create a new IndexSet with the proper splits for training
        self.index_set = IndexSet(
            indices={"train": actual_train, "test": actual_val},
            total_samples=index_set.total_samples,
            na_mask=index_set.na_mask,
        )

        self._build_datasets_and_fit(normalized_locs, callbacks, keras_verbose=0)
        return self.history

    def _calculate_sample_weights(self, train_indices, train_locs=None):
        """Calculate sample weights if enabled. Extracted to avoid duplication.

        Args:
            train_indices: Indices of training samples
            train_locs: Optional unnormalized training locations. If None, uses self.unnormedlocs
        """
        if self.config.get("weight_samples", {}).get("enabled", False):
            if self.sample_weights is not None:
                warnings.warn(
                    """Sample weights already calculated.
                    Set locator.sample_weights to None in config to disable."""
                )
            else:
                wmethod = self.config.get("weight_samples", {}).get("method")
                # Use provided train_locs or fall back to self.unnormedlocs
                locs_for_weights = (
                    train_locs if train_locs is not None else self.unnormedlocs
                )
                self.sample_weights = weight_samples(
                    wmethod,
                    trainlocs=locs_for_weights,
                    trainsamps=self.samples[train_indices],
                    weightdf=self.config.get("weight_samples", {}).get("weightdf"),
                    xbins=self.config.get("weight_samples", {}).get("xbins"),
                    ybins=self.config.get("weight_samples", {}).get("ybins"),
                    lam=self.config.get("weight_samples", {}).get("lam"),
                    bandwidth=self.config.get("weight_samples", {}).get("bandwidth"),
                )

    def _determine_batch_size(self, dataset_size):
        """Determine optimal batch size. Extracted to avoid duplication."""
        batch_size = self.config.get("batch_size", 32)
        verbose_batch_size = self.config.get("verbose_batch_size", False)

        if self.config.get("gpu_batch_size") == "auto" and not self.config.get(
            "disable_gpu", False
        ):
            try:
                optimal_batch = GPUOptimizer.get_optimal_batch_size(
                    self.model,
                    input_shape=(self.filtered_genotypes.shape[0],),
                    target_memory_usage=0.85,
                    dataset_size=dataset_size,
                    verbose=verbose_batch_size,
                )
                if verbose_batch_size:
                    print(f"Using optimized batch size: {optimal_batch}")
                batch_size = optimal_batch
            except Exception as e:
                if verbose_batch_size:
                    print(
                        f"Failed to optimize batch size: {e}. Using default: {batch_size}"
                    )
        elif isinstance(self.config.get("gpu_batch_size"), int):
            batch_size = self.config["gpu_batch_size"]

        return batch_size

    # ------------------------------------------------------------------
    # Shared helpers used by train(), train_holdout(), and train_window()
    # ------------------------------------------------------------------

    def _resolve_locations(self, samples, sample_data_file=None):
        """Load sample metadata and return locations array.

        Args:
            samples: Array of sample IDs
            sample_data_file: Optional path override for sample data file

        Returns
        -------
            tuple: (sample_data DataFrame, locs array of shape (n_samples, 2))
        """
        if hasattr(self, "_sample_data_df"):
            return self.sort_samples(samples)

        sample_data_path = sample_data_file or self.config.get("sample_data")
        if not isinstance(sample_data_path, str):
            raise ValueError(
                "sample_data file path must be provided in config or as argument "
                "when not using DataFrame input"
            )
        return self.sort_samples(samples, sample_data_path)

    def _filter_genotypes(self, genotypes, filtered_genotypes=None):
        """Filter SNPs and store result as self.filtered_genotypes.

        Args:
            genotypes: Raw GenotypeArray or window slice
            filtered_genotypes: Pre-filtered allele counts (skips filtering)

        Returns
        -------
            np.ndarray: Filtered allele count array
        """
        if filtered_genotypes is not None:
            self.filtered_genotypes = filtered_genotypes
        elif genotypes is not None:
            self.filtered_genotypes = filter_snps(
                genotypes,
                min_mac=self.config.get("min_mac", 2),
                max_snps=self.config.get("max_SNPs"),
                impute=self.config.get("impute_missing", False),
            )
        else:
            raise ValueError("Either genotypes or filtered_genotypes must be provided")
        return self.filtered_genotypes

    def _store_holdout_state(self, holdout_idx, normalized_locs):
        """Store holdout data for use by predict_holdout().

        Sets self.holdout_idx, self.holdout_gen, self.holdout_locs.

        Args:
            holdout_idx: Array of held-out sample indices
            normalized_locs: Full normalized location array
        """
        self.holdout_idx = holdout_idx
        self.holdout_gen = np.asarray(
            self.filtered_genotypes[:, holdout_idx].T, order="C"
        )
        self.holdout_locs = normalized_locs[holdout_idx]

    def _build_datasets_and_fit(
        self,
        normalized_locs,
        callbacks,
        site_order=None,
        keras_verbose=None,
    ):
        """Build tf.data pipelines and train the model.

        Requires self.filtered_genotypes, self.index_set, self.model,
        and self.sample_weights to be set before calling.

        Args:
            normalized_locs: Full normalized location array
            callbacks: List of Keras callbacks
            site_order: Optional SNP reordering for bootstrap
            keras_verbose: Verbosity for model.fit (default: from config)

        Returns
        -------
            keras.callbacks.History
        """
        batch_size = self._determine_batch_size(len(self.index_set.train))

        if keras_verbose is None:
            keras_verbose = self.config.get("keras_verbose", 1)

        train_dataset = make_tf_dataset(
            genotypes=self.filtered_genotypes,
            coordinates=normalized_locs,
            index_set=self.index_set,
            split="train",
            batch_size=batch_size,
            sample_weights=(
                self.sample_weights["sample_weights"] if self.sample_weights else None
            ),
            training=True,
            cache=True,
            site_order=site_order,
        )

        val_dataset = make_tf_dataset(
            genotypes=self.filtered_genotypes,
            coordinates=normalized_locs,
            index_set=self.index_set,
            split="test",
            batch_size=batch_size,
            training=False,
            cache=True,
            site_order=site_order,
        )

        history = self.model.fit(
            train_dataset,
            epochs=self.config.get("max_epochs", 5000),
            validation_data=val_dataset,
            callbacks=callbacks,
            verbose=keras_verbose,
        )
        self.history = history
        return history

    def _save_training_artifacts(self, boot=0, save=True):
        """Save training history and model metadata to disk.

        Args:
            boot: Bootstrap replicate number
            save: Whether to actually save (False skips)
        """
        if not save or not hasattr(self, "history"):
            return

        hist_df = pd.DataFrame(self.history.history)
        hist_df.to_csv(
            f"{self.config['out']}_boot{boot}_history.txt"
            if self.config.get("bootstrap", False)
            else f"{self.config['out']}_history.txt",
            index=False,
        )
        self._save_model_metadata(boot=boot)

    def _report_split_summary(self, split_dict, n_total, n_snps):
        """Print data split summary if verbose_splits is enabled.

        Args:
            split_dict: Dict mapping label to index array,
                e.g. {"Training": train_idx, "Validation": val_idx}
            n_total: Total number of samples
            n_snps: Number of SNPs after filtering
        """
        if not self.config.get("verbose_splits", False):
            return

        print("\nSplit summary:")
        for label, indices in split_dict.items():
            n = len(indices)
            pct = n / n_total * 100
            print(f"  {label} samples: {n} ({pct:.1f}%)")
        print(f"  Total samples: {n_total}")
        print(f"  Total SNPs: {n_snps}")

    def _normalize_and_store_locations(self, locs, samples, train_indices, test_indices):
        """Normalize locations based on training data and store for each split.

        Args:
            locs: Array of location coordinates
            samples: Array of sample IDs
            train_indices: Indices of training samples
            test_indices: Indices of test samples

        Returns
        -------
            normalized_locs: Array of all locations normalized using training parameters
        """
        # Get training locations and normalize them
        train_locs = locs[train_indices]
        self.trainIDs = samples[train_indices]
        (
            self.meanlong,
            self.sdlong,
            self.meanlat,
            self.sdlat,
            self.unnormedlocs,
            normalized_train_locs,
        ) = normalize_locs(train_locs)

        # Normalize all locations using the training parameters (vectorized)
        normalized_locs = np.empty_like(locs, dtype=np.float64)
        normalized_locs[:, 0] = (locs[:, 0] - self.meanlong) / self.sdlong
        normalized_locs[:, 1] = (locs[:, 1] - self.meanlat) / self.sdlat

        # Store normalized locations for each split
        self.trainlocs = normalized_train_locs
        self.testlocs = normalized_locs[test_indices]

        return normalized_locs
