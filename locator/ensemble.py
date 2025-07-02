"""Ensemble functionality for locator"""

import numpy as np
import pandas as pd
from tensorflow import keras
import tensorflow as tf

from .core import Locator
from .models import create_network
from .data import filter_snps_legacy as filter_snps, normalize_locs


def flip_genotypes(genotypes, locations, mask_rate=0.05):
    """Randomly flip genotype values with probability mask_rate"""
    mask = tf.random.uniform(tf.shape(genotypes)) < mask_rate
    return tf.where(mask, 1 - genotypes, genotypes), locations


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

        # Add key configuration parameters
        key_params = [
            "k_folds",
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
        ]

        # Add k_folds
        html.append(
            f"<tr><td style='padding:5px'>k_folds</td>"
            f"<td style='padding:5px'>{self.k_folds}</td></tr>"
        )

        for param in key_params[1:]:  # Skip k_folds since we already added it
            if param in self.base_config:
                html.append(
                    f"<tr><td style='padding:5px'>{param}</td>"
                    f"<td style='padding:5px'>{self.base_config[param]}</td></tr>"
                )

        html.append("</table>")

        # Add status
        html.append("<h4>Status:</h4>")
        html.append("<ul>")

        if self.models:
            html.append(f"<li>Models trained: {len(self.models)} models ✓</li>")

            # Add fold information
            if self.fold_indices:
                html.append("<li>Fold splits:")
                html.append("<ul>")
                for fold, indices in self.fold_indices.items():
                    html.append(
                        f"<li>Fold {fold}: {len(indices['train'])} train, "
                        f"{len(indices['val'])} val, {len(indices['pred'])} pred</li>"
                    )
                html.append("</ul></li>")
        else:
            html.append("<li>Models: Not trained</li>")

        # Normalization status
        if all(
            x is not None for x in [self.meanlong, self.sdlong, self.meanlat, self.sdlat]
        ):
            html.append("<li>Location normalization: Computed (averaged) ✓</li>")
        else:
            html.append("<li>Location normalization: Not computed</li>")

        # Sample data status
        if "sample_data" in self.base_config:
            html.append("<li>Sample data: Path provided</li>")
        else:
            html.append("<li>Sample data: Not provided</li>")

        # Genotype data status
        if any(x in self.base_config for x in ["zarr", "vcf", "genotype_data"]):
            html.append("<li>Genotype data: Path provided</li>")
        else:
            html.append("<li>Genotype data: Not provided</li>")

        html.append("</ul>")
        html.append("</div>")

        return "".join(html)