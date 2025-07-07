"""Ensemble functionality mixin for locator"""

import numpy as np
import pandas as pd
from tensorflow import keras

from .data import IndexSet, NormalizationParams
from .data import filter_snps_legacy as filter_snps
from .data import make_tf_dataset, normalize_locs

# from tqdm import tqdm


class EnsembleMixin:
    """Mixin class providing ensemble functionality for Locator."""

    def create_ensemble_folds(
        self, genotypes, samples, k=5, training_set_indices=None, na_action=None
    ):
        """Create k-fold splits for ensemble training using IndexSet.

        Args:
            genotypes: GenotypeArray containing genetic data
            samples: Array of sample IDs
            k: Number of folds (default: 5)
            training_set_indices: Optional array of indices to use for training+validation.
                If provided, only these samples will be used to create k-folds.
            na_action: How to handle NA samples ('separate', 'exclude', 'fail').
                If None, uses self.na_action

        Returns:
            dict: Dictionary with fold information:
                - 'index_sets': List of IndexSet objects for each fold
                - 'fold_indices': Legacy format dict for backward compatibility
                - 'sample_status': Sample status information
        """
        # Use instance default if na_action not specified
        if na_action is None:
            na_action = self.na_action

        # Get sample data and locations
        sample_data, locs = self.sort_samples(samples)

        # Get sample status
        status = self.get_sample_status(samples, sample_data)

        # Apply NA action
        if na_action == "fail" and status["n_na"] > 0:
            raise ValueError(
                f"Found {status['n_na']} samples without coordinates. "
                f"Set na_action='separate' or 'exclude' to proceed."
            )

        # Prepare indices for k-fold splitting
        n_samples = len(samples)
        na_mask = np.isnan(locs[:, 0]) | np.isnan(locs[:, 1])

        # If training_set_indices provided, create a mask for k-fold splitting
        if training_set_indices is not None:
            training_set_indices = np.array(training_set_indices)
            # Validate indices
            if not np.all(np.isin(training_set_indices, range(n_samples))):
                raise ValueError("training_set_indices contains invalid indices")

            # Create mask: True for samples NOT in training set (should be excluded from k-fold)
            exclude_mask = np.ones(n_samples, dtype=bool)
            exclude_mask[training_set_indices] = False

            # Combine with NA mask for k-fold splitting
            # Samples are excluded from k-fold if they are NA OR not in training set
            if na_action == "separate":
                kfold_exclude_mask = exclude_mask | na_mask
            else:
                kfold_exclude_mask = exclude_mask
        else:
            # Use only NA mask for k-fold splitting
            kfold_exclude_mask = (
                na_mask if na_action == "separate" else np.zeros(n_samples, dtype=bool)
            )

        # Create k-fold index sets
        fold_index_sets = IndexSet.k_fold_split(
            n=n_samples,
            k=k,
            na_mask=kfold_exclude_mask,
            seed=self.config.get("seed", 12345),
        )

        # Convert to legacy format for backward compatibility
        fold_indices = {}
        for i, index_set in enumerate(fold_index_sets):
            # For ensemble, we want train/val from the fold, and pred from NA samples
            if na_action == "separate":
                # Prediction set includes all NA samples
                pred_idx = np.where(na_mask)[0]
            else:
                # No prediction set
                pred_idx = np.array([], dtype=int)

            fold_indices[i] = {
                "train": index_set.train,
                "val": index_set.test,  # In k-fold, 'test' is used as validation
                "pred": pred_idx,
            }

        return {
            "index_sets": fold_index_sets,
            "fold_indices": fold_indices,
            "sample_status": status,
        }

    def train_ensemble(
        self,
        genotypes,
        samples,
        k=5,
        training_set_indices=None,
        na_action=None,
        augment_data=False,
        flip_rate=0.05,
        save_fold_models=True,
        verbose=True,
    ):
        """Train an ensemble of k models using k-fold cross-validation.

        This method trains k models, each on a different k-fold split of the data.
        It uses the modern tf.data pipeline for memory efficiency and supports
        all standard Locator features including NA handling and data augmentation.

        Args:
            genotypes: GenotypeArray containing genetic data
            samples: Array of sample IDs
            k: Number of folds/models in ensemble (default: 5)
            training_set_indices: Optional array of indices to restrict training
            na_action: How to handle NA samples ('separate', 'exclude', 'fail')
            augment_data: Whether to apply data augmentation (default: False)
            flip_rate: Rate for genotype flipping augmentation (default: 0.05)
            save_fold_models: Whether to save individual fold models (default: True)
            verbose: Whether to show training progress (default: True)

        Returns:
            dict: Dictionary containing:
                - 'histories': List of training histories for each fold
                - 'models': List of trained model configurations
                - 'normalization_params': Averaged normalization parameters
                - 'fold_info': Information about fold splits
        """
        # Store samples for later use
        self.samples = samples

        # Create folds using IndexSet
        fold_info = self.create_ensemble_folds(
            genotypes, samples, k, training_set_indices, na_action
        )

        # Filter SNPs once before training
        filtered_genotypes = self._filter_genotypes(genotypes)

        # Store genotypes for later prediction
        self._ensemble_genotypes = genotypes

        # Store fold information
        self._ensemble_fold_info = fold_info
        self._ensemble_models = []
        self._ensemble_histories = []
        self._ensemble_norm_params = []

        # Configure augmentation if requested
        if augment_data:
            self.config["augmentation"] = {"enabled": True, "flip_rate": flip_rate}

        # Train each fold
        for fold_idx in range(k):
            if verbose:
                print(f"\nTraining fold {fold_idx + 1}/{k}")

            # Get indices for this fold
            index_set = fold_info["index_sets"][fold_idx]
            train_idx = index_set.train
            val_idx = index_set.test  # k-fold uses 'test' as validation

            # Get locations for normalization
            _, locs = self.sort_samples(samples)
            train_locs = locs[train_idx]

            # Normalize locations using the standard function
            meanlong, sdlong, meanlat, sdlat, _, normalized_train_locs = normalize_locs(
                train_locs
            )

            # Store normalization parameters
            norm_params = {
                "meanlong": meanlong,
                "sdlong": sdlong,
                "meanlat": meanlat,
                "sdlat": sdlat,
            }
            self._ensemble_norm_params.append(norm_params)

            # Create model for this fold
            model = self._create_model(n_features=filtered_genotypes.shape[0])

            # Normalize validation locations using the same parameters
            val_locs = locs[val_idx]
            normalized_val_locs = self._apply_normalization(val_locs, norm_params)

            # Create normalized location array for all samples
            normalized_locs = np.zeros((len(samples), 2))
            normalized_locs[train_idx] = normalized_train_locs
            normalized_locs[val_idx] = normalized_val_locs

            # Create datasets using modern pipeline
            augment_config = None
            if self.config.get("augmentation", {}).get("enabled", False):
                augment_config = {
                    "enabled": True,
                    "flip_rate": self.config.get("augmentation", {}).get(
                        "flip_rate", 0.05
                    ),
                }

            train_dataset = make_tf_dataset(
                genotypes=filtered_genotypes,
                coordinates=normalized_locs,
                index_set=index_set,
                split="train",
                batch_size=self._determine_batch_size(len(train_idx)),
                augment=augment_config,
                training=True,
                cache=True,
            )

            val_dataset = make_tf_dataset(
                genotypes=filtered_genotypes,
                coordinates=normalized_locs,
                index_set=index_set,
                split="test",  # k-fold uses 'test' for validation
                batch_size=self._determine_batch_size(len(val_idx)),
                training=False,
                cache=True,
            )

            # Create callbacks
            fold_config = self.config.copy()
            fold_config["out"] = f"{self.config['out']}_fold{fold_idx}"
            fold_config["save_fold_models"] = save_fold_models

            # Temporarily update self.config for callback creation
            original_config = self.config
            self.config = fold_config
            callbacks = self._create_callbacks()
            self.config = original_config

            # Train model
            history = model.fit(
                train_dataset,
                epochs=self.config.get("max_epochs", 5000),
                verbose=self.config.get("keras_verbose", 1) if verbose else 0,
                validation_data=val_dataset,
                callbacks=callbacks,
            )

            # Store model info
            model_info = {
                "fold": fold_idx,
                "model": model,
                "history": history,
                "norm_params": norm_params,
                "train_indices": train_idx,
                "val_indices": val_idx,
                "weights_file": (
                    f"{fold_config['out']}.weights.h5" if save_fold_models else None
                ),
            }
            self._ensemble_models.append(model_info)
            self._ensemble_histories.append(history)

            # Clear session to free memory
            keras.backend.clear_session()

        # Calculate averaged normalization parameters
        avg_norm_params = self._average_normalization_params(self._ensemble_norm_params)

        # Store averaged parameters on instance
        self.meanlong = avg_norm_params["meanlong"]
        self.sdlong = avg_norm_params["sdlong"]
        self.meanlat = avg_norm_params["meanlat"]
        self.sdlat = avg_norm_params["sdlat"]

        return {
            "histories": self._ensemble_histories,
            "models": self._ensemble_models,
            "normalization_params": avg_norm_params,
            "fold_info": fold_info,
        }

    def predict_ensemble(
        self,
        genotypes=None,
        samples=None,
        indices=None,
        include_fold_predictions=False,
        return_std=False,
        return_df=True,
        save_predictions=True,
    ):
        """Make predictions using the ensemble of models.

        Args:
            genotypes: GenotypeArray for prediction (if None, uses stored data)
            samples: Sample IDs (if None, uses stored samples)
            indices: Specific indices to predict on (if None, predicts all)
            include_fold_predictions: Include individual fold predictions in output
            return_std: Return standard deviation across ensemble predictions
            return_df: Return results as DataFrame (default: True)
            save_predictions: Save predictions to disk (default: True)

        Returns:
            pd.DataFrame or np.ndarray: Ensemble predictions with optional std
        """
        # Validate inputs
        self._validate_ensemble_prediction(genotypes, samples)

        # Setup prediction indices
        if indices is None:
            indices = np.arange(len(samples))

        # Get predictions from all folds
        fold_predictions = self._collect_fold_predictions(genotypes, samples, indices)

        # Calculate ensemble statistics
        ensemble_mean = np.mean(fold_predictions, axis=0)
        ensemble_std = np.std(fold_predictions, axis=0) if return_std else None

        # Format and return results
        if return_df:
            return self._format_ensemble_predictions_df(
                samples,
                indices,
                ensemble_mean,
                ensemble_std,
                fold_predictions,
                include_fold_predictions,
                save_predictions,
            )
        else:
            return (ensemble_mean, ensemble_std) if return_std else ensemble_mean

    def _validate_ensemble_prediction(self, genotypes, samples):
        """Validate inputs for ensemble prediction."""
        if not hasattr(self, "_ensemble_models") or not self._ensemble_models:
            raise ValueError("No ensemble models trained. Run train_ensemble() first.")

        if genotypes is None or samples is None:
            raise ValueError("genotypes and samples must be provided for prediction")

    def _collect_fold_predictions(self, genotypes, samples, indices):
        """Collect predictions from all fold models."""
        filtered_genotypes = self._filter_genotypes(genotypes)
        fold_predictions = []

        for model_info in self._ensemble_models:
            predictions = self._predict_single_fold(
                model_info, filtered_genotypes, samples, indices
            )
            fold_predictions.append(predictions)

        return np.array(fold_predictions)  # Shape: (n_folds, n_samples, 2)

    def _predict_single_fold(self, model_info, filtered_genotypes, samples, indices):
        """Make predictions using a single fold model."""
        # Load model weights if needed
        model = model_info["model"]
        if model_info["weights_file"]:
            model.load_weights(model_info["weights_file"])

        # Create prediction dataset
        pred_dataset = self._create_prediction_dataset(
            filtered_genotypes, samples, indices
        )

        # Make predictions
        predictions = model.predict(pred_dataset, verbose=0)

        # Denormalize using NormalizationParams
        norm_params = NormalizationParams(
            meanlong=model_info["norm_params"]["meanlong"],
            sdlong=model_info["norm_params"]["sdlong"],
            meanlat=model_info["norm_params"]["meanlat"],
            sdlat=model_info["norm_params"]["sdlat"],
        )

        return norm_params.reverse(predictions)

    def _create_prediction_dataset(self, filtered_genotypes, samples, indices):
        """Create tf.data dataset for prediction."""
        # Create IndexSet for prediction
        pred_index_set = IndexSet.from_manual(
            train=np.array([], dtype=int),
            test=indices,
            total_samples=len(samples),
        )

        # Create dummy coordinates (not used in prediction)
        dummy_coords = np.zeros((len(samples), 2))

        # Create and return dataset
        return make_tf_dataset(
            genotypes=filtered_genotypes,
            coordinates=dummy_coords,
            index_set=pred_index_set,
            split="test",
            batch_size=self.config.get("batch_size", 32),
            training=False,
            cache=False,
        )

    def _format_ensemble_predictions_df(
        self,
        samples,
        indices,
        ensemble_mean,
        ensemble_std,
        fold_predictions,
        include_fold_predictions,
        save_predictions,
    ):
        """Format ensemble predictions as DataFrame."""
        # Create base DataFrame
        result_df = pd.DataFrame(
            {
                "sampleID": samples[indices],
                "x": ensemble_mean[:, 0],
                "y": ensemble_mean[:, 1],
            }
        )

        # Add optional columns
        if ensemble_std is not None:
            result_df["x_std"] = ensemble_std[:, 0]
            result_df["y_std"] = ensemble_std[:, 1]

        if include_fold_predictions:
            for i in range(len(self._ensemble_models)):
                result_df[f"x_fold{i}"] = fold_predictions[i, :, 0]
                result_df[f"y_fold{i}"] = fold_predictions[i, :, 1]

        # Save if requested
        if save_predictions:
            filename = f"{self.config['out']}_ensemble_predictions.csv"
            result_df.to_csv(filename, index=False)
            if self.config.get("keras_verbose", 1) >= 1:
                print(f"Saved ensemble predictions to {filename}")

        return result_df

    def _filter_genotypes(self, genotypes):
        """Filter genotypes using standard Locator filtering."""
        return filter_snps(
            genotypes,
            min_mac=self.config.get("min_mac", 2),
            max_snps=self.config.get("max_SNPs"),
            impute=self.config.get("impute_missing", False),
        )

    def _apply_normalization(self, locations, norm_params):
        """Apply normalization to location coordinates using provided parameters."""
        return np.array(
            [
                [
                    (loc[0] - norm_params["meanlong"]) / norm_params["sdlong"],
                    (loc[1] - norm_params["meanlat"]) / norm_params["sdlat"],
                ]
                for loc in locations
            ]
        )

    def _average_normalization_params(self, norm_params_list):
        """Average normalization parameters across folds."""
        avg_params = {
            "meanlong": np.mean([p["meanlong"] for p in norm_params_list]),
            "sdlong": np.mean([p["sdlong"] for p in norm_params_list]),
            "meanlat": np.mean([p["meanlat"] for p in norm_params_list]),
            "sdlat": np.mean([p["sdlat"] for p in norm_params_list]),
        }
        return avg_params
