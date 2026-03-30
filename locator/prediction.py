"""Prediction functionality for locator"""

import json
import warnings

import h5py
import numpy as np
import pandas as pd


class PredictionMixin:
    """Mixin class providing prediction functionality for Locator."""

    def predict(  # noqa: C901
        self,
        boot=0,
        verbose=True,
        prediction_genotypes=None,  # Deprecated - use genotypes instead
        genotypes=None,  # New: full genotype array for tf.data
        samples=None,  # New: sample IDs
        indices=None,  # New: which samples to predict (default: NA samples)
        return_df=False,
        save_preds_to_disk=True,
        site_order=None,
    ):
        """Make predictions for samples with unknown locations.

        Args:
            boot (int, optional): Bootstrap replicate number. Defaults to 0.
            verbose (bool, optional): Whether to print validation metrics. Defaults to True.
            prediction_genotypes (numpy.ndarray, optional): DEPRECATED - use genotypes parameter.
                Override default prediction genotypes. Used for jacknife resampling. Defaults to None.
            genotypes (numpy.ndarray, optional): Full genotype array for creating tf.data dataset.
                Should be the original unfiltered genotypes. Defaults to None.
            samples (numpy.ndarray, optional): Sample IDs corresponding to genotypes. Defaults to None.
            indices (numpy.ndarray, optional): Indices of samples to predict on.
                If None, predicts on samples without coordinates (self.pred_indices). Defaults to None.
            return_df (bool, optional): Whether to return predictions as pandas DataFrame.
                Defaults to False.
            save_preds_to_disk (bool, optional): Whether to save predictions to disk.
                Defaults to True.
            site_order (np.ndarray, optional): Array of SNP indices for bootstrap resampling.
                If provided, SNPs will be reordered according to these indices during prediction.
                Used for bootstrap analyses to ensure consistent resampling between train and predict.

        Returns
        -------
            numpy.ndarray or pandas.DataFrame: Array of predicted coordinates or DataFrame with
                x,y coordinates and sampleID columns
        """
        if self.model is None:
            raise ValueError("Model must be trained before prediction")

        # Check if using new tf.data approach or old array approach
        if genotypes is not None:
            # New tf.data approach
            if prediction_genotypes is not None:
                warnings.warn(
                    "prediction_genotypes parameter is deprecated. Use genotypes parameter instead.",
                    DeprecationWarning,
                    stacklevel=2,
                )

            # Import required modules
            from .data import IndexSet, make_tf_dataset
            from .data import filter_snps_legacy as filter_snps

            # Determine which samples to predict
            locs = None
            if indices is None:
                # For new tf.data API, determine NA samples from provided data
                # Get sample data to identify NA samples
                if hasattr(self, "_sample_data_df"):
                    sample_data, locs = self.sort_samples(samples)
                else:
                    sample_data_path = self.config.get("sample_data")
                    if sample_data_path:
                        sample_data, locs = self.sort_samples(samples, sample_data_path)
                    else:
                        # No sample data available, fall back to pred_indices
                        if hasattr(self, "pred_indices"):
                            indices = self.pred_indices
                        else:
                            empty_df = pd.DataFrame(columns=["sampleID", "x", "y"])
                            if save_preds_to_disk:
                                empty_df.to_csv(
                                    f"{self.config['out']}_predlocs.csv", index=False
                                )
                            return empty_df if return_df else None

                # If we got sample data, find NA samples
                if locs is not None:
                    na_mask = np.isnan(locs[:, 0]) | np.isnan(locs[:, 1])
                    indices = np.where(na_mask)[0]
                    if len(indices) == 0:
                        # No NA samples, predict on all if in 'separate' mode
                        if (
                            hasattr(self, "config")
                            and self.config.get("na_action") == "separate"
                        ):
                            indices = np.arange(len(samples))
                        else:
                            empty_df = pd.DataFrame(columns=["sampleID", "x", "y"])
                            if save_preds_to_disk:
                                empty_df.to_csv(
                                    f"{self.config['out']}_predlocs.csv", index=False
                                )
                            return empty_df if return_df else None

            # Check if we have any samples to predict
            if len(indices) == 0:
                empty_df = pd.DataFrame(columns=["sampleID", "x", "y"])
                if save_preds_to_disk:
                    empty_df.to_csv(f"{self.config['out']}_predlocs.csv", index=False)
                return empty_df if return_df else None

            # Use stored samples if not provided
            if samples is None:
                if hasattr(self, "samples"):
                    samples = self.samples
                else:
                    raise ValueError("samples must be provided or stored from training")

            # Filter genotypes using the same parameters as training
            if hasattr(self, "filtered_genotypes"):
                # Use stored filtered genotypes if available
                filtered_genotypes = self.filtered_genotypes
            else:
                # Apply filtering to the provided genotypes
                filtered_genotypes = filter_snps(
                    genotypes,
                    min_mac=self.config.get("min_mac", 2),
                    max_snps=self.config.get("max_SNPs"),
                    impute=self.config.get("impute_missing", False),
                )

            # Create IndexSet for prediction
            predict_index_set = IndexSet(
                indices={"predict": indices},
                total_samples=len(samples),
                na_mask=None,  # Not needed for prediction
            )

            # Create dummy coordinates for prediction (values don't matter)
            dummy_coords = np.zeros((len(samples), 2))

            # Create prediction dataset
            predict_dataset = make_tf_dataset(
                genotypes=filtered_genotypes,  # Use filtered genotypes
                coordinates=dummy_coords,
                index_set=predict_index_set,
                split="predict",
                batch_size=self.config.get("batch_size", 256),
                training=False,
                cache=True,
                site_order=site_order,
            )

            # Get predictions
            predictions = self.model.predict(predict_dataset, verbose=verbose)

            # Store the indices we predicted on for later use
            prediction_indices = indices

        else:
            # Old array-based approach (for backward compatibility)
            if prediction_genotypes is not None:
                warnings.warn(
                    "Using deprecated array-based prediction. Consider using genotypes parameter for better memory efficiency.",
                    DeprecationWarning,
                    stacklevel=2,
                )

            # Use provided prediction genotypes if available, otherwise use stored ones
            predgen = (
                prediction_genotypes
                if prediction_genotypes is not None
                else self.predgen
            )

            # Apply site resampling if site_order is provided
            if site_order is not None and predgen is not None and len(predgen) > 0:
                predgen = predgen[:, site_order]

            # Check if there are any samples to predict
            if predgen is None or len(predgen) == 0:
                # Return empty DataFrame with correct columns
                empty_df = pd.DataFrame(columns=["sampleID", "x", "y"])
                if save_preds_to_disk:
                    empty_df.to_csv(f"{self.config['out']}_predlocs.csv", index=False)
                return empty_df if return_df else None

            # Get predictions
            predictions = self.model.predict(predgen)

            # Use stored pred_indices
            prediction_indices = (
                self.pred_indices if hasattr(self, "pred_indices") else None
            )

        # Denormalize predictions
        predictions = predictions.copy()
        predictions[:, 0] = predictions[:, 0] * self.sdlong + self.meanlong
        predictions[:, 1] = predictions[:, 1] * self.sdlat + self.meanlat

        # Create DataFrame
        pred_df = pd.DataFrame(predictions, columns=["x", "y"])

        # Add sample IDs
        if samples is not None and prediction_indices is not None:
            # New approach: use provided samples and indices
            pred_df.insert(0, "sampleID", samples[prediction_indices])
        elif hasattr(self, "samples") and hasattr(self, "pred_indices"):
            # Old approach: use stored values
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

    def load_model(self, weights_path):
        """Load a trained model from saved weights.

        This method loads a model from HDF5 weights file and restores the
        preprocessing parameters needed for making predictions.

        Args:
            weights_path (str): Path to the saved HDF5 weights file

        Returns
        -------
            dict: Dictionary containing loaded metadata including normalization params

        Raises
        ------
            ValueError: If weights file cannot be loaded or is missing metadata
        """
        import os

        if not os.path.exists(weights_path):
            raise ValueError(f"Weights file not found: {weights_path}")

        # Load metadata from HDF5 file
        metadata = {}
        try:
            with h5py.File(weights_path, "r") as f:
                # Load normalization parameters
                self.meanlong = float(f.attrs.get("coord_meanlong", 0.0))
                self.sdlong = float(f.attrs.get("coord_sdlong", 1.0))
                self.meanlat = float(f.attrs.get("coord_meanlat", 0.0))
                self.sdlat = float(f.attrs.get("coord_sdlat", 1.0))

                metadata["normalization"] = {
                    "meanlong": self.meanlong,
                    "sdlong": self.sdlong,
                    "meanlat": self.meanlat,
                    "sdlat": self.sdlat,
                }

                # Load preprocessing parameters
                metadata["preprocessing"] = {
                    "min_mac": int(f.attrs.get("min_mac", 2)),
                    "max_SNPs": int(f.attrs.get("max_SNPs", -1)),
                    "impute_missing": bool(f.attrs.get("impute_missing", False)),
                }
                if metadata["preprocessing"]["max_SNPs"] == -1:
                    metadata["preprocessing"]["max_SNPs"] = None

                # Load other metadata
                metadata["n_samples"] = int(f.attrs.get("n_samples", 0))
                metadata["n_snps"] = int(f.attrs.get("n_snps", 0))
                metadata["metadata_version"] = str(
                    f.attrs.get("metadata_version", "unknown")
                )
                metadata["locator_version"] = str(
                    f.attrs.get("locator_version", "unknown")
                )
                metadata["save_date"] = str(f.attrs.get("save_date", "unknown"))

                # Load config if available
                config_json = f.attrs.get("config_json", None)
                if config_json:
                    metadata["config"] = json.loads(config_json)
                    # Update current config with loaded values
                    self.config.update(metadata["config"])

            print(f"Loaded model metadata from {weights_path}")
            print(
                f"Model trained on {metadata['n_samples']} samples with {metadata['n_snps']} SNPs"
            )
            print(
                f"Normalization params: mean_long={self.meanlong:.4f}, sd_long={self.sdlong:.4f}"
            )

        except Exception as e:
            # For backward compatibility with models saved before metadata feature
            warnings.warn(
                f"Could not load metadata from weights file: {e}\n"
                "This may be an older model without saved metadata. "
                "Normalization parameters will need to be set manually."
            )
            metadata = None

        # Create the model architecture if not already created
        if self.model is None:
            # Infer architecture from weights or use config
            # This requires knowing the input shape - will be set when genotypes are loaded
            warnings.warn(
                "Model architecture not yet created. "
                "Call train() with setup_only=True after loading genotypes."
            )

        # Load the weights if model exists
        if self.model is not None:
            self.model.load_weights(weights_path)
            print("Loaded weights into model")

        return metadata

    def predict_from_weights(
        self,
        weights_path,
        genotypes,
        samples,
        sample_data_file=None,
        save_preds_to_disk=True,
        return_df=True,
    ):
        """Convenience method to load weights and make predictions.

        This method combines loading a saved model and making predictions
        in a single call. It handles preprocessing the genotypes using
        the same parameters that were used during training.

        Args:
            weights_path (str): Path to saved HDF5 weights file
            genotypes (numpy.ndarray): Genotype data to predict on
            samples (numpy.ndarray): Sample IDs corresponding to genotypes
            sample_data_file (str, optional): Path to sample data file
            save_preds_to_disk (bool): Whether to save predictions to disk
            return_df (bool): Whether to return predictions as DataFrame

        Returns
        -------
            numpy.ndarray or pandas.DataFrame: Predictions
        """
        # Load the model and metadata
        metadata = self.load_model(weights_path)

        # Store samples
        self.samples = samples

        # Get sample data to identify prediction samples
        sample_data, locs = self._resolve_locations(samples, sample_data_file)

        # Find samples without coordinates (to predict)
        na_mask = np.isnan(locs[:, 0]) | np.isnan(locs[:, 1])
        self.pred_indices = np.where(na_mask)[0]

        if len(self.pred_indices) == 0:
            warnings.warn("No samples found without coordinates. Nothing to predict.")
            return (
                pd.DataFrame(columns=["sampleID", "x", "y"])
                if return_df
                else np.array([])
            )

        # Apply preprocessing using saved parameters
        if metadata and "preprocessing" in metadata:
            from .data import filter_snps_legacy as filter_snps

            filtered_genotypes = filter_snps(
                genotypes,
                min_mac=metadata["preprocessing"]["min_mac"],
                max_snps=metadata["preprocessing"]["max_SNPs"],
                impute=metadata["preprocessing"]["impute_missing"],
            )
        else:
            # Use current config if no metadata
            from .data import filter_snps_legacy as filter_snps

            filtered_genotypes = filter_snps(
                genotypes,
                min_mac=self.config.get("min_mac", 2),
                max_snps=self.config.get("max_SNPs"),
                impute=self.config.get("impute_missing", False),
            )

        # Prepare prediction genotypes
        self.predgen = np.transpose(filtered_genotypes[:, self.pred_indices])

        # Create model if needed
        if self.model is None:
            from .models import create_network

            n_snps = filtered_genotypes.shape[0]
            self.model = create_network(
                input_shape=n_snps,
                width=self.config.get("width", 256),
                n_layers=self.config.get("nlayers", 8),
                dropout_prop=self.config.get("dropout_prop", 0.25),
                optimizer_config={
                    "algo": self.config.get("optimizer_algo", "adam"),
                    "learning_rate": self.config.get("learning_rate", 0.001),
                    "weight_decay": self.config.get("weight_decay", 0.004),
                },
            )
            self.model.load_weights(weights_path)

        # Make predictions
        return self.predict(save_preds_to_disk=save_preds_to_disk, return_df=return_df)

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

        Returns
        -------
            If return_df is True, returns pandas DataFrame with predictions
            Otherwise returns None
        """
        if not hasattr(self, "holdout_idx") or not hasattr(self, "holdout_locs"):
            raise ValueError("No holdout data found. Run train_holdout() first.")

        if verbose:
            print("Predicting locations for holdout samples...")

        # Use tf.data approach for predictions
        from .data import IndexSet, make_tf_dataset

        # Create IndexSet for holdout samples
        holdout_index_set = IndexSet(
            indices={"predict": self.holdout_idx},
            total_samples=len(self.samples),
            na_mask=None,
        )

        # Create dummy coordinates for prediction
        dummy_coords = np.zeros((len(self.samples), 2))

        # Create prediction dataset
        predict_dataset = make_tf_dataset(
            genotypes=self.filtered_genotypes,
            coordinates=dummy_coords,
            index_set=holdout_index_set,
            split="predict",
            batch_size=self.config.get("batch_size", 256),
            training=False,
            cache=True,
        )

        # Get predictions
        predictions = self.model.predict(predict_dataset, verbose=verbose)

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
                import matplotlib.pyplot as plt  # noqa: F401
                from IPython.display import display  # noqa: F401

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
                        show=True,  # Explicitly show since we're in a notebook
                    )

            except ImportError:
                # Not in a notebook, skip plotting
                pass

            return pred_df

        return predictions
