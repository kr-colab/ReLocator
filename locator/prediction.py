"""Prediction functionality for locator"""

import numpy as np
import pandas as pd
import warnings
import h5py
import json
from tensorflow import keras


class PredictionMixin:
    """Mixin class providing prediction functionality for Locator."""
    
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

    def load_model(self, weights_path):
        """Load a trained model from saved weights.
        
        This method loads a model from HDF5 weights file and restores the
        preprocessing parameters needed for making predictions.
        
        Args:
            weights_path (str): Path to the saved HDF5 weights file
            
        Returns:
            dict: Dictionary containing loaded metadata including normalization params
            
        Raises:
            ValueError: If weights file cannot be loaded or is missing metadata
        """
        import os
        if not os.path.exists(weights_path):
            raise ValueError(f"Weights file not found: {weights_path}")
            
        # Load metadata from HDF5 file
        metadata = {}
        try:
            with h5py.File(weights_path, 'r') as f:
                # Load normalization parameters
                self.meanlong = float(f.attrs.get('coord_meanlong', 0.0))
                self.sdlong = float(f.attrs.get('coord_sdlong', 1.0))
                self.meanlat = float(f.attrs.get('coord_meanlat', 0.0))
                self.sdlat = float(f.attrs.get('coord_sdlat', 1.0))
                
                metadata['normalization'] = {
                    'meanlong': self.meanlong,
                    'sdlong': self.sdlong,
                    'meanlat': self.meanlat,
                    'sdlat': self.sdlat
                }
                
                # Load preprocessing parameters
                metadata['preprocessing'] = {
                    'min_mac': int(f.attrs.get('min_mac', 2)),
                    'max_SNPs': int(f.attrs.get('max_SNPs', -1)),
                    'impute_missing': bool(f.attrs.get('impute_missing', False))
                }
                if metadata['preprocessing']['max_SNPs'] == -1:
                    metadata['preprocessing']['max_SNPs'] = None
                    
                # Load other metadata
                metadata['n_samples'] = int(f.attrs.get('n_samples', 0))
                metadata['n_snps'] = int(f.attrs.get('n_snps', 0))
                metadata['metadata_version'] = str(f.attrs.get('metadata_version', 'unknown'))
                metadata['locator_version'] = str(f.attrs.get('locator_version', 'unknown'))
                metadata['save_date'] = str(f.attrs.get('save_date', 'unknown'))
                
                # Load config if available
                config_json = f.attrs.get('config_json', None)
                if config_json:
                    metadata['config'] = json.loads(config_json)
                    # Update current config with loaded values
                    self.config.update(metadata['config'])
                    
            print(f"Loaded model metadata from {weights_path}")
            print(f"Model trained on {metadata['n_samples']} samples with {metadata['n_snps']} SNPs")
            print(f"Normalization params: mean_long={self.meanlong:.4f}, sd_long={self.sdlong:.4f}")
            
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
            print(f"Loaded weights into model")
            
        return metadata
    
    def predict_from_weights(
        self,
        weights_path,
        genotypes,
        samples,
        sample_data_file=None,
        save_preds_to_disk=True,
        return_df=True
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
            
        Returns:
            numpy.ndarray or pandas.DataFrame: Predictions
        """
        # Load the model and metadata
        metadata = self.load_model(weights_path)
        
        # Store samples
        self.samples = samples
        
        # Get sample data to identify prediction samples
        if hasattr(self, "_sample_data_df"):
            sample_data, locs = self.sort_samples(samples)
        else:
            sample_data_path = sample_data_file or self.config.get("sample_data")
            if not sample_data_path:
                raise ValueError("sample_data must be provided")
            sample_data, locs = self.sort_samples(samples, sample_data_path)
            
        # Find samples without coordinates (to predict)
        na_mask = np.isnan(locs[:, 0]) | np.isnan(locs[:, 1])
        self.pred_indices = np.where(na_mask)[0]
        
        if len(self.pred_indices) == 0:
            warnings.warn("No samples found without coordinates. Nothing to predict.")
            return pd.DataFrame(columns=['sampleID', 'x', 'y']) if return_df else np.array([])
            
        # Apply preprocessing using saved parameters
        if metadata and 'preprocessing' in metadata:
            from .data import filter_snps_legacy as filter_snps
            
            filtered_genotypes = filter_snps(
                genotypes,
                min_mac=metadata['preprocessing']['min_mac'],
                max_snps=metadata['preprocessing']['max_SNPs'],
                impute=metadata['preprocessing']['impute_missing']
            )
        else:
            # Use current config if no metadata
            from .data import filter_snps_legacy as filter_snps
            
            filtered_genotypes = filter_snps(
                genotypes,
                min_mac=self.config.get('min_mac', 2),
                max_snps=self.config.get('max_SNPs'),
                impute=self.config.get('impute_missing', False)
            )
            
        # Prepare prediction genotypes
        self.predgen = np.transpose(filtered_genotypes[:, self.pred_indices])
        
        # Create model if needed
        if self.model is None:
            from .models import create_network
            
            # Infer input shape from filtered genotypes
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
                }
            )
            
            # Now load the weights
            self.model.load_weights(weights_path)
            
        # Make predictions
        return self.predict(
            save_preds_to_disk=save_preds_to_disk,
            return_df=return_df
        )

    def sort_samples(self, samples=None, sample_data_file=None, reorder=True):
        """Sort samples and match with location data.

        This method matches samples with their location data and ensures consistent ordering
        between genotype and location data. It can use either a stored DataFrame from
        initialization or a provided sample data file.

        Args:
            samples (numpy.ndarray): Array of sample IDs from the genotype data
            sample_data_file (str, optional): Override path to tab-delimited file with
                columns 'sampleID', 'x', 'y'. If not provided, uses stored sample data.
            reorder (bool): If True, automatically reorder metadata to match genotype order.
                If False, raise error on order mismatch (default: True)

        Returns:
            tuple: A tuple containing:
                - sample_data (pandas.DataFrame): DataFrame with sample metadata and coordinates
                - locs (numpy.ndarray): Array of x,y coordinates for each sample

        Raises:
            ValueError: If samples not provided or if no sample data available
            ValueError: If sample IDs don't match between genotype and sample data (when reorder=False)
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

        # Convert the sampleID column to match the type of samples
        sample_data["sampleID"] = sample_data["sampleID"].astype(str)
        samples_str = [str(s) for s in samples]

        # Verify sample order matches using the correct column name
        # First check if we have the same number of samples
        if len(sample_data) != len(samples):
            if reorder:
                # Different number of samples - need to handle this case
                print(f"Sample count mismatch: {len(samples)} in genotypes, {len(sample_data)} in metadata")
                # We'll handle this by adding NA entries for missing samples during reordering
            else:
                raise ValueError(
                    f"Sample count mismatch: genotypes has {len(samples)} samples but metadata has {len(sample_data)}. "
                    f"Set reorder=True to handle this automatically."
                )
        
        # Check order for the samples we do have
        min_samples = min(len(sample_data), len(samples))
        order_matches = len(sample_data) == len(samples) and all(
            sample_data["sampleID"].iloc[x] == samples_str[x] for x in range(min_samples)
        )
        
        if not order_matches:
            if reorder:
                # Create a mapping DataFrame with genotype order
                sample_order_df = pd.DataFrame({
                    'sampleID': samples_str,
                    'geno_order': range(len(samples_str))
                })
                
                # Merge to reorder metadata to match genotype order
                reordered_data = sample_order_df.merge(
                    sample_data, 
                    on='sampleID', 
                    how='left'
                )
                
                # Check for samples in genotypes but not in metadata
                missing_in_meta = reordered_data[['x', 'y']].isna().any(axis=1).sum()
                if missing_in_meta > 0:
                    missing_ids = reordered_data[reordered_data['x'].isna()]['sampleID'].tolist()
                    warnings.warn(
                        f"{missing_in_meta} samples in genotypes have no metadata. "
                        f"First 10 missing: {missing_ids[:10]}"
                    )
                    # For k-fold and other analyses that need all samples to have metadata,
                    # this will cause issues. The user should use a complete metadata file.
                    if missing_in_meta == len(reordered_data):
                        raise ValueError(
                            "No samples from genotypes found in metadata! "
                            "Check that sample IDs match between files."
                        )
                
                # Check for samples in metadata but not in genotypes
                samples_set = set(samples_str)
                extra_in_meta = sample_data[~sample_data['sampleID'].isin(samples_set)]
                if len(extra_in_meta) > 0:
                    extra_ids = extra_in_meta['sampleID'].tolist()
                    warnings.warn(
                        f"{len(extra_in_meta)} samples in metadata are not in genotypes. "
                        f"First 10 extra: {extra_ids[:10]}"
                    )
                
                # Sort by genotype order and drop the order column
                sample_data = reordered_data.sort_values('geno_order').drop('geno_order', axis=1)
                
                # Print summary of reordering
                print(f"Reordered metadata to match genotype sample order.")
                print(f"Total samples in genotypes: {len(samples)}")
                print(f"Samples with coordinates: {len(samples) - missing_in_meta}")
                if missing_in_meta > 0:
                    print(f"Samples without coordinates (NA): {missing_in_meta}")
                    print(f"Note: K-fold CV will only use the {len(samples) - missing_in_meta} samples with known locations")
                
            else:
                raise ValueError(
                    "Sample ordering failed! Check that sample IDs match the genotype data. "
                    "Set reorder=True to automatically reorder metadata to match genotype order."
                )

        # Extract location data
        locs = np.array(sample_data[["x", "y"]])

        return sample_data, locs

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
                        show=True,  # Explicitly show since we're in a notebook
                    )

            except ImportError:
                # Not in a notebook, skip plotting
                pass

            return pred_df

        return predictions