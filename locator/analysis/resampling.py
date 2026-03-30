"""Resampling analysis methods (jacknife, bootstrap)."""

import numpy as np
import pandas as pd
from tensorflow import keras
from tqdm import tqdm


class ResamplingMixin:
    """Mixin providing resampling-based analysis methods."""

    def run_jacknife(  # noqa: C901
        self,
        genotypes,
        samples,
        prop=0.05,
        return_df=False,
        save_full_pred_matrix=True,
        na_action=None,
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
            na_action: How to handle NA samples ('separate', 'exclude', 'fail').
                If None, uses self.na_action

        Returns
        -------
            pandas.DataFrame or None: If return_df=True, returns DataFrame containing
                all predictions, with columns named 'x_0', 'y_0', 'x_1', 'y_1', etc.
                for each jacknife replicate. Row index contains sample IDs.

        Notes
        -----
            - With na_action='separate': Trains on samples with known locations,
              can predict on samples with NA locations
            - With na_action='exclude': Only uses samples with known locations
            - With na_action='fail': Raises error if any NA samples found
        """
        # Store samples
        self.samples = samples

        na_action, status = self._validate_na_action(
            samples, na_action, "Jacknife analysis"
        )

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
        self.train(genotypes=genotypes, samples=samples, na_action=na_action)

        # Store original data for reuse
        original_filtered_genotypes = (
            self.filtered_genotypes if hasattr(self, "filtered_genotypes") else None
        )
        original_index_set = self.index_set if hasattr(self, "index_set") else None

        # Store original locations and model
        original_trainlocs = self.trainlocs if hasattr(self, "trainlocs") else None
        original_testlocs = self.testlocs if hasattr(self, "testlocs") else None

        # Calculate number of jacknife replicates
        n_jack = int(np.ceil(1.0 / prop))
        print(f"starting jacknife resampling ({n_jack} replicates)")

        for boot in tqdm(range(n_jack)):
            # Generate indices of sites to keep (jackknife drops a subset)
            if original_filtered_genotypes is not None:
                n_sites = original_filtered_genotypes.shape[0]
            else:
                raise ValueError(
                    "Jacknife requires filtered_genotypes from initial training"
                )

            # For jacknife, we systematically drop different subsets
            # This ensures each SNP is dropped in exactly one replicate
            sites_per_replicate = int(n_sites * prop)
            start_idx = boot * sites_per_replicate
            end_idx = min(start_idx + sites_per_replicate, n_sites)

            # Create array of all site indices except those being dropped
            all_sites = np.arange(n_sites)
            sites_to_keep = np.concatenate([all_sites[:start_idx], all_sites[end_idx:]])

            # Clear model to force retraining
            self.model = None
            self.sample_weights = None

            # Restore filtered genotypes and index set
            if original_filtered_genotypes is not None:
                self.filtered_genotypes = original_filtered_genotypes
                self.index_set = original_index_set

            # Train with subset of sites using site_order
            # site_order acts as a selection of which sites to use
            self.train(
                genotypes=genotypes,
                samples=samples,
                boot=boot,
                train_locs=original_trainlocs,
                test_locs=original_testlocs,
                site_order=sites_to_keep,  # Use subset of sites
                na_action=na_action,
            )

            # Get predictions using the trained model with tf.data
            preds = self.predict(
                boot=boot,
                verbose=False,
                genotypes=genotypes,  # Pass full genotypes for tf.data
                samples=samples,
                indices=self.pred_indices if hasattr(self, "pred_indices") else None,
                site_order=sites_to_keep,  # Pass same site order for predictions
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

    def run_bootstraps(  # noqa: C901
        self,
        genotypes,
        samples,
        n_bootstraps=50,
        return_df=False,
        save_full_pred_matrix=True,
        na_action=None,
    ):
        """Run bootstrap analysis by resampling SNPs with replacement.

        Args:
            genotypes: Array of genotype data
            samples: Sample IDs corresponding to genotypes
            n_bootstraps: Number of bootstrap replicates to run
            return_df: Whether to return DataFrame with all predictions
            save_full_pred_matrix: Whether to save full prediction matrix to disk
            na_action: How to handle NA samples ('separate', 'exclude', 'fail').
                If None, uses self.na_action

        Returns
        -------
            pandas.DataFrame or None: If return_df=True, returns DataFrame with predictions
                for each bootstrap, otherwise None

        Notes
        -----
            - With na_action='separate': Trains on samples with known locations,
              can predict on samples with NA locations
            - With na_action='exclude': Only uses samples with known locations
            - With na_action='fail': Raises error if any NA samples found
        """
        # Store samples
        self.samples = samples

        na_action, status = self._validate_na_action(
            samples, na_action, "Bootstrap analysis"
        )

        # Set bootstrap flag in config
        self.config["bootstrap"] = True
        self.config["nboots"] = n_bootstraps

        # Initial training to set up model and data - pass na_action
        self.train(genotypes=genotypes, samples=samples, na_action=na_action)

        # Store original locations and filtered genotypes for reuse
        original_trainlocs = self.trainlocs
        original_testlocs = self.testlocs
        original_filtered_genotypes = (
            self.filtered_genotypes if hasattr(self, "filtered_genotypes") else None
        )

        original_index_set = self.index_set if hasattr(self, "index_set") else None

        # Pre-calculate KDE bandwidth if needed
        original_bandwidth = None
        bandwidth_calculated = False

        if (
            self.config.get("weight_samples", {}).get("enabled", False)
            and self.config.get("weight_samples", {}).get("method") == "KD"
        ):
            existing_bandwidth = self.config.get("weight_samples", {}).get("bandwidth")

            if existing_bandwidth is None and len(original_trainlocs) > 1:
                print("Pre-calculating optimal KDE bandwidth for bootstrap analysis...")

                from ..sample_weights import get_global_bandwidth_optimizer

                optimizer = get_global_bandwidth_optimizer()

                optimal_bandwidth = optimizer.get_bandwidth(
                    original_trainlocs,
                    cache_key=f"bootstrap_n{len(original_trainlocs)}",
                    n_bandwidths=self.config.get("weight_samples", {}).get(
                        "n_bandwidths", 100
                    ),
                    verbose=True,
                )

                # Temporarily set in config
                self.config["weight_samples"]["bandwidth"] = optimal_bandwidth
                bandwidth_calculated = True

                print(f"Using bandwidth: {optimal_bandwidth:.3f}")

        # Create lists to store predictions
        pred_dfs = []

        print("starting bootstrap resampling")

        for boot in tqdm(range(n_bootstraps)):
            # Set random seed
            np.random.seed(np.random.choice(range(int(1e6)), 1))

            # Resample sites with replacement (no data copying!)
            # Get number of SNPs from filtered_genotypes when using tf.data pipeline
            if original_filtered_genotypes is not None:
                n_snps = original_filtered_genotypes.shape[0]
            elif self.traingen is not None:
                n_snps = self.traingen.shape[1]
            else:
                raise ValueError(
                    "Unable to determine number of SNPs for bootstrap resampling"
                )

            site_order = np.random.choice(n_snps, n_snps, replace=True)

            # Clear existing model and weights
            self.model = None
            self.sample_weights = None

            # Restore filtered genotypes and index set for tf.data pipeline
            if original_filtered_genotypes is not None:
                self.filtered_genotypes = original_filtered_genotypes
                self.index_set = original_index_set

            # Train with bootstrapped sites using site_order parameter
            # When using tf.data pipeline, pass original genotypes
            self.train(
                genotypes=genotypes if original_filtered_genotypes is not None else None,
                samples=samples,
                boot=boot,
                train_gen=self.traingen,  # Use original data
                test_gen=self.testgen,
                pred_gen=self.predgen,
                train_locs=original_trainlocs,
                test_locs=original_testlocs,
                site_order=site_order,  # Pass site order for bootstrap resampling
            )

            # Get predictions using tf.data approach
            preds = self.predict(
                boot=boot,
                verbose=False,
                genotypes=genotypes,  # Pass full genotypes for tf.data
                samples=samples,
                indices=self.pred_indices if hasattr(self, "pred_indices") else None,
                site_order=site_order,  # Pass same site order for consistent resampling
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

        # Restore original bandwidth setting if we changed it
        if bandwidth_calculated:
            if original_bandwidth is None:
                # Remove the key if it wasn't there originally
                self.config.get("weight_samples", {}).pop("bandwidth", None)
            else:
                self.config["weight_samples"]["bandwidth"] = original_bandwidth

        if return_df:
            # Concatenate all predictions and add sampleIDs
            all_predictions = pd.concat([preds[["sampleID"]], *pred_dfs], axis=1)

            if save_full_pred_matrix:
                all_predictions.to_csv(
                    f"{self.config['out']}_bootstrap_predlocs.csv", index=False
                )
            return all_predictions

        return None
