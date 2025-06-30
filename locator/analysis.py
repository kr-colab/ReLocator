"""Analysis functionality for locator"""

import numpy as np
import pandas as pd
import copy
from tqdm import tqdm
from tensorflow import keras
import zarr

from .data import filter_snps_legacy as filter_snps, normalize_locs, IndexSet, make_tf_dataset


class AnalysisMixin:
    """Mixin class providing analysis functionality for Locator."""
    
    def run_windows(
        self,
        genotypes,
        samples,
        window_start=0,
        window_size=5e5,
        window_stop=None,
        return_df=False,
        save_full_pred_matrix=True,
        na_action=None,
    ):
        """Run windowed prediction analysis.

        Args:
            genotypes: GenotypeArray containing genetic data
            samples: Array of sample IDs
            window_start: Start position for windows (default: 0)
            window_size: Size of windows in base pairs (default: 500kb)
            window_stop: Stop position for windows (default: None)
            return_df: Whether to return DataFrame with all predictions
            save_full_pred_matrix: Whether to save full prediction matrix to disk
            na_action: How to handle NA samples ('separate', 'exclude', 'fail'). 
                If None, uses self.na_action
            
        Returns:
            pandas.DataFrame or None: If return_df=True, returns DataFrame with predictions
                for each window, otherwise None
                
        Notes:
            - With na_action='separate': Trains on samples with known locations,
              can predict on samples with NA locations
            - With na_action='exclude': Only uses samples with known locations
            - With na_action='fail': Raises error if any NA samples found
            
        Warning:
            Window analysis currently treats all SNP positions as continuous along a single
            coordinate axis. If your data contains multiple chromosomes, windows may span
            across chromosome boundaries. For accurate per-chromosome analysis, consider
            filtering your data to single chromosomes before running window analysis.
        """
        # Store samples
        self.samples = samples
        
        # Use instance default if na_action not specified
        if na_action is None:
            na_action = self.na_action
            
        # Get sample status
        status = self.get_sample_status(samples)
        
        # Report status
        print(f"Window analysis: {status['n_known']} samples with coordinates, {status['n_na']} without")
        if status['n_na'] > 0:
            print(f"NA handling mode: {na_action}")
        
        # Apply NA action
        if na_action == 'fail' and status['n_na'] > 0:
            raise ValueError(
                f"Found {status['n_na']} samples without coordinates. "
                f"Set na_action='separate' or 'exclude' to proceed."
            )

        # Get positions if not already stored
        if not hasattr(self, "positions") or self.positions is None:
            if hasattr(self, "_genotype_df"):
                # Use positions from DataFrame columns
                self.positions = np.array(self._genotype_df.columns, dtype=int)
            elif self.config.get("zarr"):
                # Get positions from zarr file
                callset = zarr.open_group(self.config["zarr"], mode="r")
                self.positions = callset["variants/POS"][:]
            elif self.config.get("vcf"):
                # Re-read VCF to get positions and chromosomes
                print("Loading SNP positions from VCF...")
                import allel
                vcf = allel.read_vcf(self.config["vcf"], fields=['POS', 'CHROM'])
                if vcf is not None and "variants/POS" in vcf:
                    self.positions = vcf["variants/POS"]
                    if "variants/CHROM" in vcf:
                        self.chromosomes = vcf["variants/CHROM"]
                    print(f"Loaded {len(self.positions)} SNP positions")
                else:
                    raise ValueError(f"Could not load positions from VCF: {self.config['vcf']}")
            else:
                raise ValueError(
                    "SNP positions required for windowed analysis. Use VCF, zarr input or "
                    "genotype DataFrame with position-labeled columns."
                )
        
        # Ensure positions were found
        if not hasattr(self, "positions") or self.positions is None:
            raise ValueError(
                "SNP positions required for windowed analysis. Use zarr input or "
                "genotype DataFrame with position-labeled columns."
            )

        if window_stop is None:
            window_stop = max(self.positions)

        # Check if we have chromosome information and warn if multiple chromosomes
        if hasattr(self, "chromosomes") and self.chromosomes is not None:
            unique_chroms = np.unique(self.chromosomes)
            if len(unique_chroms) > 1:
                import warnings
                warnings.warn(
                    f"Multiple chromosomes detected ({len(unique_chroms)}). "
                    f"Window analysis currently treats all positions as continuous, "
                    f"which may create windows spanning multiple chromosomes. "
                    f"Consider filtering to single chromosomes for more accurate results."
                )

        windows = range(int(window_start), int(window_stop), int(window_size))

        # Initial training to set up model and data
        first_window = (self.positions >= int(window_start)) & (
            self.positions < int(window_start + window_size)
        )
        if sum(first_window) > 0:
            window_genos = genotypes[first_window, :, :]
            self.train(genotypes=window_genos, samples=samples, na_action=na_action)

        # Create lists to store predictions
        pred_dfs = []

        print("starting window analysis")
        for start in tqdm(windows):
            stop = start + int(window_size)
            in_window = (self.positions >= start) & (self.positions < stop)

            if sum(in_window) > 0:
                # Get genotypes for this window
                window_genos = genotypes[in_window, :, :]

                # Clear existing model and weights
                self.model = None
                self.sample_weights = None

                # Train on window data
                self.train(genotypes=window_genos, samples=samples, na_action=na_action)

                # Get predictions using self.predgen which is already properly formatted
                preds = self.predict(
                    return_df=True, save_preds_to_disk=not save_full_pred_matrix
                )

                if return_df:
                    # Rename columns to include window start
                    window_preds = preds[["sampleID", "x", "y"]].copy()
                    window_preds.columns = ["sampleID", f"x_win{start}", f"y_win{start}"]
                    pred_dfs.append(window_preds)

                # Clear keras session
                keras.backend.clear_session()

        if return_df:
            # Merge all predictions from different windows
            if not pred_dfs:
                print("Warning: No windows contained SNPs. No predictions generated.")
                return None
                
            # Start with the first window's predictions
            all_predictions = pred_dfs[0]
            
            # Merge subsequent windows
            for pred_df in pred_dfs[1:]:
                all_predictions = all_predictions.merge(pred_df, on='sampleID', how='outer')

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

        Returns:
            pandas.DataFrame or None: If return_df=True, returns DataFrame containing
                all predictions, with columns named 'x_0', 'y_0', 'x_1', 'y_1', etc.
                for each jacknife replicate. Row index contains sample IDs.
                
        Notes:
            - With na_action='separate': Trains on samples with known locations,
              can predict on samples with NA locations
            - With na_action='exclude': Only uses samples with known locations
            - With na_action='fail': Raises error if any NA samples found
        """
        # Store samples
        self.samples = samples
        
        # Use instance default if na_action not specified
        if na_action is None:
            na_action = self.na_action
            
        # Get sample status
        status = self.get_sample_status(samples)
        
        # Report status
        print(f"Jacknife analysis: {status['n_known']} samples with coordinates, {status['n_na']} without")
        if status['n_na'] > 0:
            print(f"NA handling mode: {na_action}")
        
        # Apply NA action
        if na_action == 'fail' and status['n_na'] > 0:
            raise ValueError(
                f"Found {status['n_na']} samples without coordinates. "
                f"Set na_action='separate' or 'exclude' to proceed."
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
        original_filtered_genotypes = self.filtered_genotypes if hasattr(self, 'filtered_genotypes') else None
        original_index_set = self.index_set if hasattr(self, 'index_set') else None
        
        # Store original locations and model
        original_trainlocs = self.trainlocs if hasattr(self, 'trainlocs') else None
        original_testlocs = self.testlocs if hasattr(self, 'testlocs') else None
        
        # Calculate number of jacknife replicates
        n_jack = int(np.ceil(1.0 / prop))
        print(f"starting jacknife resampling ({n_jack} replicates)")
        
        for boot in tqdm(range(n_jack)):
            # Generate indices of sites to keep (jackknife drops a subset)
            if original_filtered_genotypes is not None:
                n_sites = original_filtered_genotypes.shape[0]
            else:
                raise ValueError("Jacknife requires filtered_genotypes from initial training")
            
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
                indices=self.pred_indices if hasattr(self, 'pred_indices') else None,
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

    def run_bootstraps(
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
            
        Returns:
            pandas.DataFrame or None: If return_df=True, returns DataFrame with predictions
                for each bootstrap, otherwise None
                
        Notes:
            - With na_action='separate': Trains on samples with known locations,
              can predict on samples with NA locations
            - With na_action='exclude': Only uses samples with known locations
            - With na_action='fail': Raises error if any NA samples found
        """
        # Store samples
        self.samples = samples
        
        # Use instance default if na_action not specified
        if na_action is None:
            na_action = self.na_action
            
        # Get sample status
        status = self.get_sample_status(samples)
        
        # Report status
        print(f"Bootstrap analysis: {status['n_known']} samples with coordinates, {status['n_na']} without")
        if status['n_na'] > 0:
            print(f"NA handling mode: {na_action}")
        
        # Apply NA action
        if na_action == 'fail' and status['n_na'] > 0:
            raise ValueError(
                f"Found {status['n_na']} samples without coordinates. "
                f"Set na_action='separate' or 'exclude' to proceed."
            )

        # Set bootstrap flag in config
        self.config["bootstrap"] = True
        self.config["nboots"] = n_bootstraps

        # Initial training to set up model and data - pass na_action
        self.train(genotypes=genotypes, samples=samples, na_action=na_action)

        # Store original locations and filtered genotypes for reuse
        original_trainlocs = self.trainlocs
        original_testlocs = self.testlocs
        original_filtered_genotypes = self.filtered_genotypes if hasattr(self, 'filtered_genotypes') else None
        
        # Handle prediction indices based on whether we're using tf.data pipeline
        if hasattr(self, 'pred_indices') and self.pred_indices is not None:
            n_pred = len(self.pred_indices)
        elif self.predgen is not None:
            n_pred = self.predgen.shape[0]
        else:
            n_pred = 0
            
        original_normalized_locs = np.vstack([
            self.trainlocs,
            self.testlocs,
            np.full((n_pred, 2), np.nan) if n_pred > 0 else np.empty((0, 2))
        ])
        original_index_set = self.index_set if hasattr(self, 'index_set') else None
        
        # Pre-calculate KDE bandwidth if needed
        original_bandwidth = None
        bandwidth_calculated = False
        
        if (self.config.get("weight_samples", {}).get("enabled", False) and
            self.config.get("weight_samples", {}).get("method") == "KD"):
            
            existing_bandwidth = self.config.get("weight_samples", {}).get("bandwidth")
            
            if existing_bandwidth is None and len(original_trainlocs) > 1:
                print("Pre-calculating optimal KDE bandwidth for bootstrap analysis...")
                
                from .sample_weights import get_global_bandwidth_optimizer
                optimizer = get_global_bandwidth_optimizer()
                
                optimal_bandwidth = optimizer.get_bandwidth(
                    original_trainlocs,
                    cache_key=f"bootstrap_n{len(original_trainlocs)}",
                    n_bandwidths=self.config.get("weight_samples", {}).get("n_bandwidths", 100),
                    verbose=True
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
                raise ValueError("Unable to determine number of SNPs for bootstrap resampling")
                
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
                indices=self.pred_indices if hasattr(self, 'pred_indices') else None,
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

    def run_holdouts(
        self,
        genotypes,
        samples,
        k=10,
        n_reps=10,
        holdout_indices=None,
        return_df=False,
        save_full_pred_matrix=True,
        na_action=None,
    ):
        """Run multiple holdout replicates for cross-validation.

        Args:
            genotypes: Array of genotype data
            samples: Sample IDs corresponding to genotypes
            k: Number of samples to hold out in each replicate
            n_reps: Number of holdout replicates to run
            holdout_indices: Optional list of lists, each containing indices to hold out
            return_df: Whether to return DataFrame with all predictions
            save_full_pred_matrix: Whether to save full prediction matrix to disk
            na_action: How to handle NA samples ('separate', 'exclude', 'fail'). 
                If None, uses self.na_action
        Returns:
            pandas.DataFrame or None: If return_df=True, returns DataFrame with predictions
                for each holdout replicate containing columns:
                - sampleID: Sample identifier
                - x_pred: Predicted longitude
                - y_pred: Predicted latitude
                - rep: Replicate number (0 to n_reps-1)
                
                Note: True locations are not included. Merge with sample metadata to calculate errors.
                
        Notes:
            - With na_action='separate': Currently behaves like 'exclude' (holdouts 
              must have known locations). Future versions may support predicting NA samples.
            - With na_action='exclude': Only uses samples with known locations (current behavior)
            - With na_action='fail': Raises error if any NA samples found
        """
        # Store samples
        self.samples = samples
        
        # Use instance default if na_action not specified
        if na_action is None:
            na_action = self.na_action
            
        # Get sample status
        status = self.get_sample_status(samples)
        
        # Report status
        print(f"Holdout analysis: {status['n_known']} samples with coordinates, {status['n_na']} without")
        if status['n_na'] > 0:
            print(f"NA handling mode: {na_action}")
            if na_action == 'separate':
                print("Note: Holdout analysis requires known locations; 'separate' behaves like 'exclude'")
        
        # Apply NA action
        if na_action == 'fail' and status['n_na'] > 0:
            raise ValueError(
                f"Found {status['n_na']} samples without coordinates. "
                f"Set na_action='separate' or 'exclude' to proceed."
            )

        # Create lists to store predictions
        pred_dfs = []

        # Get sample data and locations
        if hasattr(self, "_sample_data_df"):
            sample_data, locs = self.sort_samples(samples)
        else:
            sample_data_path = self.config.get("sample_data")
            if not sample_data_path:
                raise ValueError("sample_data file path must be provided in config")
            sample_data, locs = self.sort_samples(samples, sample_data_path)

        # Get indices of samples with known locations
        known_idx = np.argwhere(~np.isnan(locs[:, 0]))
        known_idx = np.array([x[0] for x in known_idx])

        if k >= len(known_idx):
            raise ValueError(
                f"k ({k}) must be less than number of samples with known locations ({len(known_idx)})"
            )

        # Pre-calculate KDE bandwidth if needed
        original_bandwidth = None
        bandwidth_calculated = False
        
        if (self.config.get("weight_samples", {}).get("enabled", False) and
            self.config.get("weight_samples", {}).get("method") == "KD"):
            
            existing_bandwidth = self.config.get("weight_samples", {}).get("bandwidth")
            
            if existing_bandwidth is None:
                # Get all samples with coordinates for bandwidth calculation
                all_train_locs = locs[known_idx]
                
                if len(all_train_locs) > 1:
                    print("Pre-calculating optimal KDE bandwidth for holdout analysis...")
                    
                    from .sample_weights import get_global_bandwidth_optimizer
                    optimizer = get_global_bandwidth_optimizer()
                    
                    optimal_bandwidth = optimizer.get_bandwidth(
                        all_train_locs,
                        cache_key=f"holdouts_k{k}_n{len(all_train_locs)}",
                        n_bandwidths=self.config.get("weight_samples", {}).get("n_bandwidths", 100),
                        verbose=True
                    )
                    
                    # Temporarily set in config
                    self.config["weight_samples"]["bandwidth"] = optimal_bandwidth
                    bandwidth_calculated = True
                    
                    print(f"Using bandwidth: {optimal_bandwidth:.3f}")

        print(f"Running {n_reps} holdout replicates")

        for rep in tqdm(range(n_reps)):
            # Clear existing model and weights
            self.model = None
            self.sample_weights = None

            # Select holdout indices for this replicate
            if holdout_indices is not None and rep < len(holdout_indices):
                rep_holdout_idx = holdout_indices[rep]
            else:
                # Random selection
                rep_holdout_idx = np.random.choice(known_idx, k, replace=False)

            # Train with holdout
            self.train_holdout(
                genotypes=genotypes,
                samples=samples,
                holdout_indices=rep_holdout_idx,
            )

            # Get predictions for holdout samples
            preds = self.predict_holdout(
                verbose=False,
                return_df=True,
                save_preds_to_disk=not save_full_pred_matrix,
                plot_summary=False,  # Don't plot during analysis runs
            )

            if return_df:
                # Rename columns to include replicate number
                holdout_preds = preds[["x_pred", "y_pred"]].copy()
                holdout_preds.columns = [f"x_rep{rep}", f"y_rep{rep}"]
                holdout_preds["sampleID"] = preds["sampleID"]
                pred_dfs.append(holdout_preds)

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
            # Merge all predictions
            all_predictions = pred_dfs[0]
            for df in pred_dfs[1:]:
                all_predictions = pd.merge(
                    all_predictions, df, on="sampleID", how="outer"
                )
            if save_full_pred_matrix:
                all_predictions.to_csv(
                    f"{self.config['out']}_holdouts_predlocs.csv", index=False
                )
            return all_predictions
        return None

    def run_jacknife_holdouts(
        self,
        genotypes,
        samples,
        k=10,
        prop=0.05,
        n_boots=50,
        holdout_indices=None,
        return_df=False,
        save_full_pred_matrix=True,
        na_action=None,
    ):
        """Run jacknife analysis on holdout samples.

        Args:
            genotypes: Array of genotype data
            samples: Sample IDs corresponding to genotypes
            k: Number of samples to hold out
            prop: Proportion of SNPs to drop in each jacknife replicate
            n_boots: Number of jacknife replicates
            holdout_indices: Optional specific indices to hold out
            return_df: Whether to return DataFrame with all predictions
            save_full_pred_matrix: Whether to save full prediction matrix to disk
            na_action: How to handle NA samples ('separate', 'exclude', 'fail'). 
                If None, uses self.na_action
            
        Returns:
            pandas.DataFrame or None: If return_df=True, returns DataFrame with predictions
                for each jacknife replicate containing columns:
                - sampleID: Sample identifier
                - x_pred: Predicted longitude  
                - y_pred: Predicted latitude
                - boot: Jacknife replicate number (0 to n_boots-1)
                
                Note: True locations are not included. Merge with sample metadata to calculate errors.
                
        Notes:
            - With na_action='separate': Currently behaves like 'exclude' (holdouts 
              must have known locations). Future versions may support predicting NA samples.
            - With na_action='exclude': Only uses samples with known locations (current behavior)
            - With na_action='fail': Raises error if any NA samples found
        """
        # Store samples
        self.samples = samples
        
        # Use instance default if na_action not specified
        if na_action is None:
            na_action = self.na_action
            
        # Get sample status
        status = self.get_sample_status(samples)
        
        # Report status
        print(f"Jacknife holdout analysis: {status['n_known']} samples with coordinates, {status['n_na']} without")
        if status['n_na'] > 0:
            print(f"NA handling mode: {na_action}")
            if na_action == 'separate':
                print("Note: Holdout analysis requires known locations; 'separate' behaves like 'exclude'")
        
        # Apply NA action
        if na_action == 'fail' and status['n_na'] > 0:
            raise ValueError(
                f"Found {status['n_na']} samples without coordinates. "
                f"Set na_action='separate' or 'exclude' to proceed."
            )

        # Set jacknife flag
        self.config["jacknife"] = True
        self.config["nboots"] = n_boots

        # Train with holdout
        self.train_holdout(
            genotypes=genotypes,
            samples=samples,
            k=k,
            holdout_indices=holdout_indices,
        )

        # Pre-calculate KDE bandwidth if needed
        original_bandwidth = None
        bandwidth_calculated = False
        
        if (self.config.get("weight_samples", {}).get("enabled", False) and
            self.config.get("weight_samples", {}).get("method") == "KD"):
            
            existing_bandwidth = self.config.get("weight_samples", {}).get("bandwidth")
            
            if existing_bandwidth is None and hasattr(self, 'trainlocs') and len(self.trainlocs) > 1:
                print("Pre-calculating optimal KDE bandwidth for jacknife holdout analysis...")
                
                from .sample_weights import get_global_bandwidth_optimizer
                optimizer = get_global_bandwidth_optimizer()
                
                optimal_bandwidth = optimizer.get_bandwidth(
                    self.trainlocs,
                    cache_key=f"jacknife_holdouts_n{len(self.trainlocs)}",
                    n_bandwidths=self.config.get("weight_samples", {}).get("n_bandwidths", 100),
                    verbose=True
                )
                
                # Temporarily set in config
                self.config["weight_samples"]["bandwidth"] = optimal_bandwidth
                bandwidth_calculated = True
                
                print(f"Using bandwidth: {optimal_bandwidth:.3f}")

        # Calculate allele frequencies
        print("Calculating allele frequencies...")
        ac = genotypes.to_allele_counts()[:, :, 1]
        af = np.array([np.sum(ac[i, :]) / (ac.shape[1] * 2) for i in range(ac.shape[0])])

        # Create lists to store predictions
        pred_dfs = []

        print(f"Running {n_boots} jacknife replicates for holdout samples")

        for boot in tqdm(range(n_boots)):
            # Create modified genotypes for holdout samples
            holdout_gen_modified = copy.deepcopy(self.holdout_gen)

            # Randomly select sites to remove
            sites_to_remove = np.random.choice(
                holdout_gen_modified.shape[1],
                int(holdout_gen_modified.shape[1] * prop),
                replace=False,
            )

            # Replace with random draws from allele frequency
            for i in sites_to_remove:
                holdout_gen_modified[:, i] = np.random.binomial(
                    2, af[i], size=holdout_gen_modified.shape[0]
                )

            # Get predictions
            predictions = self.model.predict(holdout_gen_modified, verbose=0)

            # Denormalize
            predictions[:, 0] = predictions[:, 0] * self.sdlong + self.meanlong
            predictions[:, 1] = predictions[:, 1] * self.sdlat + self.meanlat

            # Create dataframe
            boot_df = pd.DataFrame(
                {
                    f"x_boot{boot}": predictions[:, 0],
                    f"y_boot{boot}": predictions[:, 1],
                    "sampleID": self.samples[self.holdout_idx],
                }
            )

            pred_dfs.append(boot_df)

        # Restore original bandwidth setting if we changed it
        if bandwidth_calculated:
            if original_bandwidth is None:
                # Remove the key if it wasn't there originally
                self.config.get("weight_samples", {}).pop("bandwidth", None)
            else:
                self.config["weight_samples"]["bandwidth"] = original_bandwidth

        if return_df:
            # Merge all predictions
            all_predictions = pred_dfs[0]
            for df in pred_dfs[1:]:
                all_predictions = pd.merge(all_predictions, df, on="sampleID")

            if save_full_pred_matrix:
                all_predictions.to_csv(
                    f"{self.config['out']}_jacknife_holdouts_predlocs.csv", index=False
                )
            return all_predictions

        return None

    def run_windows_holdouts(
        self,
        genotypes,
        samples,
        k=10,
        window_start=0,
        window_size=5e5,
        window_stop=None,
        holdout_indices=None,
        return_df=False,
        save_full_pred_matrix=True,
        na_action=None,
    ):
        """Run windowed analysis on holdout samples.

        Args:
            genotypes: Array of genotype data
            samples: Sample IDs corresponding to genotypes
            k: Number of samples to hold out
            window_start: Start position for windows
            window_size: Size of windows in base pairs
            window_stop: Stop position for windows
            holdout_indices: Optional specific indices to hold out
            return_df: Whether to return DataFrame with all predictions
            save_full_pred_matrix: Whether to save full prediction matrix to disk
            na_action: How to handle NA samples ('separate', 'exclude', 'fail'). 
                If None, uses self.na_action
            
        Returns:
            pandas.DataFrame or None: If return_df=True, returns DataFrame with predictions
                for each window, otherwise None
                
        Notes:
            - With na_action='separate': Currently behaves like 'exclude' (holdouts 
              must have known locations). Future versions may support predicting NA samples.
            - With na_action='exclude': Only uses samples with known locations (current behavior)
            - With na_action='fail': Raises error if any NA samples found
            
        Warning:
            Window analysis currently treats all SNP positions as continuous along a single
            coordinate axis. If your data contains multiple chromosomes, windows may span
            across chromosome boundaries. For accurate per-chromosome analysis, consider
            filtering your data to single chromosomes before running window analysis.
        """
        # Store samples and genotypes for efficient access
        self.samples = samples
        self.genotypes = genotypes
        
        # Use instance default if na_action not specified
        if na_action is None:
            na_action = self.na_action
            
        # Get sample status and create NA mask
        status = self.get_sample_status(samples)
        na_mask = None
        if status['n_na'] > 0:
            # Create boolean mask for NA samples
            if isinstance(samples, pd.DataFrame):
                na_mask = samples['x'].isna() | samples['y'].isna()
            else:
                # Use stored sample data or load from config
                if hasattr(self, "_sample_data_df"):
                    sample_data = self._sample_data_df
                else:
                    sample_data_path = self.config.get("sample_data")
                    if sample_data_path:
                        sample_data = pd.read_csv(sample_data_path, sep="\t")
                    else:
                        raise ValueError("No sample data available")
                
                merged = pd.DataFrame({"sampleID": samples})
                merged = merged.merge(sample_data, on="sampleID", how="left")
                na_mask = merged['x'].isna() | merged['y'].isna()
            na_mask = na_mask.values
        
        # Report status
        print(f"Windows holdout analysis: {status['n_known']} samples with coordinates, {status['n_na']} without")
        if status['n_na'] > 0:
            print(f"NA handling mode: {na_action}")
            if na_action == 'separate':
                print("Note: Holdout analysis requires known locations; 'separate' behaves like 'exclude'")
        
        # Apply NA action
        if na_action == 'fail' and status['n_na'] > 0:
            raise ValueError(
                f"Found {status['n_na']} samples without coordinates. "
                f"Set na_action='separate' or 'exclude' to proceed."
            )

        # Get positions and create holdout IndexSet
        if not hasattr(self, "positions") or self.positions is None:
            if hasattr(self, "_genotype_df"):
                self.positions = np.array(self._genotype_df.columns, dtype=int)
            elif self.config.get("zarr"):
                callset = zarr.open_group(self.config["zarr"], mode="r")
                self.positions = callset["variants/POS"][:]
            elif self.config.get("vcf"):
                # Re-read VCF to get positions and chromosomes
                print("Loading SNP positions from VCF...")
                import allel
                vcf = allel.read_vcf(self.config["vcf"], fields=['POS', 'CHROM'])
                if vcf is not None and "variants/POS" in vcf:
                    self.positions = vcf["variants/POS"]
                    if "variants/CHROM" in vcf:
                        self.chromosomes = vcf["variants/CHROM"]
                    print(f"Loaded {len(self.positions)} SNP positions")
                else:
                    raise ValueError(f"Could not load positions from VCF: {self.config['vcf']}")
            else:
                raise ValueError(
                    "SNP positions required for windowed analysis. Use VCF, zarr input or "
                    "genotype DataFrame with position-labeled columns."
                )
        
        # Create IndexSet for holdout splitting
        n_samples = len(samples)
        if holdout_indices is not None:
            # Use provided holdout indices
            holdout_idx = np.array(holdout_indices)
            train_idx = np.setdiff1d(np.arange(n_samples), holdout_idx)
            
            # Apply NA mask if needed
            if na_mask is not None and (na_action == 'exclude' or na_action == 'separate'):
                # Only keep samples with known coordinates
                valid_mask = ~na_mask
                holdout_idx = holdout_idx[valid_mask[holdout_idx]]
                train_idx = train_idx[valid_mask[train_idx]]
            
            index_set = IndexSet(
                indices={'train': train_idx, 'test': holdout_idx},
                total_samples=n_samples,
                na_mask=na_mask
            )
        else:
            # Random holdout selection using IndexSet
            index_set = IndexSet.random_split(
                n=n_samples,
                splits={'train': 1.0 - k/n_samples, 'test': k/n_samples},
                seed=self.config.get('seed', 42),
                na_mask=na_mask,
                na_action=na_action if na_action != 'separate' else 'exclude'
            )

        if window_stop is None:
            window_stop = max(self.positions)

        # Check if we have chromosome information and warn if multiple chromosomes
        if hasattr(self, "chromosomes") and self.chromosomes is not None:
            unique_chroms = np.unique(self.chromosomes)
            if len(unique_chroms) > 1:
                import warnings
                warnings.warn(
                    f"Multiple chromosomes detected ({len(unique_chroms)}). "
                    f"Window analysis currently treats all positions as continuous, "
                    f"which may create windows spanning multiple chromosomes. "
                    f"Consider filtering to single chromosomes for more accurate results."
                )

        windows = range(int(window_start), int(window_stop), int(window_size))

        # Create lists to store predictions
        pred_dfs = []

        # Pre-calculate KDE bandwidth if needed
        original_bandwidth = None
        bandwidth_calculated = False
        
        if (self.config.get("weight_samples", {}).get("enabled", False) and
            self.config.get("weight_samples", {}).get("method") == "KD"):
            
            existing_bandwidth = self.config.get("weight_samples", {}).get("bandwidth")
            
            if existing_bandwidth is None:
                # Get sample data and locations
                if hasattr(self, "_sample_data_df"):
                    sample_data, locs = self.sort_samples(samples)
                else:
                    sample_data_path = self.config.get("sample_data")
                    if not sample_data_path:
                        raise ValueError("sample_data file path must be provided in config")
                    sample_data, locs = self.sort_samples(samples, sample_data_path)
                
                # Get training locations (exclude holdout samples)
                train_mask = np.ones(len(samples), dtype=bool)
                train_mask[index_set.test] = False
                train_mask = train_mask & ~np.isnan(locs[:, 0])
                train_locs = locs[train_mask]
                
                if len(train_locs) > 1:
                    print("Pre-calculating optimal KDE bandwidth for windows holdout analysis...")
                    
                    from .sample_weights import get_global_bandwidth_optimizer
                    optimizer = get_global_bandwidth_optimizer()
                    
                    optimal_bandwidth = optimizer.get_bandwidth(
                        train_locs,
                        cache_key=f"windows_holdouts_n{len(train_locs)}",
                        n_bandwidths=self.config.get("weight_samples", {}).get("n_bandwidths", 100),
                        verbose=True
                    )
                    
                    # Temporarily set in config
                    self.config["weight_samples"]["bandwidth"] = optimal_bandwidth
                    bandwidth_calculated = True
                    
                    print(f"Using bandwidth: {optimal_bandwidth:.3f}")

        print(f"Running windowed analysis for holdout samples")
        
        # Store the full IndexSet for use across windows
        self.index_set = index_set
        
        # Pre-normalize locations for efficiency
        if hasattr(self, "_sample_data_df"):
            _, locs = self.sort_samples(samples)
        else:
            sample_data_path = self.config.get("sample_data")
            if not sample_data_path:
                raise ValueError("sample_data file path must be provided in config")
            _, locs = self.sort_samples(samples, sample_data_path)
        
        # Normalize locations once
        self.meanlong, self.sdlong, self.meanlat, self.sdlat, self.unnormedlocs, normalized_locs = (
            normalize_locs(locs)
        )

        for start in tqdm(windows):
            stop = start + int(window_size)
            snp_mask = (self.positions >= start) & (self.positions < stop)
            snp_indices = np.where(snp_mask)[0]

            if len(snp_indices) > 0:
                # Clear existing model and weights
                self.model = None
                self.sample_weights = None

                # Use efficient window training method
                self.train_window(
                    genotypes=genotypes,
                    samples=samples,
                    window_snp_indices=snp_indices,
                    index_set=index_set,
                    normalized_locs=normalized_locs,
                )

                # Get predictions for holdout samples
                preds = self.predict_holdout(
                    verbose=False,
                    return_df=True,
                    save_preds_to_disk=not save_full_pred_matrix,
                    plot_summary=False,  # Don't plot during analysis runs
                )

                if return_df:
                    # Rename columns to include window position
                    window_preds = preds[["x_pred", "y_pred"]].copy()
                    window_preds.columns = [f"x_pos{start}", f"y_pos{start}"]
                    window_preds["sampleID"] = preds["sampleID"]
                    pred_dfs.append(window_preds)

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
            # Check if any windows had predictions
            if not pred_dfs:
                print("Warning: No windows contained SNPs. No predictions generated.")
                return None
                
            # Merge all predictions
            all_predictions = pred_dfs[0]
            for df in pred_dfs[1:]:
                all_predictions = pd.merge(all_predictions, df, on="sampleID")

            if save_full_pred_matrix:
                all_predictions.to_csv(
                    f"{self.config['out']}_windows_holdouts_predlocs.csv", index=False
                )
            return all_predictions

        return None

    def run_leave_one_out(
        self,
        genotypes,
        samples,
        return_df=True,
        save_full_pred_matrix=True,
        na_action=None,
    ):
        """
        Perform leave-one-out cross-validation: for each sample with a known location,
        train without it and predict its location.

        This is a convenience wrapper around run_k_fold_holdouts with k equal to the
        number of samples with known locations.

        Args:
            genotypes: Array of genotype data
            samples: Sample IDs corresponding to genotypes
            return_df: Whether to return DataFrame with all predictions
            save_full_pred_matrix: Whether to save full prediction matrix to disk
            na_action: How to handle NA samples ('separate', 'exclude', 'fail'). 
                If None, uses self.na_action

        Returns:
            pandas.DataFrame or None: DataFrame with predictions for each left-out sample
        """
        # Get sample status to determine k
        status = self.get_sample_status(samples)
        n_known = status['n_known']
        
        if n_known == 0:
            raise ValueError("No samples with known coordinates for leave-one-out CV")
        
        print(f"Running leave-one-out cross-validation for {n_known} samples")
        
        # Run k-fold with k equal to number of known samples
        # This will create folds with exactly 1 sample each
        result = self.run_k_fold_holdouts(
            genotypes=genotypes,
            samples=samples,
            k=n_known,
            return_df=return_df,
            save_full_pred_matrix=False,  # We'll save with our own name
            verbose=False,  # We already printed our message
            na_action=na_action
        )
        
        # Save with leave-one-out specific filename if requested
        if result is not None and save_full_pred_matrix:
            result.to_csv(
                f"{self.config['out']}_leave_one_out_predlocs.csv", index=False
            )
        
        return result

    def run_k_fold_holdouts(
        self,
        genotypes,
        samples,
        k=10,
        return_df=False,
        save_full_pred_matrix=True,
        verbose=True,
        na_action=None,
    ):
        """
        Run true k-fold cross-validation with nonoverlapping holdout sets.

        Args:
            genotypes: Array of genotype data
            samples: Sample IDs corresponding to genotypes
            k: Number of folds (holdout sets)
            return_df: Whether to return DataFrame with all predictions
            save_full_pred_matrix: Whether to save full prediction matrix to disk
            verbose: Whether to show training progress and intermediate output
            na_action: How to handle NA samples ('separate', 'exclude', 'fail'). 
                If None, uses self.na_action
        Returns:
            pandas.DataFrame or None: If return_df=True, returns DataFrame with one prediction 
                per held-out sample containing columns:
                - sampleID: Sample identifier
                - x_pred: Predicted longitude
                - y_pred: Predicted latitude
                
                Note: True locations are not included. To calculate prediction errors, merge
                the returned DataFrame with your sample metadata using the sampleID column.
            
        Notes:
            - With na_action='separate': Currently behaves like 'exclude' (k-fold requires 
              known locations). Future versions may support predicting NA samples.
            - With na_action='exclude': Only uses samples with known locations (current behavior)
            - With na_action='fail': Raises error if any NA samples found
            
        Example:
            >>> # Run k-fold cross-validation
            >>> predictions = locator.run_k_fold_holdouts(genotypes, samples, k=10, return_df=True)
            >>> 
            >>> # Merge with true locations to calculate errors
            >>> sample_data = pd.read_csv('samples.tsv', sep='\t')
            >>> merged = predictions.merge(sample_data[['sampleID', 'x', 'y']], on='sampleID')
            >>> merged['error_km'] = np.sqrt(
            ...     (merged['x'] - merged['x_pred'])**2 + 
            ...     (merged['y'] - merged['y_pred'])**2
            ... ) * 111.32  # Convert degrees to km
        """
        self.samples = samples
        
        # Use instance default if na_action not specified
        if na_action is None:
            na_action = self.na_action
            
        # Get sample status
        status = self.get_sample_status(samples)
        
        # Report status
        if verbose:
            print(f"K-fold CV: {status['n_known']} samples with coordinates, {status['n_na']} without")
            if status['n_na'] > 0:
                print(f"NA handling mode: {na_action}")
                if na_action == 'separate':
                    print("Note: K-fold CV requires known locations; 'separate' behaves like 'exclude'")
        
        # Apply NA action
        if na_action == 'fail' and status['n_na'] > 0:
            raise ValueError(
                f"Found {status['n_na']} samples without coordinates. "
                f"Set na_action='separate' or 'exclude' to proceed."
            )
        
        pred_rows = []

        # Get sample data and locations
        if hasattr(self, "_sample_data_df"):
            sample_data, locs = self.sort_samples(samples)
        else:
            sample_data_path = self.config.get("sample_data")
            if not sample_data_path:
                raise ValueError("sample_data file path must be provided in config")
            sample_data, locs = self.sort_samples(samples, sample_data_path)

        # Create NA mask
        na_mask = np.isnan(locs[:, 0])
        n_total_samples = len(locs)
        n_samples_with_coords = np.sum(~na_mask)
        
        if k > n_samples_with_coords:
            raise ValueError(
                f"k ({k}) must be less than or equal to number of samples with known locations ({n_samples_with_coords})"
            )

        # Create list to store IndexSets for each fold
        # Use a fixed base seed for reproducibility, but different for each run
        base_seed = np.random.randint(0, 2**31 - 1)
        fold_index_sets = []
        for fold_idx in range(k):
            index_set = IndexSet.from_k_fold(
                n=n_total_samples,
                k=k,
                fold=fold_idx,
                seed=base_seed,  # All folds use same shuffle for consistent k-fold split
                na_mask=na_mask
            )
            fold_index_sets.append(index_set)

        # Store original keras_verbose setting
        original_keras_verbose = self.config.get('keras_verbose', 1)
        
        # Set keras_verbose based on verbose parameter
        if not verbose:
            self.config['keras_verbose'] = 0
        
        # Pre-calculate KDE bandwidth if needed
        original_bandwidth = None
        bandwidth_calculated = False
        
        if (self.config.get("weight_samples", {}).get("enabled", False) and
            self.config.get("weight_samples", {}).get("method") == "KD"):
            
            existing_bandwidth = self.config.get("weight_samples", {}).get("bandwidth")
            
            if existing_bandwidth is None:
                # Get all samples with coordinates for bandwidth calculation
                coords_mask = ~na_mask
                all_train_locs = locs[coords_mask]
                
                if len(all_train_locs) > 1:
                    if verbose:
                        print("Pre-calculating optimal KDE bandwidth for k-fold CV...")
                    
                    from .sample_weights import get_global_bandwidth_optimizer
                    optimizer = get_global_bandwidth_optimizer()
                    
                    optimal_bandwidth = optimizer.get_bandwidth(
                        all_train_locs,
                        cache_key=f"kfold_k{k}_n{len(all_train_locs)}",
                        n_bandwidths=self.config.get("weight_samples", {}).get("n_bandwidths", 100),
                        verbose=verbose
                    )
                    
                    # Temporarily set in config
                    self.config["weight_samples"]["bandwidth"] = optimal_bandwidth
                    bandwidth_calculated = True
                    
                    if verbose:
                        print(f"Using bandwidth: {optimal_bandwidth:.3f}")
        
        if verbose:
            print(f"Running true {k}-fold cross-validation with nonoverlapping holdout sets")
            fold_iterator = tqdm(enumerate(fold_index_sets), total=k, desc="K-fold progress")
        else:
            fold_iterator = enumerate(fold_index_sets)

        for fold_num, index_set in fold_iterator:
            self.model = None
            # Reset sample weights to ensure proper recalculation for each fold
            self.sample_weights = None
            # Use the test indices from this fold as holdout
            holdout_indices = index_set.test
            self.train_holdout(
                genotypes=genotypes,
                samples=samples,
                holdout_indices=holdout_indices,
            )
            preds = self.predict_holdout(
                verbose=False,
                return_df=True,
                save_preds_to_disk=not save_full_pred_matrix,
                plot_summary=False,  # Never plot during k-fold CV
            )
            # preds: one row per held-out sample in this fold
            for _, row in preds.iterrows():
                pred_rows.append({
                    "sampleID": row["sampleID"],
                    "x_pred": row["x_pred"],
                    "y_pred": row["y_pred"],
                    "fold": fold_num
                })
            keras.backend.clear_session()

        # Restore original keras_verbose setting
        self.config['keras_verbose'] = original_keras_verbose
        
        # Restore original bandwidth setting if we changed it
        if bandwidth_calculated:
            if original_bandwidth is None:
                # Remove the key if it wasn't there originally
                self.config.get("weight_samples", {}).pop("bandwidth", None)
            else:
                self.config["weight_samples"]["bandwidth"] = original_bandwidth

        if return_df:
            all_predictions = pd.DataFrame(pred_rows)
            if save_full_pred_matrix:
                all_predictions.to_csv(
                    f"{self.config['out']}_kfold_holdouts_predlocs.csv", index=False
                )
            return all_predictions
        return None