"""Analysis functionality for locator"""

import numpy as np
import pandas as pd
import copy
from tqdm import tqdm
from tensorflow import keras
import zarr

from .utils import filter_snps, normalize_locs


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
            
        Returns:
            pandas.DataFrame or None: If return_df=True, returns DataFrame with predictions
                for each window, otherwise None
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
        """Run bootstrap analysis by resampling SNPs with replacement.
        
        Args:
            genotypes: Array of genotype data
            samples: Sample IDs corresponding to genotypes
            n_bootstraps: Number of bootstrap replicates to run
            return_df: Whether to return DataFrame with all predictions
            save_full_pred_matrix: Whether to save full prediction matrix to disk
            
        Returns:
            pandas.DataFrame or None: If return_df=True, returns DataFrame with predictions
                for each bootstrap, otherwise None
        """
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

    def run_holdouts(
        self,
        genotypes,
        samples,
        k=10,
        n_reps=10,
        holdout_indices=None,
        return_df=False,
        save_full_pred_matrix=True,
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
            
        Returns:
            pandas.DataFrame or None: If return_df=True, returns DataFrame with predictions
                for each holdout replicate, otherwise None
        """
        # Store samples
        self.samples = samples

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

        print(f"Running {n_reps} holdout replicates")

        for rep in tqdm(range(n_reps)):
            # Clear existing model
            self.model = None

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
            )

            if return_df:
                # Rename columns to include replicate number
                holdout_preds = preds[["x_pred", "y_pred"]].copy()
                holdout_preds.columns = [f"x_rep{rep}", f"y_rep{rep}"]
                holdout_preds["sampleID"] = preds["sampleID"]
                pred_dfs.append(holdout_preds)

            # Clear keras session
            keras.backend.clear_session()

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
            
        Returns:
            pandas.DataFrame or None: If return_df=True, returns DataFrame with predictions
                for each jacknife replicate, otherwise None
        """
        # Store samples
        self.samples = samples

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
            
        Returns:
            pandas.DataFrame or None: If return_df=True, returns DataFrame with predictions
                for each window, otherwise None
        """
        # Store samples
        self.samples = samples

        # Get positions
        if not hasattr(self, "positions"):
            if hasattr(self, "_genotype_df"):
                self.positions = np.array(self._genotype_df.columns, dtype=int)
            elif self.config.get("zarr"):
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

        # Create lists to store predictions
        pred_dfs = []

        print(f"Running windowed analysis for holdout samples")

        for start in tqdm(windows):
            stop = start + int(window_size)
            in_window = (self.positions >= start) & (self.positions < stop)

            if sum(in_window) > 0:
                # Get genotypes for this window
                window_genos = genotypes[in_window, :, :]

                # Clear existing model
                self.model = None

                # Train with holdout on window data
                self.train_holdout(
                    genotypes=window_genos,
                    samples=samples,
                    k=k,
                    holdout_indices=holdout_indices,
                )

                # Get predictions for holdout samples
                preds = self.predict_holdout(
                    verbose=False,
                    return_df=True,
                    save_preds_to_disk=not save_full_pred_matrix,
                )

                if return_df:
                    # Rename columns to include window position
                    window_preds = preds[["x_pred", "y_pred"]].copy()
                    window_preds.columns = [f"x_pos{start}", f"y_pos{start}"]
                    window_preds["sampleID"] = preds["sampleID"]
                    pred_dfs.append(window_preds)

                # Clear keras session
                keras.backend.clear_session()

        if return_df:
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