"""Holdout analysis methods (holdouts, jacknife holdouts, leave-one-out, k-fold)."""

import copy

import numpy as np
import pandas as pd
from tensorflow import keras
from tqdm import tqdm

from ..data import IndexSet


class HoldoutMixin:
    """Mixin providing holdout-based analysis methods."""

    def run_holdouts(  # noqa: C901
        self,
        genotypes,
        samples,
        k=10,
        n_reps=10,
        holdout_indices=None,
        holdout_sample_ids=None,
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
            holdout_sample_ids: Optional list of sample IDs to hold out. If provided,
                these specific samples will be held out (overrides k and holdout_indices).
                Can be a single list (used for all replicates) or list of lists
                (different samples per replicate).
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

        Notes
        -----
            - With na_action='separate': Currently behaves like 'exclude' (holdouts
              must have known locations). Future versions may support predicting NA samples.
            - With na_action='exclude': Only uses samples with known locations (current behavior)
            - With na_action='fail': Raises error if any NA samples found
        """
        # Store samples
        self.samples = samples

        na_action, status = self._validate_na_action(
            samples, na_action, "Holdout analysis"
        )
        if status["n_na"] > 0 and na_action == "separate":
            print(
                "Note: Holdout analysis requires known locations; 'separate' behaves like 'exclude'"
            )

        # Create lists to store predictions
        pred_dfs = []

        # Get sample data and locations
        sample_data, locs = self._resolve_locations(samples)

        # Get indices of samples with known locations
        known_idx = np.argwhere(~np.isnan(locs[:, 0]))
        known_idx = np.array([x[0] for x in known_idx])

        # Handle holdout_sample_ids if provided
        if holdout_sample_ids is not None:
            # Convert samples to list if it's a numpy array
            if hasattr(samples, "tolist"):
                samples_list = samples.tolist()
            else:
                samples_list = list(samples)

            # Convert sample IDs to indices
            if isinstance(holdout_sample_ids[0], str):
                # Single list of sample IDs for all replicates
                try:
                    holdout_indices = [
                        [samples_list.index(sid) for sid in holdout_sample_ids]
                    ]
                except ValueError:
                    missing = [
                        sid for sid in holdout_sample_ids if sid not in samples_list
                    ]
                    raise ValueError(f"Sample IDs not found in samples list: {missing}")
                # Replicate for all n_reps if needed
                holdout_indices = holdout_indices * n_reps
                k = len(holdout_sample_ids)  # Update k to match
            else:
                # List of lists - different sample IDs per replicate
                holdout_indices = []
                for rep_ids in holdout_sample_ids:
                    try:
                        rep_indices = [samples_list.index(sid) for sid in rep_ids]
                    except ValueError:
                        missing = [sid for sid in rep_ids if sid not in samples_list]
                        raise ValueError(
                            f"Sample IDs not found in samples list: {missing}"
                        )
                    holdout_indices.append(rep_indices)
                n_reps = len(holdout_indices)  # Update n_reps to match
                k = len(holdout_indices[0]) if holdout_indices else 0

        if k >= len(known_idx):
            raise ValueError(
                f"k ({k}) must be less than number of samples with known locations ({len(known_idx)})"
            )

        # Pre-calculate KDE bandwidth if needed
        original_bandwidth = None
        bandwidth_calculated = False

        if (
            self.config.get("weight_samples", {}).get("enabled", False)
            and self.config.get("weight_samples", {}).get("method") == "KD"
        ):
            existing_bandwidth = self.config.get("weight_samples", {}).get("bandwidth")

            if existing_bandwidth is None:
                # Get all samples with coordinates for bandwidth calculation
                all_train_locs = locs[known_idx]

                if len(all_train_locs) > 1:
                    print(
                        "Pre-calculating optimal KDE bandwidth for holdout analysis..."
                    )

                    from ..sample_weights import get_global_bandwidth_optimizer

                    optimizer = get_global_bandwidth_optimizer()

                    optimal_bandwidth = optimizer.get_bandwidth(
                        all_train_locs,
                        cache_key=f"holdouts_k{k}_n{len(all_train_locs)}",
                        n_bandwidths=self.config.get("weight_samples", {}).get(
                            "n_bandwidths", 100
                        ),
                        verbose=True,
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

    def run_jacknife_holdouts(  # noqa: C901
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

        Returns
        -------
            pandas.DataFrame or None: If return_df=True, returns DataFrame with predictions
                for each jacknife replicate containing columns:
                - sampleID: Sample identifier
                - x_pred: Predicted longitude
                - y_pred: Predicted latitude
                - boot: Jacknife replicate number (0 to n_boots-1)

                Note: True locations are not included. Merge with sample metadata to calculate errors.

        Notes
        -----
            - With na_action='separate': Currently behaves like 'exclude' (holdouts
              must have known locations). Future versions may support predicting NA samples.
            - With na_action='exclude': Only uses samples with known locations (current behavior)
            - With na_action='fail': Raises error if any NA samples found
        """
        # Store samples
        self.samples = samples

        na_action, status = self._validate_na_action(
            samples, na_action, "Jacknife holdout analysis"
        )
        if status["n_na"] > 0 and na_action == "separate":
            print(
                "Note: Holdout analysis requires known locations; 'separate' behaves like 'exclude'"
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

        if (
            self.config.get("weight_samples", {}).get("enabled", False)
            and self.config.get("weight_samples", {}).get("method") == "KD"
        ):
            existing_bandwidth = self.config.get("weight_samples", {}).get("bandwidth")

            if (
                existing_bandwidth is None
                and hasattr(self, "trainlocs")
                and len(self.trainlocs) > 1
            ):
                print(
                    "Pre-calculating optimal KDE bandwidth for jacknife holdout analysis..."
                )

                from ..sample_weights import get_global_bandwidth_optimizer

                optimizer = get_global_bandwidth_optimizer()

                optimal_bandwidth = optimizer.get_bandwidth(
                    self.trainlocs,
                    cache_key=f"jacknife_holdouts_n{len(self.trainlocs)}",
                    n_bandwidths=self.config.get("weight_samples", {}).get(
                        "n_bandwidths", 100
                    ),
                    verbose=True,
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

        Returns
        -------
            pandas.DataFrame or None: DataFrame with predictions for each left-out sample
        """
        # Get sample status to determine k
        status = self.get_sample_status(samples)
        n_known = status["n_known"]

        if n_known == 0:
            raise ValueError("No samples with known coordinates for leave-one-out CV")

        print(f"Running leave-one-out cross-validation for {n_known} samples")

        # For large leave-one-out, warn about memory usage
        if n_known > 50 and not self.config.get("disable_gpu", False):
            print("Warning: Leave-one-out with many samples may accumulate GPU memory.")
            print(
                "Consider setting config['disable_gpu'] = True if you encounter memory issues."
            )

            # Also ensure HDF5 optimization is enabled for LOO
            if not self.config.get("holdout_no_intermediate_saves", True):
                print(
                    "Tip: Enabling holdout_no_intermediate_saves will improve performance."
                )
                self.config["holdout_no_intermediate_saves"] = True

        # Run k-fold with k equal to number of known samples
        # This will create folds with exactly 1 sample each
        result = self.run_k_fold_holdouts(
            genotypes=genotypes,
            samples=samples,
            k=n_known,
            return_df=return_df,
            save_full_pred_matrix=False,  # We'll save with our own name
            verbose=False,  # We already printed our message
            na_action=na_action,
        )

        # Save with leave-one-out specific filename if requested
        if result is not None and save_full_pred_matrix:
            result.to_csv(
                f"{self.config['out']}_leave_one_out_predlocs.csv", index=False
            )

        return result

    def run_k_fold_holdouts(  # noqa: C901
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

        Notes
        -----
            - With na_action='separate': Currently behaves like 'exclude' (k-fold requires
              known locations). Future versions may support predicting NA samples.
            - With na_action='exclude': Only uses samples with known locations (current behavior)
            - With na_action='fail': Raises error if any NA samples found

        Example:
            >>> # Run k-fold cross-validation
            >>> predictions = locator.run_k_fold_holdouts(genotypes, samples, k=10, return_df=True)
            >>>
            >>> # Merge with true locations to calculate errors
            >>> sample_data = pd.read_csv('samples.tsv', sep='\\t')
            >>> merged = predictions.merge(sample_data[['sampleID', 'x', 'y']], on='sampleID')
            >>> merged['error_km'] = np.sqrt(
            ...     (merged['x'] - merged['x_pred'])**2 +
            ...     (merged['y'] - merged['y_pred'])**2
            ... ) * 111.32  # Convert degrees to km
        """
        self.samples = samples

        # Resolve na_action default and validate; _validate_na_action always
        # prints, but k-fold optionally suppresses output via verbose flag.
        if na_action is None:
            na_action = self.na_action
        status = self.get_sample_status(samples)

        if verbose:
            print(
                f"K-fold CV: {status['n_known']} samples with coordinates, "
                f"{status['n_na']} without"
            )
            if status["n_na"] > 0:
                print(f"NA handling mode: {na_action}")
                if na_action == "separate":
                    print(
                        "Note: K-fold CV requires known locations; 'separate' behaves like 'exclude'"
                    )

        if na_action == "fail" and status["n_na"] > 0:
            raise ValueError(
                f"Found {status['n_na']} samples without coordinates. "
                f"Set na_action='separate' or 'exclude' to proceed."
            )

        pred_rows = []

        # Get sample data and locations
        sample_data, locs = self._resolve_locations(samples)

        # Create NA mask
        na_mask = np.isnan(locs[:, 0])
        n_total_samples = len(locs)
        n_samples_with_coords = np.sum(~na_mask)

        if k > n_samples_with_coords:
            raise ValueError(
                f"k ({k}) must be less than or equal to number of samples with known locations ({n_samples_with_coords})"
            )

        # Create list to store IndexSets for each fold
        # Use a fixed seed based on config seed or numpy's current state
        if "seed" in self.config and self.config["seed"] is not None:
            kfold_seed = self.config["seed"]
        else:
            # Generate a seed from current numpy state to ensure consistency
            kfold_seed = np.random.randint(0, 2**31)

        fold_index_sets = []
        for fold_idx in range(k):
            index_set = IndexSet.from_k_fold(
                n=n_total_samples,
                k=k,
                fold=fold_idx,
                seed=kfold_seed,  # Use consistent seed for all folds
                na_mask=na_mask,
            )
            fold_index_sets.append(index_set)

        # Store original keras_verbose setting
        original_keras_verbose = self.config.get("keras_verbose", 1)

        # Set keras_verbose based on verbose parameter
        if not verbose:
            self.config["keras_verbose"] = 0

        # Pre-calculate KDE bandwidth if needed
        original_bandwidth = None
        bandwidth_calculated = False

        if (
            self.config.get("weight_samples", {}).get("enabled", False)
            and self.config.get("weight_samples", {}).get("method") == "KD"
        ):
            existing_bandwidth = self.config.get("weight_samples", {}).get("bandwidth")

            if existing_bandwidth is None:
                # Get all samples with coordinates for bandwidth calculation
                coords_mask = ~na_mask
                all_train_locs = locs[coords_mask]

                if len(all_train_locs) > 1:
                    if verbose:
                        print("Pre-calculating optimal KDE bandwidth for k-fold CV...")

                    from ..sample_weights import get_global_bandwidth_optimizer

                    optimizer = get_global_bandwidth_optimizer()

                    optimal_bandwidth = optimizer.get_bandwidth(
                        all_train_locs,
                        cache_key=f"kfold_k{k}_n{len(all_train_locs)}",
                        n_bandwidths=self.config.get("weight_samples", {}).get(
                            "n_bandwidths", 100
                        ),
                        verbose=verbose,
                    )

                    # Temporarily set in config
                    self.config["weight_samples"]["bandwidth"] = optimal_bandwidth
                    bandwidth_calculated = True

                    if verbose:
                        print(f"Using bandwidth: {optimal_bandwidth:.3f}")

        if verbose:
            print(
                f"Running true {k}-fold cross-validation with nonoverlapping holdout sets"
            )
            fold_iterator = tqdm(
                enumerate(fold_index_sets), total=k, desc="K-fold progress"
            )
        else:
            fold_iterator = enumerate(fold_index_sets)

        for fold_num, index_set in fold_iterator:
            # Clear any existing model and GPU memory before starting
            if self.model is not None:
                del self.model
                self.model = None
            # Reset sample weights to ensure proper recalculation for each fold
            self.sample_weights = None

            # Clear Keras session and GPU memory
            keras.backend.clear_session()
            if not self.config.get("disable_gpu", False):
                # Force garbage collection to free GPU memory
                import gc

                gc.collect()

            # Store original output path and modify for this fold
            original_out = self.config.get("out", "locator")
            self.config["out"] = f"{original_out}_fold{fold_num}"

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
                pred_rows.append(
                    {
                        "sampleID": row["sampleID"],
                        "x_pred": row["x_pred"],
                        "y_pred": row["y_pred"],
                        "fold": fold_num,
                    }
                )

            # Clear model reference again after prediction
            if self.model is not None:
                del self.model
                self.model = None

            # Restore original output path
            self.config["out"] = original_out

        # Restore original keras_verbose setting
        self.config["keras_verbose"] = original_keras_verbose

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
