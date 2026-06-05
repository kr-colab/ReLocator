"""Windowed analysis methods."""

import numpy as np
import pandas as pd
from tensorflow import keras
from tqdm import tqdm

from ..data import IndexSet, normalize_locs
from ..data.filters import is_dosage_matrix


class WindowedMixin:
    """Mixin providing windowed analysis methods."""

    def run_windows(  # noqa: C901
        self,
        genotypes,
        samples,
        window_start=0,
        window_size=5e5,
        window_stop=None,
        respect_chromosomes=True,
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
            respect_chromosomes: Whether to respect chromosome boundaries when creating
                windows (default: True). If True, windows will not span chromosome
                boundaries. Requires chromosome information from VCF/Zarr input.
            return_df: Whether to return DataFrame with all predictions
            save_full_pred_matrix: Whether to save full prediction matrix to disk
            na_action: How to handle NA samples ('separate', 'exclude', 'fail').
                If None, uses self.na_action

        Returns
        -------
            pandas.DataFrame or None: If return_df=True, returns DataFrame with predictions
                for each window, otherwise None

        Notes
        -----
            - With na_action='separate': Trains on samples with known locations,
              can predict on samples with NA locations
            - With na_action='exclude': Only uses samples with known locations
            - With na_action='fail': Raises error if any NA samples found

        Warning:
            When respect_chromosomes=False, window analysis treats all SNP positions as
            continuous along a single coordinate axis. If your data contains multiple
            chromosomes, windows may span across chromosome boundaries. Use
            respect_chromosomes=True (default) for biologically meaningful windows.
        """
        # Store samples
        self.samples = samples

        na_action, status = self._validate_na_action(
            samples, na_action, "Window analysis"
        )

        # Get positions if not already stored
        self._ensure_positions()

        if window_stop is None:
            window_stop = max(self.positions)

        # Generate windows using the new helper function
        from ..data.windows import generate_genomic_windows

        chromosomes = getattr(self, "chromosomes", None)
        windows = generate_genomic_windows(
            positions=self.positions,
            chromosomes=chromosomes,
            window_start=window_start,
            window_size=int(window_size),
            window_stop=window_stop,
            respect_chromosomes=respect_chromosomes,
            min_snps_per_window=self.config.get("min_snps_per_window", 1),
            verbose=self.config.get("verbose", False),
        )

        # Initial training to set up model and data
        if len(windows) > 0:
            first_window_indices = windows[0]["indices"]
            if np.sum(first_window_indices) > 0:
                if is_dosage_matrix(genotypes):
                    window_genos = genotypes[first_window_indices, :]
                else:
                    window_genos = genotypes[first_window_indices, :, :]
                self.train(genotypes=window_genos, samples=samples, na_action=na_action)

        # Create lists to store predictions
        pred_dfs = []

        print(f"Starting window analysis ({len(windows)} windows)")
        for window in tqdm(windows):
            if window["n_snps"] > 0:
                # Get genotypes for this window
                if is_dosage_matrix(genotypes):
                    window_genos = genotypes[window["indices"], :]
                else:
                    window_genos = genotypes[window["indices"], :, :]

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
                    # Rename columns to include window label
                    window_preds = preds[["sampleID", "x", "y"]].copy()
                    window_preds.columns = [
                        "sampleID",
                        f"x_{window['label']}",
                        f"y_{window['label']}",
                    ]
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
                all_predictions = all_predictions.merge(
                    pred_df, on="sampleID", how="outer"
                )

            if save_full_pred_matrix:
                all_predictions.to_csv(
                    f"{self.config['out']}_windows_predlocs.csv", index=False
                )
            return all_predictions

        return None

    def run_windows_holdouts(  # noqa: C901
        self,
        genotypes,
        samples,
        k=10,
        window_start=0,
        window_size=5e5,
        window_stop=None,
        respect_chromosomes=True,
        holdout_indices=None,
        holdout_sample_ids=None,
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
            respect_chromosomes: Whether to respect chromosome boundaries when creating
                windows (default: True). If True, windows will not span chromosome
                boundaries. Requires chromosome information from VCF/Zarr input.
            holdout_indices: Optional specific indices to hold out
            holdout_sample_ids: Optional list of sample IDs to hold out. If provided,
                these specific samples will be held out (overrides k and holdout_indices).
            return_df: Whether to return DataFrame with all predictions
            save_full_pred_matrix: Whether to save full prediction matrix to disk
            na_action: How to handle NA samples ('separate', 'exclude', 'fail').
                If None, uses self.na_action

        Returns
        -------
            pandas.DataFrame or None: If return_df=True, returns DataFrame with predictions
                for each window, otherwise None

        Notes
        -----
            - With na_action='separate': Currently behaves like 'exclude' (holdouts
              must have known locations). Future versions may support predicting NA samples.
            - With na_action='exclude': Only uses samples with known locations (current behavior)
            - With na_action='fail': Raises error if any NA samples found

        Warning:
            When respect_chromosomes=False, window analysis treats all SNP positions as
            continuous along a single coordinate axis. If your data contains multiple
            chromosomes, windows may span across chromosome boundaries. Use
            respect_chromosomes=True (default) for biologically meaningful windows.
        """
        # Store samples and genotypes for efficient access
        self.samples = samples
        self.genotypes = genotypes

        na_action, status = self._validate_na_action(
            samples, na_action, "Windows holdout analysis"
        )
        if status["n_na"] > 0 and na_action == "separate":
            print(
                "Note: Holdout analysis requires known locations; 'separate' behaves like 'exclude'"
            )

        # Create NA mask for IndexSet construction
        na_mask = None
        if status["n_na"] > 0:
            # Create boolean mask for NA samples
            if isinstance(samples, pd.DataFrame):
                na_mask = samples["x"].isna() | samples["y"].isna()
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
                na_mask = merged["x"].isna() | merged["y"].isna()
            na_mask = na_mask.values

        # Get positions and create holdout IndexSet
        self._ensure_positions()

        # Handle holdout_sample_ids if provided
        if holdout_sample_ids is not None:
            # Convert samples to list if it's a numpy array
            if hasattr(samples, "tolist"):
                samples_list = samples.tolist()
            else:
                samples_list = list(samples)

            # Convert sample IDs to indices
            try:
                holdout_indices = [samples_list.index(sid) for sid in holdout_sample_ids]
            except ValueError:
                missing = [sid for sid in holdout_sample_ids if sid not in samples_list]
                raise ValueError(f"Sample IDs not found in samples list: {missing}")
            k = len(holdout_indices)  # Update k to match

        # Create IndexSet for holdout splitting
        n_samples = len(samples)
        if holdout_indices is not None:
            # Use provided holdout indices
            holdout_idx = np.array(holdout_indices)
            train_idx = np.setdiff1d(np.arange(n_samples), holdout_idx)

            # Apply NA mask if needed
            if na_mask is not None and (
                na_action == "exclude" or na_action == "separate"
            ):
                # Only keep samples with known coordinates
                valid_mask = ~na_mask
                holdout_idx = holdout_idx[valid_mask[holdout_idx]]
                train_idx = train_idx[valid_mask[train_idx]]

            index_set = IndexSet(
                indices={"train": train_idx, "test": holdout_idx},
                total_samples=n_samples,
                na_mask=na_mask,
            )
        else:
            # Random holdout selection using IndexSet
            index_set = IndexSet.random_split(
                n=n_samples,
                splits={"train": 1.0 - k / n_samples, "test": k / n_samples},
                seed=self.config.get("seed", 42),
                na_mask=na_mask,
                na_action=na_action if na_action != "separate" else "exclude",
            )

        if window_stop is None:
            window_stop = max(self.positions)

        # Generate windows using the new helper function
        from ..data.windows import generate_genomic_windows

        chromosomes = getattr(self, "chromosomes", None)
        windows = generate_genomic_windows(
            positions=self.positions,
            chromosomes=chromosomes,
            window_start=window_start,
            window_size=int(window_size),
            window_stop=window_stop,
            respect_chromosomes=respect_chromosomes,
            min_snps_per_window=self.config.get("min_snps_per_window", 1),
            verbose=self.config.get("verbose", False),
        )

        # Create lists to store predictions
        pred_dfs = []

        # Pre-calculate KDE bandwidth if needed
        original_bandwidth = None
        bandwidth_calculated = False

        if (
            self.config.get("weight_samples", {}).get("enabled", False)
            and self.config.get("weight_samples", {}).get("method") == "KD"
        ):
            existing_bandwidth = self.config.get("weight_samples", {}).get("bandwidth")

            if existing_bandwidth is None:
                # Get sample data and locations
                sample_data, locs = self._resolve_locations(samples)

                # Get training locations (exclude holdout samples)
                train_mask = np.ones(len(samples), dtype=bool)
                train_mask[index_set.test] = False
                train_mask = train_mask & ~np.isnan(locs[:, 0])
                train_locs = locs[train_mask]

                if len(train_locs) > 1:
                    print(
                        "Pre-calculating optimal KDE bandwidth for windows holdout analysis..."
                    )

                    from ..sample_weights import get_global_bandwidth_optimizer

                    optimizer = get_global_bandwidth_optimizer()

                    optimal_bandwidth = optimizer.get_bandwidth(
                        train_locs,
                        cache_key=f"windows_holdouts_n{len(train_locs)}",
                        n_bandwidths=self.config.get("weight_samples", {}).get(
                            "n_bandwidths", 100
                        ),
                        verbose=True,
                    )

                    # Temporarily set in config
                    self.config["weight_samples"]["bandwidth"] = optimal_bandwidth
                    bandwidth_calculated = True

                    print(f"Using bandwidth: {optimal_bandwidth:.3f}")

        print(f"Running windowed analysis for holdout samples ({len(windows)} windows)")

        # Store the full IndexSet for use across windows
        self.index_set = index_set

        # Pre-normalize locations for efficiency
        _, locs = self._resolve_locations(samples)

        # Normalize locations once
        (
            self.meanlong,
            self.sdlong,
            self.meanlat,
            self.sdlat,
            self.unnormedlocs,
            normalized_locs,
        ) = normalize_locs(locs)

        for window in tqdm(windows):
            snp_indices = np.where(window["indices"])[0]

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
                    # Rename columns to include window label
                    window_preds = preds[["x_pred", "y_pred"]].copy()
                    window_preds.columns = [
                        f"x_{window['label']}",
                        f"y_{window['label']}",
                    ]
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
