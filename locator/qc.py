"""Quality control functions for genotype data."""

import warnings
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd


def calculate_sample_stats(genotypes) -> pd.DataFrame:
    """
    Calculate per-sample statistics for genotype data.

    Args:
        genotypes: GenotypeArray of shape (n_sites, n_samples, ploidy)

    Returns:
        DataFrame with columns:
        - sampleID: Sample index (will be replaced with actual IDs later)
        - n_missing: Number of missing genotypes
        - missing_rate: Proportion of missing genotypes
        - n_heterozygous: Number of heterozygous sites
        - heterozygosity: Proportion of heterozygous sites (excluding missing)
        - mean_genotype: Mean genotype value (excluding missing)
    """
    n_sites, n_samples = genotypes.shape[:2]

    # Convert to genotype counts (0, 1, 2) and identify missing
    if hasattr(genotypes, "is_missing"):
        # scikit-allel GenotypeArray
        gt_counts = genotypes.to_n_alt()
        # Get missing mask - shape (n_sites, n_samples)
        missing_mask = genotypes.is_missing()
    else:
        # Numpy array - sum across ploidy dimension
        gt_counts = np.sum(genotypes, axis=2)
        # Identify missing data (assuming -1 or any negative value)
        missing_mask = np.any(genotypes < 0, axis=2)

    # Calculate statistics for each sample
    stats = []
    for i in range(n_samples):
        sample_gt = gt_counts[:, i]

        # Missing data from the mask
        is_missing = missing_mask[:, i]
        n_missing = np.sum(is_missing)
        missing_rate = n_missing / n_sites

        # Non-missing genotypes
        valid_gt = sample_gt[~is_missing]

        if len(valid_gt) > 0:
            # Heterozygosity (genotype count = 1)
            n_het = np.sum(valid_gt == 1)
            het_rate = n_het / len(valid_gt)

            # Mean genotype
            mean_gt = np.mean(valid_gt)
        else:
            n_het = 0
            het_rate = np.nan
            mean_gt = np.nan

        stats.append(
            {
                "sampleID": i,  # Will be replaced with actual IDs
                "n_missing": n_missing,
                "missing_rate": missing_rate,
                "n_heterozygous": n_het,
                "heterozygosity": het_rate,
                "mean_genotype": mean_gt,
            }
        )

    return pd.DataFrame(stats)


def detect_outliers_mad(values: np.ndarray, n_mad: float = 3.0) -> np.ndarray:
    """
    Detect outliers using Median Absolute Deviation (MAD).

    Args:
        values: Array of values to check
        n_mad: Number of MADs from median to consider outlier

    Returns:
        Boolean array indicating outliers
    """
    median = np.nanmedian(values)
    mad = np.nanmedian(np.abs(values - median))

    # Use a small constant to avoid division by zero
    if mad == 0:
        mad = 1e-6

    # Modified Z-score using MAD
    modified_z_scores = np.abs(values - median) / (1.4826 * mad)

    return modified_z_scores > n_mad


def detect_outliers_iqr(values: np.ndarray, multiplier: float = 1.5) -> np.ndarray:
    """
    Detect outliers using Interquartile Range (IQR).

    Args:
        values: Array of values to check
        multiplier: IQR multiplier for outlier threshold

    Returns:
        Boolean array indicating outliers
    """
    q1 = np.nanpercentile(values, 25)
    q3 = np.nanpercentile(values, 75)
    iqr = q3 - q1

    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr

    return (values < lower_bound) | (values > upper_bound)


def detect_outliers_zscore(values: np.ndarray, n_std: float = 3.0) -> np.ndarray:
    """
    Detect outliers using Z-score.

    Args:
        values: Array of values to check
        n_std: Number of standard deviations from mean

    Returns:
        Boolean array indicating outliers
    """
    mean = np.nanmean(values)
    std = np.nanstd(values)

    if std == 0:
        return np.zeros_like(values, dtype=bool)

    z_scores = np.abs(values - mean) / std
    return z_scores > n_std


def detect_outliers_threshold(values: np.ndarray, threshold: float) -> np.ndarray:
    """
    Detect outliers using a simple threshold.

    Args:
        values: Array of values to check
        threshold: Values above this are considered outliers

    Returns:
        Boolean array indicating outliers
    """
    return values > threshold


def _apply_outlier_detection(
    missing_rates: np.ndarray,
    method: str,
    threshold: Optional[float] = None,
    n_std: float = 3.0,
    n_mad: float = 3.0,
    iqr_multiplier: float = 1.5,
) -> np.ndarray:
    """Apply the specified outlier detection method."""
    if method == "mad":
        return detect_outliers_mad(missing_rates, n_mad)
    elif method == "iqr":
        return detect_outliers_iqr(missing_rates, iqr_multiplier)
    elif method == "zscore":
        return detect_outliers_zscore(missing_rates, n_std)
    elif method == "threshold":
        if threshold is None:
            threshold = 0.1  # Default 10% missing
        return detect_outliers_threshold(missing_rates, threshold)
    else:
        raise ValueError(
            f"Unknown method: {method}. Choose from 'mad', 'iqr', 'zscore', 'threshold'"
        )


def _create_summary_text(
    stats_df: pd.DataFrame,
    outlier_mask: np.ndarray,
    missing_rates: np.ndarray,
    method: str,
    threshold: Optional[float] = None,
    n_mad: float = 3.0,
    iqr_multiplier: float = 1.5,
    n_std: float = 3.0,
    suggest_exclusions: bool = True,
) -> str:
    """Create summary text for QC results."""
    n_samples = len(stats_df)
    n_outliers = np.sum(outlier_mask)
    mean_missing = np.mean(missing_rates)
    median_missing = np.median(missing_rates)
    outlier_samples = stats_df.loc[outlier_mask, "sampleID"].tolist()

    summary_lines = [
        "Genotype Quality Control Summary:",
        f"Total samples: {n_samples}",
        f"Mean missing rate: {mean_missing:.3f}",
        f"Median missing rate: {median_missing:.3f}",
        "",
        f"Outlier detection method: {method}",
    ]

    # Add method-specific parameters
    if method == "threshold":
        summary_lines.append(f"Threshold: {threshold:.3f}")
    elif method == "mad":
        summary_lines.append(f"MAD multiplier: {n_mad}")
    elif method == "iqr":
        summary_lines.append(f"IQR multiplier: {iqr_multiplier}")
    elif method == "zscore":
        summary_lines.append(f"Z-score threshold: {n_std}")

    summary_lines.extend([f"Outliers found: {n_outliers}", ""])

    # Add outlier details
    if n_outliers > 0:
        outlier_df = stats_df[outlier_mask].sort_values("missing_rate", ascending=False)
        summary_lines.append("Top outlier samples:")
        for idx, row in outlier_df.head(5).iterrows():
            summary_lines.append(
                f"  {row['sampleID']}: {row['missing_rate']:.3f} missing"
            )

        if n_outliers > 5:
            summary_lines.append(f"  ... and {n_outliers - 5} more")

    # Add exclusion suggestion
    if suggest_exclusions and n_outliers > 0:
        summary_lines.extend(
            [
                "",
                "Suggested action:",
                f"locator.exclude_samples({outlier_samples[:5]!r}{', ...' if n_outliers > 5 else ''}, reason='high_missingness')",
            ]
        )

    return "\n".join(summary_lines)


def _create_qc_plots(
    stats_df: pd.DataFrame, outlier_mask: np.ndarray, missing_rates: np.ndarray
):
    """Create QC visualization plots."""
    try:
        import matplotlib.pyplot as plt

        n_samples = len(stats_df)
        n_outliers = np.sum(outlier_mask)
        mean_missing = np.mean(missing_rates)

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle("Genotype Quality Control", fontsize=14)

        # Missing rate distribution
        ax = axes[0, 0]
        ax.hist(
            missing_rates[~outlier_mask],
            bins=30,
            alpha=0.7,
            label="Normal",
            color="blue",
        )
        if n_outliers > 0:
            ax.hist(
                missing_rates[outlier_mask],
                bins=10,
                alpha=0.7,
                label="Outliers",
                color="red",
            )
        ax.axvline(
            mean_missing,
            color="black",
            linestyle="--",
            label=f"Mean: {mean_missing:.3f}",
        )
        ax.set_xlabel("Missing Rate")
        ax.set_ylabel("Count")
        ax.set_title("Missing Data Distribution")
        ax.legend()

        # Boxplot of missing rates
        ax = axes[0, 1]
        box_data = [missing_rates[~outlier_mask]]
        labels = ["Normal"]
        if n_outliers > 0:
            box_data.append(missing_rates[outlier_mask])
            labels.append("Outliers")
        bp = ax.boxplot(box_data, labels=labels, patch_artist=True)
        for patch, color in zip(bp["boxes"], ["blue", "red"]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax.set_ylabel("Missing Rate")
        ax.set_title("Missing Rate by Group")

        # Heterozygosity vs Missing Rate
        ax = axes[1, 0]
        ax.scatter(
            stats_df["missing_rate"],
            stats_df["heterozygosity"],
            c=["red" if x else "blue" for x in outlier_mask],
            alpha=0.6,
        )
        ax.set_xlabel("Missing Rate")
        ax.set_ylabel("Heterozygosity")
        ax.set_title("Heterozygosity vs Missing Rate")

        # Sample index vs missing rate
        ax = axes[1, 1]
        ax.scatter(
            range(n_samples),
            missing_rates,
            c=["red" if x else "blue" for x in outlier_mask],
            alpha=0.6,
        )
        ax.set_xlabel("Sample Index")
        ax.set_ylabel("Missing Rate")
        ax.set_title("Missing Rate by Sample Order")

        plt.tight_layout()
        return fig

    except ImportError:
        warnings.warn("Matplotlib not available. Skipping plot generation.")
        return None


def check_genotypes(
    genotypes,
    samples: Optional[Union[List[str], np.ndarray]] = None,
    method: str = "mad",
    threshold: Optional[float] = None,
    n_std: float = 3.0,
    n_mad: float = 3.0,
    iqr_multiplier: float = 1.5,
    plot: bool = True,
    return_stats: bool = True,
    suggest_exclusions: bool = True,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Analyze genotype quality metrics and identify samples with high missingness.

    This function calculates per-sample statistics including missing data rate,
    heterozygosity, and mean genotype values. It can identify outlier samples
    based on various methods and optionally create visualizations.

    Args:
        genotypes: GenotypeArray of shape (n_sites, n_samples, ploidy)
        samples: Optional array of sample IDs. If None, uses indices.
        method: Outlier detection method. Options:
            - 'mad': Median Absolute Deviation (robust to outliers)
            - 'iqr': Interquartile Range
            - 'zscore': Standard deviations from mean
            - 'threshold': Simple cutoff value
        threshold: For method='threshold', samples with missing_rate above this
                  are flagged. Default 0.1 (10%) if None.
        n_std: For method='zscore', number of standard deviations
        n_mad: For method='mad', number of MADs from median
        iqr_multiplier: For method='iqr', multiplier for IQR range
        plot: Whether to create visualization
        return_stats: Whether to return the statistics DataFrame
        suggest_exclusions: Whether to suggest samples for exclusion
        verbose: Whether to print summary information

    Returns:
        Dictionary containing:
        - 'stats': DataFrame with per-sample statistics (if return_stats=True)
        - 'outliers': List of outlier sample IDs
        - 'outlier_indices': Array of outlier indices
        - 'plot': Matplotlib figure (if plot=True)
        - 'summary': Text summary of findings
    """
    # Calculate per-sample statistics
    stats_df = calculate_sample_stats(genotypes)

    # Add actual sample IDs if provided
    if samples is not None:
        if len(samples) != len(stats_df):
            raise ValueError(
                f"Number of samples ({len(samples)}) doesn't match genotype data ({len(stats_df)})"
            )
        stats_df["sampleID"] = samples

    # Detect outliers
    missing_rates = stats_df["missing_rate"].values
    outlier_mask = _apply_outlier_detection(
        missing_rates, method, threshold, n_std, n_mad, iqr_multiplier
    )

    # Mark outliers in DataFrame
    stats_df["is_outlier"] = outlier_mask

    # Get outlier information
    outlier_indices = np.where(outlier_mask)[0]
    outlier_samples = stats_df.loc[outlier_mask, "sampleID"].tolist()

    # Create summary
    summary = _create_summary_text(
        stats_df,
        outlier_mask,
        missing_rates,
        method,
        threshold,
        n_mad,
        iqr_multiplier,
        n_std,
        suggest_exclusions,
    )

    if verbose:
        print(summary)

    # Prepare results
    results = {
        "outliers": outlier_samples,
        "outlier_indices": outlier_indices,
        "summary": summary,
        "method": method,
        "n_outliers": len(outlier_samples),
    }

    if return_stats:
        results["stats"] = stats_df

    # Create plot if requested
    if plot:
        fig = _create_qc_plots(stats_df, outlier_mask, missing_rates)
        if fig is not None:
            results["plot"] = fig

    return results


def _calculate_n_snps(
    n_sites_original: int, n_snps: Optional[int], fraction: Optional[float]
) -> int:
    """Calculate target number of SNPs from either n_snps or fraction."""
    if n_snps is not None and fraction is not None:
        raise ValueError("Specify either n_snps or fraction, not both")

    if fraction is not None:
        if not 0.0 < fraction <= 1.0:
            raise ValueError("fraction must be between 0 and 1")
        n_snps = int(n_sites_original * fraction)

    if n_snps is None:
        raise ValueError("Must specify either n_snps or fraction")

    if n_snps > n_sites_original:
        raise ValueError(
            f"Requested {n_snps} SNPs but only {n_sites_original} available"
        )

    return n_snps


def _select_snp_indices(
    method: str, n_sites_original: int, n_snps: int, seed: Optional[int]
) -> np.ndarray:
    """Select SNP indices based on the specified method."""
    if method == "random":
        # Random selection
        if seed is not None:
            np.random.seed(seed)

        selected_indices = np.random.choice(n_sites_original, size=n_snps, replace=False)
        # Sort indices to maintain relative SNP order
        selected_indices = np.sort(selected_indices)

    elif method == "uniform":
        # Uniform spacing - select every Nth SNP
        step = n_sites_original / n_snps

        # Generate uniformly spaced indices
        selected_indices = np.round(np.arange(0, n_sites_original, step)).astype(int)

        # Ensure we don't exceed array bounds and have exactly n_snps
        selected_indices = selected_indices[selected_indices < n_sites_original]
        if len(selected_indices) > n_snps:
            selected_indices = selected_indices[:n_snps]

    else:
        raise ValueError(
            f"Unknown method '{method}'. Supported methods: 'random', 'uniform'"
        )

    return selected_indices


def _print_subsetting_summary(
    method: str,
    n_sites_original: int,
    n_snps: int,
    selected_indices: np.ndarray,
    seed: Optional[int],
):
    """Print summary of subsetting operation."""
    reduction_pct = (1 - n_snps / n_sites_original) * 100
    print("Genotype subsetting summary:")
    print(f"  Method: {method}")
    print(f"  Original SNPs: {n_sites_original:,}")
    print(f"  Selected SNPs: {len(selected_indices):,}")
    print(f"  Reduction: {reduction_pct:.1f}%")

    if method == "random" and seed is not None:
        print(f"  Random seed: {seed}")
    elif method == "uniform":
        actual_spacing = n_sites_original / len(selected_indices)
        print(f"  Average spacing: every {actual_spacing:.1f} SNPs")


def subset_genotypes(
    genotypes,
    method="random",
    n_snps=None,
    fraction=None,
    positions=None,
    chromosomes=None,
    seed=None,
    return_indices=False,
    verbose=True,
):
    """
    Subset genotypes using various strategies to reduce the number of SNPs.

    This function provides different methods to downsample SNPs from genotype data,
    which can be useful for reducing computational requirements or testing analyses
    with smaller datasets.

    Args:
        genotypes: GenotypeArray of shape (n_sites, n_samples, ploidy)
        method: Subsetting method. Options are 'random' (random selection of SNPs)
            or 'uniform' (select every Nth SNP for uniform spacing)
        n_snps: Target number of SNPs to retain (required for both methods)
        fraction: Alternative to n_snps - fraction of SNPs to retain (0.0-1.0)
        positions: Array of SNP positions (optional, for future methods)
        chromosomes: Array of chromosome IDs (optional, for future methods)
        seed: Random seed for reproducibility (only used for 'random' method)
        return_indices: If True, also return the indices of selected SNPs
        verbose: Whether to print information about the subsetting

    Returns:
        If return_indices=False:
            GenotypeArray: Subsetted genotype array
        If return_indices=True:
            tuple: (subsetted_genotypes, selected_indices)

    Raises:
        ValueError: If parameters are invalid or inconsistent

    Examples:
        >>> # Random subsetting to 100k SNPs
        >>> genotypes_subset = subset_genotypes(
        ...     genotypes,
        ...     method='random',
        ...     n_snps=100000,
        ...     seed=42
        ... )

        >>> # Uniform subsetting to 50% of SNPs
        >>> genotypes_subset = subset_genotypes(
        ...     genotypes,
        ...     method='uniform',
        ...     fraction=0.5
        ... )

        >>> # Get indices for reproducibility
        >>> geno_sub, indices = subset_genotypes(
        ...     genotypes,
        ...     method='random',
        ...     n_snps=50000,
        ...     return_indices=True
        ... )
    """
    # Get dimensions
    n_sites_original = genotypes.shape[0]

    # Calculate target number of SNPs
    n_snps = _calculate_n_snps(n_sites_original, n_snps, fraction)

    # Check if subsetting is needed
    if n_snps == n_sites_original:
        if verbose:
            print(f"No subsetting needed: requested {n_snps} SNPs equals available SNPs")
        if return_indices:
            return genotypes, np.arange(n_sites_original)
        return genotypes

    # Select indices based on method
    selected_indices = _select_snp_indices(method, n_sites_original, n_snps, seed)

    # Subset the genotypes
    genotypes_subset = genotypes[selected_indices]

    # Report results
    if verbose:
        _print_subsetting_summary(
            method, n_sites_original, n_snps, selected_indices, seed
        )

    if return_indices:
        return genotypes_subset, selected_indices
    return genotypes_subset
