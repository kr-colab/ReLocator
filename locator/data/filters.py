"""Centralized data filtering, imputation, and normalization utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from tqdm import tqdm


@dataclass
class FilterStats:
    """Track what was filtered and why."""

    n_samples_original: int
    n_samples_filtered: int
    n_snps_original: int
    n_snps_filtered: int
    mac_threshold: int
    samples_removed_na: list[str] = None
    n_biallelic_filtered: int = 0
    n_mac_filtered: int = 0
    n_random_subset: int = 0


@dataclass
class NormalizationParams:
    """Store normalization parameters for coordinates."""

    meanlong: float
    sdlong: float
    meanlat: float
    sdlat: float

    def apply(self, locs: np.ndarray) -> np.ndarray:
        """Apply normalization to coordinates."""
        return np.array(
            [
                [
                    (x[0] - self.meanlong) / self.sdlong,
                    (x[1] - self.meanlat) / self.sdlat,
                ]
                for x in locs
            ]
        )

    def reverse(self, normalized_locs: np.ndarray) -> np.ndarray:
        """Reverse normalization to get original coordinates."""
        return np.array(
            [
                [x[0] * self.sdlong + self.meanlong, x[1] * self.sdlat + self.meanlat]
                for x in normalized_locs
            ]
        )


def normalize_locs(
    locs: np.ndarray,
) -> Tuple[float, float, float, float, np.ndarray, np.ndarray]:
    """Normalize location coordinates.

    Args:
        locs: Array of shape (n_samples, 2) containing longitude and latitude

    Returns
    -------
        Tuple of (meanlong, sdlong, meanlat, sdlat, unnormedlocs, normedlocs)
    """
    # Only copy if we're going to modify the original
    unnormedlocs = locs.copy()
    meanlong = np.nanmean(locs[:, 0])
    sdlong = np.nanstd(locs[:, 0])
    meanlat = np.nanmean(locs[:, 1])
    sdlat = np.nanstd(locs[:, 1])

    # Create new array for normalized locations
    normedlocs = np.empty_like(locs, dtype=np.float64)
    normedlocs[:, 0] = (locs[:, 0] - meanlong) / sdlong
    normedlocs[:, 1] = (locs[:, 1] - meanlat) / sdlat

    return meanlong, sdlong, meanlat, sdlat, unnormedlocs, normedlocs


def normalize_locs_params(
    locs: np.ndarray,
) -> Tuple[NormalizationParams, np.ndarray, np.ndarray]:
    """Normalize location coordinates and return parameters object.

    Args:
        locs: Array of shape (n_samples, 2) containing longitude and latitude

    Returns
    -------
        Tuple of (NormalizationParams, unnormedlocs, normedlocs)
    """
    unnormedlocs = locs.copy()
    meanlong = np.nanmean(locs[:, 0])
    sdlong = np.nanstd(locs[:, 0])
    meanlat = np.nanmean(locs[:, 1])
    sdlat = np.nanstd(locs[:, 1])

    params = NormalizationParams(meanlong, sdlong, meanlat, sdlat)
    normedlocs = params.apply(locs)

    return params, unnormedlocs, normedlocs


def impute_missing(genotypes) -> np.ndarray:
    """Replace missing data with binomial draws from allele frequency.

    Args:
        genotypes: GenotypeArray with missing data

    Returns
    -------
        Allele counts array with imputed values
    """
    print("imputing missing data")
    dc = genotypes.count_alleles()[:, 1]
    ac = genotypes.to_allele_counts()[:, :, 1]
    missingness = genotypes.is_missing()
    ninds = np.array([np.sum(x) for x in ~missingness])
    af = np.array([dc[x] / (2 * ninds[x]) for x in range(len(ninds))])

    for i in tqdm(range(np.shape(ac)[0])):
        for j in range(np.shape(ac)[1]):
            if missingness[i, j]:
                ac[i, j] = np.random.binomial(2, af[i])
    return ac


def is_dosage_matrix(genotypes) -> bool:
    """Detect a continuous-dosage matrix vs an allel.GenotypeArray.

    GL-derived inputs flow through `_load_from_matrix` as 2D float ndarrays
    of shape (n_sites, n_samples); hard-call inputs are allel.GenotypeArray
    of shape (n_sites, n_samples, ploidy). Downstream filtering, training,
    and prediction code dispatches on this distinction.
    """
    return (
        isinstance(genotypes, np.ndarray)
        and genotypes.ndim == 2
        and np.issubdtype(genotypes.dtype, np.floating)
    )


def filter_dosage_matrix(
    dosage: np.ndarray,
    min_mac: int = 2,
    max_snps: Optional[int] = None,
) -> np.ndarray:
    """MAC and max_snps filters for continuous dosage input.

    Mean dosage at a site is in [0, 2]; minor-allele frequency is
    ``min(mean, 2 - mean) / 2``, and the implied minor-allele count is
    ``MAF * 2 * n_samples``. Sites below ``min_mac`` are dropped.

    Imputation must happen upstream: NaN values raise ``ValueError`` rather
    than silently dropping every site (a NaN mean propagates through the MAC
    comparison and would otherwise mask all sites out).

    Args:
        dosage: ``(n_sites, n_samples)`` float ndarray with values in [0, 2].
        min_mac: Minimum implied minor-allele count for a site to be kept.
        max_snps: Optional cap on retained sites (random subset).

    Returns
    -------
        Contiguous float32 ndarray of filtered sites.
    """
    if np.isnan(dosage).any():
        raise ValueError(
            "dosage matrix contains NaN values; impute upstream "
            "(e.g. via gl_to_locator.py site-mean fill) before passing "
            "to ReLocator."
        )
    n_sites, n_samples = dosage.shape
    mean_dosage = dosage.mean(axis=1)
    minor_freq = np.minimum(mean_dosage, 2.0 - mean_dosage) / 2.0
    implied_mac = minor_freq * 2.0 * n_samples
    mask = implied_mac >= float(min_mac)
    dosage = dosage[mask, :]
    if max_snps is not None and max_snps < dosage.shape[0]:
        idx = np.random.choice(dosage.shape[0], max_snps, replace=False)
        dosage = dosage[np.sort(idx), :]
    return np.ascontiguousarray(dosage, dtype=np.float32)


def filter_snps(
    genotypes,
    min_mac: int = 1,
    max_snps: Optional[int] = None,
    impute: bool = False,
    verbose: bool = False,
) -> Tuple[np.ndarray, FilterStats]:
    """Filter SNPs based on criteria and return statistics.

    Args:
        genotypes: GenotypeArray to filter
        min_mac: Minimum minor allele count for filtering
        max_snps: Maximum number of SNPs to retain
        impute: Whether to impute missing data
        verbose: Whether to print progress messages

    Returns
    -------
        Tuple of (filtered allele counts array, FilterStats)
    """
    if verbose:
        print("filtering SNPs")

    # Initialize stats
    n_snps_original = genotypes.shape[0]
    n_samples_original = genotypes.shape[1]
    n_biallelic_filtered = 0
    n_mac_filtered = 0
    n_random_subset = 0

    # Count alleles once and reuse
    allele_counts = genotypes.count_alleles()

    # Filter for biallelic sites
    biallel = allele_counts.is_biallelic()
    n_biallelic_filtered = n_snps_original - np.sum(biallel)

    # Combine biallelic and MAC filters if needed
    if min_mac > 1:
        # Get derived allele counts from already computed allele_counts
        derived_counts = allele_counts[biallel, 1]
        mac_filter = derived_counts >= min_mac
        n_mac_filtered = len(mac_filter) - np.sum(mac_filter)
        # Combine filters
        combined_filter = np.zeros(n_snps_original, dtype=bool)
        combined_filter[biallel] = mac_filter
        genotypes = genotypes[combined_filter, :, :]
    else:
        genotypes = genotypes[biallel, :, :]

    # Impute or convert to allele counts
    if impute:
        ac = impute_missing(genotypes)
    else:
        ac = genotypes.to_allele_counts()[:, :, 1]

    # Random subset if requested
    if max_snps is not None and max_snps < ac.shape[0]:
        n_random_subset = ac.shape[0] - max_snps
        ac = ac[np.random.choice(range(ac.shape[0]), max_snps, replace=False), :]

    # Create stats
    stats = FilterStats(
        n_samples_original=n_samples_original,
        n_samples_filtered=ac.shape[1],
        n_snps_original=n_snps_original,
        n_snps_filtered=ac.shape[0],
        mac_threshold=min_mac,
        n_biallelic_filtered=n_biallelic_filtered,
        n_mac_filtered=n_mac_filtered,
        n_random_subset=n_random_subset,
    )

    if verbose:
        print(f"filtered {stats.n_samples_filtered} individual genotypes")
        print(f"{stats.n_snps_filtered} SNPs after filtering")
        print(f"  - {stats.n_biallelic_filtered} non-biallelic sites removed")
        print(f"  - {stats.n_mac_filtered} sites with MAC < {min_mac} removed")
        if stats.n_random_subset > 0:
            print(f"  - {stats.n_random_subset} sites removed by random subsampling")
        print("\n")

    return ac, stats


# Backward compatibility wrapper
def filter_snps_legacy(
    genotypes,
    min_mac: int = 1,
    max_snps: Optional[int] = None,
    impute: bool = False,
    verbose: bool = False,
) -> np.ndarray:
    """Legacy wrapper for filter_snps that only returns allele counts."""
    ac, _ = filter_snps(genotypes, min_mac, max_snps, impute, verbose)
    return ac
