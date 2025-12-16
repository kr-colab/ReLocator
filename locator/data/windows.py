"""
Utilities for genomic window generation with chromosome awareness.
"""

import warnings
from typing import Any, Dict, List, Optional, TypedDict

import numpy as np


class WindowSpec(TypedDict):
    """Specification for a genomic window."""

    start: int  # Start position (inclusive)
    stop: int  # Stop position (exclusive)
    chromosome: Optional[str]  # Chromosome name (None if not respecting chromosomes)
    indices: np.ndarray  # Boolean mask for SNPs in this window
    label: str  # Label for DataFrame columns (e.g., "chr1_pos1000000")
    n_snps: int  # Number of SNPs in window


def sort_chromosomes(chroms: List[str]) -> List[str]:
    """
    Sort chromosomes in natural order (1,2,...,10,11,...,X,Y,MT).

    Args:
        chroms: List of chromosome names

    Returns
    -------
        List of sorted chromosome names
    """
    numeric = []
    alpha = []

    for c in chroms:
        # Handle various naming conventions
        c_clean = str(c).replace("chr", "").replace("Chr", "").replace("CHR", "")
        try:
            numeric.append((int(c_clean), c))
        except ValueError:
            alpha.append(c)

    # Sort numeric chromosomes numerically
    numeric.sort(key=lambda x: x[0])
    numeric_sorted = [x[1] for x in numeric]

    # Sort non-numeric alphabetically, with special ordering for common ones
    special_order = {"X": 1000, "Y": 1001, "MT": 1002, "M": 1002}
    alpha.sort(key=lambda x: special_order.get(x.upper(), ord(x[0])))

    return numeric_sorted + alpha


def generate_genomic_windows(  # noqa: C901
    positions: np.ndarray,
    chromosomes: Optional[np.ndarray] = None,
    window_start: int = 0,
    window_size: int = 500000,
    window_stop: Optional[int] = None,
    respect_chromosomes: bool = True,
    min_snps_per_window: int = 1,
    verbose: bool = False,
) -> List[Dict[str, Any]]:
    """
    Generate window specifications respecting chromosome boundaries.

    Args:
        positions: Array of SNP positions
        chromosomes: Array of chromosome names for each SNP (optional)
        window_start: Start position for windows (default: 0)
        window_size: Size of windows in base pairs (default: 500kb)
        window_stop: Stop position for windows (default: max position)
        respect_chromosomes: Whether to respect chromosome boundaries (default: True)
        min_snps_per_window: Minimum SNPs required per window (default: 1)
        verbose: Whether to print progress information

    Returns
    -------
        List of window dictionaries, each containing:
        - 'start': Start position
        - 'stop': Stop position
        - 'chromosome': Chromosome name (if respect_chromosomes=True)
        - 'indices': Boolean mask for SNPs in this window
        - 'label': Label for column naming
        - 'n_snps': Number of SNPs in window

    Raises
    ------
        ValueError: If inputs are invalid
    """
    # Validate inputs
    if len(positions) == 0:
        raise ValueError("No SNP positions provided")

    if chromosomes is not None and len(chromosomes) != len(positions):
        raise ValueError(
            f"Length mismatch: {len(positions)} positions vs {len(chromosomes)} chromosomes"
        )

    if window_size <= 0:
        raise ValueError(f"Window size must be positive, got {window_size}")

    if window_start < 0:
        raise ValueError(f"Window start must be non-negative, got {window_start}")

    # Ensure numpy arrays
    positions = np.asarray(positions)
    if chromosomes is not None:
        chromosomes = np.asarray(chromosomes)

    windows = []

    if chromosomes is None or not respect_chromosomes:
        # Legacy behavior - simple numeric windows
        if window_stop is None:
            window_stop = int(np.max(positions))

        # Issue warning if multiple chromosomes detected
        if chromosomes is not None and not respect_chromosomes:
            unique_chroms = np.unique(chromosomes)
            if len(unique_chroms) > 1:
                warnings.warn(
                    f"Multiple chromosomes detected ({len(unique_chroms)}) but "
                    f"respect_chromosomes=False. Windows may span chromosome boundaries. "
                    f"Set respect_chromosomes=True to analyze chromosomes separately."
                )

        # Generate windows based on position values only
        current_start = int(window_start)
        window_count = 0

        while current_start < window_stop:
            current_stop = min(current_start + int(window_size), window_stop)

            # Create mask for SNPs in this window
            mask = (positions >= current_start) & (positions < current_stop)
            n_snps = int(np.sum(mask))

            # Only include windows with sufficient SNPs
            if n_snps >= min_snps_per_window:
                windows.append(
                    {
                        "start": current_start,
                        "stop": current_stop,
                        "chromosome": None,
                        "indices": mask,
                        "label": f"pos{current_start}",
                        "n_snps": n_snps,
                    }
                )
                window_count += 1

            current_start += int(window_size)

        if verbose:
            print(f"Generated {window_count} windows (ignoring chromosomes)")

    else:
        # Chromosome-aware window generation
        unique_chroms = np.unique(chromosomes)
        sorted_chroms = sort_chromosomes(list(unique_chroms))

        if verbose:
            print(
                f"Processing {len(sorted_chroms)} chromosomes: {', '.join(sorted_chroms[:5])}"
            )
            if len(sorted_chroms) > 5:
                print(f"  ... and {len(sorted_chroms) - 5} more")

        for chrom in sorted_chroms:
            # Get positions for this chromosome
            chrom_mask = chromosomes == chrom
            chrom_positions = positions[chrom_mask]

            if len(chrom_positions) == 0:
                continue

            # Determine range for this chromosome
            chrom_min = int(np.min(chrom_positions))
            chrom_max = int(np.max(chrom_positions))

            # Apply user-specified bounds if they overlap with chromosome
            if window_stop is not None and window_stop <= chrom_min:
                continue  # Skip this chromosome entirely
            if window_start >= chrom_max:
                continue  # Skip this chromosome entirely

            # Determine actual start/stop for this chromosome
            actual_start = max(int(window_start), chrom_min)
            actual_stop = min(
                int(window_stop) if window_stop else chrom_max + 1, chrom_max + 1
            )

            # Generate windows within this chromosome
            current_start = actual_start
            chrom_window_count = 0

            while current_start < actual_stop:
                current_stop = min(current_start + int(window_size), actual_stop)

                # Create mask for SNPs in this window AND chromosome
                mask = (
                    (positions >= current_start)
                    & (positions < current_stop)
                    & (chromosomes == chrom)
                )
                n_snps = int(np.sum(mask))

                # Only include windows with sufficient SNPs
                if n_snps >= min_snps_per_window:
                    # Clean chromosome name for label
                    chrom_clean = str(chrom).replace("chr", "").replace("Chr", "")

                    windows.append(
                        {
                            "start": current_start,
                            "stop": current_stop,
                            "chromosome": str(chrom),  # Ensure string type
                            "indices": mask,
                            "label": f"chr{chrom_clean}_pos{current_start}",
                            "n_snps": n_snps,
                        }
                    )
                    chrom_window_count += 1

                current_start += int(window_size)

            if verbose and chrom_window_count > 0:
                print(f"  Chromosome {chrom}: {chrom_window_count} windows")

    if verbose:
        total_snps = sum(w["n_snps"] for w in windows)
        print(f"Total: {len(windows)} windows covering {total_snps} SNP observations")

    return windows
