"""Numba-accelerated kernels for diploid genotype counts."""

from __future__ import annotations

import numpy as np

try:
    import numba

    HAVE_NUMBA = True
except ImportError:  # pragma: no cover - exercised only without numba installed
    HAVE_NUMBA = False


if HAVE_NUMBA:

    @numba.njit(parallel=True, cache=True, boundscheck=False)
    def _count_ref_alt_diploid(gt: np.ndarray):
        n_snps = gt.shape[0]
        n_samples = gt.shape[1]
        ref = np.zeros(n_snps, dtype=np.int32)
        alt = np.zeros(n_snps, dtype=np.int32)
        has_higher = np.zeros(n_snps, dtype=np.bool_)
        for i in numba.prange(n_snps):
            r = 0
            a = 0
            h = False
            for j in range(n_samples):
                v0 = gt[i, j, 0]
                v1 = gt[i, j, 1]
                if v0 == 0:
                    r += 1
                elif v0 == 1:
                    a += 1
                elif v0 >= 2:
                    h = True
                if v1 == 0:
                    r += 1
                elif v1 == 1:
                    a += 1
                elif v1 >= 2:
                    h = True
            ref[i] = r
            alt[i] = a
            has_higher[i] = h
        return ref, alt, has_higher


def count_ref_alt_diploid(gt: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Parallel ref/alt counts per SNP for a diploid genotype array.

    Replaces ``allel.GenotypeArray.count_alleles()`` for the common
    diploid case. scikit-allel's kernel is Cython but single-threaded; on
    multi-core hosts this kernel uses ``numba.prange`` along the SNP axis
    and is typically an order of magnitude faster on WGS-scale inputs.

    Parameters
    ----------
    gt : ndarray of shape (n_snps, n_samples, 2), int8 (or compatible)
        Genotype codes: -1 = missing, 0 = ref, 1 = alt, >= 2 = higher
        alleles (tri-allelic or more).

    Returns
    -------
    ref : (n_snps,) int32
        Per-SNP count of ref (allele 0) calls. Missing genotypes contribute
        nothing.
    alt : (n_snps,) int32
        Per-SNP count of alt (allele 1) calls.
    has_higher : (n_snps,) bool
        True for any SNP where an allele >= 2 was observed.

    Notes
    -----
    For ploidy != 2 or when numba is unavailable, the caller should use
    scikit-allel's ``count_alleles()`` instead.
    """
    if not HAVE_NUMBA:
        raise RuntimeError(
            "count_ref_alt_diploid requires numba; install numba or use the "
            "scikit-allel fallback path."
        )
    if gt.ndim != 3 or gt.shape[2] != 2:
        raise ValueError(
            f"count_ref_alt_diploid expects shape (n_snps, n_samples, 2); got {gt.shape}"
        )
    # numba dispatches on dtype, so non-int8 inputs would trigger a separate
    # specialization; non-contiguous strides also defeat prange's sequential
    # access pattern.
    if gt.dtype != np.int8:
        gt = gt.astype(np.int8, copy=False)
    if not gt.flags["C_CONTIGUOUS"]:
        gt = np.ascontiguousarray(gt)
    return _count_ref_alt_diploid(gt)
