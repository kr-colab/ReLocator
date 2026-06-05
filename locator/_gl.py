"""Genotype-likelihood (ANGSD beagle) parsing and dosage encoding helpers.

These are internal helpers used by `locator.loaders.DataLoaderMixin._load_from_gl`.
The public entry points are `loc.load_genotypes(gl=..., bam_list=..., gl_mode=...)`
and the `locator --gl ... --bam_list ... --gl_mode {dosage,full_gl}` CLI flags.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def sample_ids_from_bam_list(bam_list_path):
    ids = []
    with open(bam_list_path) as fh:
        for line in fh:
            bam = line.strip()
            if bam:
                ids.append(Path(bam).stem)
    if not ids:
        raise ValueError(f"No BAM paths found in: {bam_list_path}")
    return ids


def load_beagle(beagle_path):
    """Load beagle.gz file.

    Returns
    -------
    markers : list of str, length n_sites
    gl_flat : np.ndarray float32, shape (n_sites, n_samples * 3)
              columns are GL triplets [AA, AB, BB] per sample.
    """
    print(f"Loading beagle file: {beagle_path}", flush=True)
    # Columns are read positionally (header skipped) because the beagle header
    # repeats each sample ID three times, which would otherwise need de-duping.
    # Compression is inferred from the path suffix, so .gz and plain files both
    # work without an explicit branch.
    try:
        df = pd.read_csv(beagle_path, sep="\t", header=None, skiprows=1)
    except pd.errors.EmptyDataError:
        raise ValueError(f"No data rows found in beagle file: {beagle_path}")
    if df.empty:
        raise ValueError(f"No data rows found in beagle file: {beagle_path}")

    markers = df.iloc[:, 0].astype(str).tolist()
    try:
        # skip marker, allele1, allele2; the rest are GL triplets per sample.
        gl_flat = df.iloc[:, 3:].to_numpy(dtype=np.float32)
    except (ValueError, TypeError) as e:
        raise ValueError(f"Failed to parse beagle GL values as float32. Error: {e}")

    print(f"  Loaded {len(markers)} sites", flush=True)
    return markers, gl_flat


def parse_markers(markers):
    """Parse beagle ``chrom_pos`` marker strings into chromosomes and positions.

    ANGSD beagle ``marker`` entries encode the genomic site as
    ``<chrom>_<pos>`` (e.g. ``chr01_2039``). Chromosome names may themselves
    contain underscores (e.g. ``scaffold_12_2039``), so the split is on the
    LAST underscore only.

    Returns
    -------
    chromosomes : np.ndarray (object), length n_sites
    positions : np.ndarray (int64), length n_sites
    """
    chroms = []
    positions = []
    bad = []
    for m in markers:
        chrom, sep, pos_str = m.rpartition("_")
        if not sep:
            bad.append(m)
            continue
        try:
            positions.append(int(pos_str))
        except ValueError:
            bad.append(m)
            continue
        chroms.append(chrom)
    if bad:
        raise ValueError(
            f"Could not parse {len(bad)} beagle marker(s) as 'chrom_pos'; "
            f"first offenders: {bad[:10]}"
        )
    return np.array(chroms, dtype=object), np.array(positions, dtype=np.int64)


def validate_dimensions(gl_flat, n_samples, beagle_path, bam_list_path):
    expected_cols = n_samples * 3
    if gl_flat.shape[1] != expected_cols:
        raise ValueError(
            f"Beagle file has {gl_flat.shape[1]} GL columns but expected "
            f"{expected_cols} ({n_samples} samples × 3 GL values per sample).\n"
            f"  Beagle: {beagle_path}\n"
            f"  BAM list: {bam_list_path}\n"
            f"Ensure --bam_list is the same file used in the ANGSD run."
        )


def reshape_gl(gl_flat, n_samples):
    """Reshape (n_sites, n_samples*3) to (n_sites, n_samples, 3)."""
    return gl_flat.reshape(gl_flat.shape[0], n_samples, 3)


def expected_dosage(gl):
    """E[dosage] = P(AB) + 2 * P(BB) under a flat prior."""
    return gl[:, :, 1] + 2.0 * gl[:, :, 2]


def detect_missing(gl, gl_missing_threshold):
    """Return bool mask (n_sites, n_samples), True = sample missing at site."""
    return gl.max(axis=2) < gl_missing_threshold


def impute_dosage_with_site_mean(dosage, missing_mask):
    """Impute missing dosage values with site-mean across non-missing samples."""
    present = ~missing_mask  # (n_sites, n_samples)
    counts = present.sum(axis=1)  # (n_sites,)
    # einsum zeroes the missing entries' contribution, so only present values
    # enter the per-site sum (matching present.mean()).
    sums = np.einsum("ij,ij->i", dosage, present.astype(dosage.dtype))
    site_mean = np.zeros(dosage.shape[0], dtype=dosage.dtype)
    nz = counts > 0
    site_mean[nz] = sums[nz] / counts[nz]
    # sites with no present sample keep site_mean = 0.0 (the loop's fallback).
    dosage[missing_mask] = np.broadcast_to(site_mean[:, None], dosage.shape)[
        missing_mask
    ]
    return dosage


def impute_gl_with_site_mean(gl, missing_mask):
    """Impute missing GL triplets with site-mean triplet across non-missing samples."""
    present = ~missing_mask  # (n_sites, n_samples)
    counts = present.sum(axis=1)  # (n_sites,)
    sums = np.einsum("ijk,ij->ik", gl, present.astype(gl.dtype))  # (n_sites, 3)
    # initialize to the uniform fallback so all-missing sites need no special case.
    site_mean = np.full((gl.shape[0], gl.shape[2]), 1.0 / 3, dtype=gl.dtype)
    nz = counts > 0
    site_mean[nz] = sums[nz] / counts[nz][:, None]
    mask3d = np.broadcast_to(missing_mask[:, :, None], gl.shape)
    gl[mask3d] = np.broadcast_to(site_mean[:, None, :], gl.shape)[mask3d]
    return gl


def filter_sites(dosage, missing_mask, min_maf, max_missing_frac):
    """Apply site-level filters using imputed dosage and pre-imputation missing mask."""
    missing_frac = missing_mask.mean(axis=1)
    pass_missing = missing_frac <= max_missing_frac

    mean_dos = dosage.mean(axis=1)
    maf = np.minimum(mean_dos, 2.0 - mean_dos) / 2.0
    pass_maf = maf >= min_maf

    keep = pass_missing & pass_maf
    reasons = {
        "max_missing_frac": int((~pass_missing).sum()),
        "min_maf": int((~pass_maf & pass_missing).sum()),
        "total_removed": int((~keep).sum()),
        "total_kept": int(keep.sum()),
    }
    return keep, reasons
