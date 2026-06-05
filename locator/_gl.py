"""Genotype-likelihood (ANGSD beagle) parsing and dosage encoding helpers.

These are internal helpers used by `locator.loaders.DataLoaderMixin._load_from_gl`.
The public entry points are `loc.load_genotypes(gl=..., bam_list=..., gl_mode=...)`
and the `locator --gl ... --bam_list ... --gl_mode {dosage,full_gl}` CLI flags.
"""

from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np


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
    rows = []
    markers = []
    open_fn = gzip.open if str(beagle_path).endswith(".gz") else open
    with open_fn(beagle_path, "rt") as fh:
        fh.readline()  # discard header
        for line in fh:
            fields = line.rstrip("\n").split("\t")
            markers.append(fields[0])
            rows.append(fields[3:])  # skip marker, allele1, allele2

    if not rows:
        raise ValueError(f"No data rows found in beagle file: {beagle_path}")

    try:
        gl_flat = np.array(rows, dtype=np.float32)
    except ValueError as e:
        raise ValueError(f"Failed to parse beagle GL values as float32. Error: {e}")

    print(f"  Loaded {len(markers)} sites", flush=True)
    return markers, gl_flat


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
    for i in range(dosage.shape[0]):
        mask = missing_mask[i]
        if not mask.any():
            continue
        present = dosage[i, ~mask]
        dosage[i, mask] = present.mean() if present.size > 0 else 0.0
    return dosage


def impute_gl_with_site_mean(gl, missing_mask):
    """Impute missing GL triplets with site-mean triplet across non-missing samples."""
    for i in range(gl.shape[0]):
        mask = missing_mask[i]
        if not mask.any():
            continue
        present = gl[i, ~mask, :]
        if present.size == 0:
            gl[i, mask, :] = np.array([1.0 / 3, 1.0 / 3, 1.0 / 3], dtype=gl.dtype)
        else:
            gl[i, mask, :] = present.mean(axis=0)
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
