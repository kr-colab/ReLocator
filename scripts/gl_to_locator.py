#!/usr/bin/env python3
"""
gl_to_locator.py

Convert ANGSD -doGlf 2 beagle.gz output to a feature matrix compatible with
ReLocator's --geno tab-delimited input.

Run with --help for full option descriptions and usage examples.

Two output modes are supported via --gl_mode:

  dosage  (default)
      One column per site, value = expected dosage under a flat prior:
          E[dosage] = P(AB) + 2 * P(BB)
      Missing samples (max(GL) < --gl_missing_threshold) are imputed with
      site-mean dosage across non-missing samples.

  full_gl
      Three columns per site (suffixed _AA, _AB, _BB), each holding the
      normalized GL probability for that genotype. Preserves genotype
      uncertainty information that the dosage scalar collapses. Missing
      samples are imputed with the site-mean GL triplet.

Output is tab-delimited with 'sampleID' as the first column. Rows are samples,
columns are sites (or site_AA / site_AB / site_BB triplets in full_gl mode).
"""

import argparse
import gzip
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(
        description="Convert ANGSD beagle GL file to ReLocator feature matrix."
    )
    p.add_argument(
        "--beagle",
        required=True,
        help="ANGSD output.beagle.gz file (from -doGlf 2)",
    )
    p.add_argument(
        "--bam_list",
        required=True,
        help=(
            "BAM file list used in the ANGSD run (one BAM path per line). "
            "Sample IDs are derived from Path(bam).stem. "
            "Order must match the ANGSD run exactly."
        ),
    )
    p.add_argument(
        "--out",
        required=True,
        help="Output tab-delimited feature file for ReLocator --geno",
    )
    p.add_argument(
        "--gl_mode",
        choices=("dosage", "full_gl"),
        default="dosage",
        help=(
            "Output encoding. 'dosage' (default): one column per site holding "
            "the expected allele dosage. 'full_gl': three columns per site "
            "(P_AA, P_AB, P_BB) preserving genotype uncertainty."
        ),
    )
    p.add_argument(
        "--min_maf",
        type=float,
        default=0.01,
        help=(
            "Drop sites with mean dosage MAF below this value. "
            "MAF = min(mean_dosage, 2 - mean_dosage) / 2. "
            "Default: 0.01"
        ),
    )
    p.add_argument(
        "--max_missing_frac",
        type=float,
        default=0.10,
        help=(
            "Drop sites where the fraction of effectively missing samples "
            "exceeds this value. Default: 0.10"
        ),
    )
    p.add_argument(
        "--gl_missing_threshold",
        type=float,
        default=0.40,
        help=(
            "A sample at a site is treated as missing if max(GL_AA, GL_AB, GL_BB) "
            "is below this value (near-uniform = no sequencing information). "
            "Default: 0.40"
        ),
    )
    p.add_argument(
        "--sample_data",
        default=None,
        help=(
            "Optional: ReLocator sample_data.txt file. If provided, derived "
            "sample IDs are cross-checked against the sampleID column and any "
            "mismatches are reported before writing output."
        ),
    )
    return p.parse_args()


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
        raise ValueError(
            f"Failed to parse beagle GL values as float32. Error: {e}"
        )

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


def cross_check_sample_ids(derived_ids, sample_data_path):
    sd = pd.read_csv(sample_data_path, sep="\t", dtype=str)
    if "sampleID" not in sd.columns:
        print(
            "WARNING: sample_data file has no 'sampleID' column; skipping cross-check.",
            file=sys.stderr,
        )
        return

    sd_ids = set(sd["sampleID"].values)
    derived_set = set(derived_ids)
    in_derived_not_sd = derived_set - sd_ids
    in_sd_not_derived = sd_ids - derived_set

    if in_derived_not_sd:
        print(
            f"WARNING: {len(in_derived_not_sd)} sample IDs derived from BAM list "
            f"are NOT in sample_data.txt:\n  "
            + "\n  ".join(sorted(in_derived_not_sd)),
            file=sys.stderr,
        )
    if in_sd_not_derived:
        print(
            f"WARNING: {len(in_sd_not_derived)} sample IDs in sample_data.txt "
            f"are NOT in BAM list:\n  "
            + "\n  ".join(sorted(in_sd_not_derived)),
            file=sys.stderr,
        )
    if not in_derived_not_sd and not in_sd_not_derived:
        print(f"  Sample ID cross-check: all {len(derived_ids)} IDs match.", flush=True)


def write_dosage(out_path, dosage, sample_ids, markers_kept):
    """Write (n_samples, n_sites) dosage matrix as TSV.

    Values are continuous expected dosage in [0, 2]. ReLocator's --matrix
    loader was patched (loaders.py + training.py) to detect float dtype and
    feed continuous dosage straight to the network, bypassing the
    GenotypeArray round trip used by hard-call inputs.
    """
    df = pd.DataFrame(
        dosage.T.astype(np.float32),
        index=sample_ids,
        columns=[f"site_{m}" for m in markers_kept],
    )
    df.index.name = "sampleID"
    df.to_csv(out_path, sep="\t")


def write_full_gl(out_path, gl, sample_ids, markers_kept):
    """Write GL triplets as (n_samples, 3 * n_sites) matrix.

    For each kept site `m`, three consecutive columns: site_{m}_AA, _AB, _BB.
    """
    n_sites = gl.shape[0]
    n_samples = gl.shape[1]
    flat = gl.transpose(1, 0, 2).reshape(n_samples, n_sites * 3)
    cols = []
    for m in markers_kept:
        cols.extend([f"site_{m}_AA", f"site_{m}_AB", f"site_{m}_BB"])
    df = pd.DataFrame(flat.astype(np.float32), index=sample_ids, columns=cols)
    df.index.name = "sampleID"
    df.to_csv(out_path, sep="\t")


def main():
    args = parse_args()

    sample_ids = sample_ids_from_bam_list(args.bam_list)
    n_samples = len(sample_ids)
    print(f"BAM list: {n_samples} samples", flush=True)

    if args.sample_data:
        cross_check_sample_ids(sample_ids, args.sample_data)

    markers, gl_flat = load_beagle(args.beagle)
    n_sites = len(markers)
    validate_dimensions(gl_flat, n_samples, args.beagle, args.bam_list)

    gl = reshape_gl(gl_flat, n_samples)
    del gl_flat

    missing_mask = detect_missing(gl, args.gl_missing_threshold)
    n_missing = int(missing_mask.sum())
    print(
        f"Missing genotypes: {n_missing} / {n_sites * n_samples} "
        f"({100.0 * n_missing / (n_sites * n_samples):.1f}%) "
        f"[threshold: max(GL) < {args.gl_missing_threshold}]",
        flush=True,
    )

    # Always compute dosage — used for site-level MAF filtering even in full_gl mode.
    dosage = expected_dosage(gl).copy()
    dosage = impute_dosage_with_site_mean(dosage, missing_mask)

    keep, reasons = filter_sites(
        dosage, missing_mask, args.min_maf, args.max_missing_frac
    )
    print(
        f"Site filtering:\n"
        f"  Removed (missing rate > {args.max_missing_frac}): {reasons['max_missing_frac']}\n"
        f"  Removed (MAF < {args.min_maf}):                   {reasons['min_maf']}\n"
        f"  Total removed:                                     {reasons['total_removed']}\n"
        f"  Sites retained:                                    {reasons['total_kept']}",
        flush=True,
    )
    if reasons["total_kept"] == 0:
        raise ValueError(
            "No sites passed filters. Check --min_maf and --max_missing_frac."
        )

    markers_kept = [m for m, k in zip(markers, keep, strict=True) if k]

    if args.gl_mode == "dosage":
        out_arr = dosage[keep]  # (n_sites_kept, n_samples)
        write_dosage(args.out, out_arr, sample_ids, markers_kept)
        n_features = out_arr.shape[0]
    else:
        gl_kept = gl[keep].copy()  # (n_sites_kept, n_samples, 3)
        gl_kept = impute_gl_with_site_mean(gl_kept, missing_mask[keep])
        write_full_gl(args.out, gl_kept, sample_ids, markers_kept)
        n_features = gl_kept.shape[0] * 3

    print(
        f"Written: {args.out}\n"
        f"  Mode: {args.gl_mode}\n"
        f"  Shape: {n_samples} samples × {n_features} features "
        f"({len(markers_kept)} sites)",
        flush=True,
    )


if __name__ == "__main__":
    main()
