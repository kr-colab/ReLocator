"""Microsatellite parsing and dosage encoding helpers.

These are internal helpers used by `locator.loaders.DataLoaderMixin._load_from_microsat`.
The public entry points are `loc.load_genotypes(microsat=...)` and the
`locator --microsat ...` CLI flag.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Literal

import numpy as np
import pandas as pd

MISSING_STRINGS = {"NA", "NAN", ".", "", "0,0", "0/0"}


def parse_genotype(cell: str) -> tuple[int | None, int | None]:
    """Parse a diploid genotype cell.

    Accepts: '12,14'  '12/14'  '12 14'  '12|14'  '14'  'NA'  '.'  ''
    Single value (no separator) is interpreted as homozygote shorthand.
    Returns ``(int, int)`` or ``(None, None)`` for missing.
    """
    cell = str(cell).strip()
    if cell.upper() in MISSING_STRINGS:
        return (None, None)

    for sep in (",", "/", " ", "|"):
        if sep in cell:
            parts = cell.split(sep, 1)
            try:
                return (int(parts[0].strip()), int(parts[1].strip()))
            except (ValueError, IndexError):
                return (None, None)

    try:
        v = int(cell)
        return (v, v)
    except ValueError:
        return (None, None)


def build_allele_catalog(
    df: pd.DataFrame,
    loci: list[str],
    min_allele_freq: float,
) -> dict[str, list[int]]:
    """For each locus, return the sorted list of alleles passing MAF.

    Fully-missing loci (no parseable genotypes in any sample) map to an
    empty list; the caller is responsible for dropping those before
    encoding.
    """
    catalog: dict[str, list[int]] = {}

    for locus in loci:
        allele_counts: dict[int, int] = defaultdict(int)
        total_alleles = 0

        for val in df[locus]:
            a1, a2 = parse_genotype(val)
            if a1 is None:
                continue
            allele_counts[a1] += 1
            allele_counts[a2] += 1
            total_alleles += 2

        if total_alleles == 0:
            catalog[locus] = []
            continue

        catalog[locus] = sorted(
            a
            for a, cnt in allele_counts.items()
            if cnt / total_alleles >= min_allele_freq
        )

    return catalog


def encode_dosage_block(
    df: pd.DataFrame,
    active_loci: list[str],
    catalog: dict[str, list[int]],
) -> tuple[np.ndarray, list[str]]:
    """Build the (n_samples, K) dosage block. Missing genotypes get site-mean imputation.

    Returns ``(matrix, column_names)`` where each column is named
    ``dosage_<locus>_<allele>``. K = sum(len(catalog[l]) for l in active_loci).
    Alleles observed in df but not in ``catalog[l]`` (e.g. dropped by MAF) are
    silently ignored — the dosage at that allele's column simply stays 0.
    """
    allele_index = {
        loc: {a: i for i, a in enumerate(catalog[loc])}
        for loc in active_loci
        if catalog[loc]
    }
    col_names: list[str] = []
    col_groups: list[tuple[int, int]] = []
    n_features = 0
    for loc in active_loci:
        n_alleles = len(catalog[loc])
        col_groups.append((n_features, n_features + n_alleles))
        for a in catalog[loc]:
            col_names.append(f"dosage_{loc}_{a}")
        n_features += n_alleles

    n_samples = len(df)
    matrix = np.zeros((n_samples, n_features), dtype=np.float32)
    missing = np.zeros((n_samples, len(active_loci)), dtype=bool)

    # Double-loop with per-cell df.loc + parse_genotype is fine for typical
    # microsat panels (n_loci * n_samples in the thousands). For larger
    # panels, vectorize via df.itertuples() / df.values and a single parse
    # pass shared with build_allele_catalog.
    for i, sid in enumerate(df.index):
        for li, locus in enumerate(active_loci):
            if not catalog[locus]:
                continue
            a1, a2 = parse_genotype(df.loc[sid, locus])
            if a1 is None:
                missing[i, li] = True
                continue
            idx = allele_index[locus]
            offset = col_groups[li][0]
            for a in (a1, a2):
                if a in idx:
                    matrix[i, offset + idx[a]] += 1.0

    for li, (start, end) in enumerate(col_groups):
        if start == end:
            continue
        m = missing[:, li]
        if not m.any():
            continue
        present = ~m
        if not present.any():
            continue
        site_mean = matrix[present, start:end].mean(axis=0)
        matrix[m, start:end] = site_mean

    return matrix, col_names


# Space is intentionally excluded from PAIR_SEPARATORS: CSV parsing can
# introduce leading/trailing whitespace on numeric cells, which would
# trigger false pair-format detection. parse_genotype() does accept " "
# as a within-cell separator (e.g. "10 11"), so single-cell space-separated
# pair format still works downstream — just not for auto-detection.
PAIR_SEPARATORS = (",", "/", "|")


def detect_format(df: pd.DataFrame) -> Literal["pair", "two_column"]:
    """Return ``"pair"`` or ``"two_column"`` based on cell content.

    Pair format: cells contain a separator (``,`` ``/`` ``|``).
    Two-column format: no cells contain pair separators; locus pairs are
    reconstructed from consecutive columns. Raises ``ValueError`` if
    two-column format has an odd number of locus columns.
    """
    locus_cols = [c for c in df.columns if c != "sampleID"]
    has_separator = False
    for c in locus_cols:
        if df[c].astype(str).str.contains(r"[,/|]", regex=True, na=False).any():
            has_separator = True
            break

    if has_separator:
        return "pair"

    if len(locus_cols) % 2 != 0:
        raise ValueError(
            f"Two-column format detected but locus column count ({len(locus_cols)}) "
            f"is odd; cannot reconstruct diploid pairs."
        )
    return "two_column"


def convert_two_column_to_pair(df: pd.DataFrame) -> pd.DataFrame:
    """Convert a two-column-format DataFrame to pair format.

    Consecutive locus columns are merged: ``variant_0``/``variant_1`` →
    ``locus_0`` with cell values ``"a1,a2"``. Locus names use a generic
    ``locus_<i>`` scheme (the original column names are not preserved).
    """
    locus_cols = [c for c in df.columns if c != "sampleID"]
    if len(locus_cols) % 2 != 0:
        raise ValueError(
            f"Cannot convert: odd number of locus columns ({len(locus_cols)}); "
            f"columns must form consecutive diploid pairs."
        )

    out = pd.DataFrame({"sampleID": df["sampleID"].values})
    for i in range(0, len(locus_cols), 2):
        c1, c2 = locus_cols[i], locus_cols[i + 1]
        out[f"locus_{i // 2}"] = df[c1].astype(str) + "," + df[c2].astype(str)
    return out
