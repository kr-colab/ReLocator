"""Tests for the GL input-extension script.

The script is invoked as a subprocess against synthetic fixtures so the test
exercises the same code path users hit. Fixtures are tiny (a handful of samples
and sites) so the suite stays fast.
"""

import gzip
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"


def run_script(script, *args):
    """Invoke a scripts/*.py file with the current Python; return CompletedProcess."""
    cmd = [sys.executable, str(SCRIPTS_DIR / script), *map(str, args)]
    return subprocess.run(cmd, capture_output=True, text=True, check=False)


# ---------------------------------------------------------------------------
# GL converter
# ---------------------------------------------------------------------------

def write_synthetic_beagle(path, n_sites=20, n_samples=4, seed=0):
    """Write a small valid beagle.gz with random GL triplets that sum to 1."""
    rng = np.random.default_rng(seed)
    rows = []
    header = ["marker", "allele1", "allele2"]
    for s in range(n_samples):
        header += [f"Ind{s}", f"Ind{s}", f"Ind{s}"]
    for i in range(n_sites):
        triplets = rng.dirichlet(np.ones(3), size=n_samples)
        flat = triplets.flatten().tolist()
        rows.append([f"chr1_{i}", "0", "1", *[f"{v:.6f}" for v in flat]])

    with gzip.open(path, "wt") as fh:
        fh.write("\t".join(header) + "\n")
        for r in rows:
            fh.write("\t".join(r) + "\n")
    return n_sites, n_samples


def write_bam_list(path, n_samples):
    with open(path, "w") as fh:
        for i in range(n_samples):
            fh.write(f"/path/to/sample_{i}.bam\n")
    return [f"sample_{i}" for i in range(n_samples)]


def test_gl_to_locator_dosage(tmp_path):
    n_sites, n_samples = 30, 5
    beagle = tmp_path / "in.beagle.gz"
    bam_list = tmp_path / "bams.txt"
    out = tmp_path / "geno.txt"

    write_synthetic_beagle(beagle, n_sites=n_sites, n_samples=n_samples, seed=1)
    expected_ids = write_bam_list(bam_list, n_samples)

    res = run_script(
        "gl_to_locator.py",
        "--beagle", beagle, "--bam_list", bam_list, "--out", out,
        "--min_maf", "0.0", "--max_missing_frac", "1.0",
    )
    assert res.returncode == 0, res.stderr

    df = pd.read_csv(out, sep="\t")
    assert list(df["sampleID"]) == expected_ids
    feat_cols = [c for c in df.columns if c != "sampleID"]
    assert len(feat_cols) == n_sites
    vals = df[feat_cols].values
    assert vals.shape == (n_samples, n_sites)
    # Continuous expected dosage in [0, 2] (the patched locator --matrix
    # loader accepts float values directly).
    assert vals.min() >= 0.0 and vals.max() <= 2.0


def test_gl_to_locator_full_gl(tmp_path):
    n_sites, n_samples = 12, 3
    beagle = tmp_path / "in.beagle.gz"
    bam_list = tmp_path / "bams.txt"
    out = tmp_path / "geno.txt"

    write_synthetic_beagle(beagle, n_sites=n_sites, n_samples=n_samples, seed=2)
    write_bam_list(bam_list, n_samples)

    res = run_script(
        "gl_to_locator.py",
        "--beagle", beagle, "--bam_list", bam_list, "--out", out,
        "--gl_mode", "full_gl",
        "--min_maf", "0.0", "--max_missing_frac", "1.0",
    )
    assert res.returncode == 0, res.stderr

    df = pd.read_csv(out, sep="\t")
    feat_cols = [c for c in df.columns if c != "sampleID"]
    # Three columns per site, suffixed _AA/_AB/_BB.
    assert len(feat_cols) == 3 * n_sites
    suffixes = [c.rsplit("_", 1)[-1] for c in feat_cols]
    assert set(suffixes) == {"AA", "AB", "BB"}
    # All triplet rows should sum approximately to 1 (since input is normalized).
    arr = df[feat_cols].values.reshape(n_samples, n_sites, 3)
    np.testing.assert_allclose(arr.sum(axis=2), 1.0, atol=1e-3)


def test_gl_to_locator_dimension_mismatch_errors(tmp_path):
    """If bam_list count disagrees with beagle column count, must error clearly."""
    beagle = tmp_path / "in.beagle.gz"
    bam_list = tmp_path / "bams.txt"
    out = tmp_path / "geno.txt"

    write_synthetic_beagle(beagle, n_sites=4, n_samples=4)
    write_bam_list(bam_list, n_samples=2)  # wrong count

    res = run_script(
        "gl_to_locator.py",
        "--beagle", beagle, "--bam_list", bam_list, "--out", out,
        "--min_maf", "0.0", "--max_missing_frac", "1.0",
    )
    assert res.returncode != 0
    assert "expected" in res.stderr.lower() or "expected" in res.stdout.lower()
