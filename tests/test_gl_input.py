"""Tests for DataLoaderMixin._load_from_gl and the load_genotypes plumbing.

The native GL loader replaces the previous gl_to_locator.py preprocessing
step. Tests exercise the public surface via Locator(...).load_genotypes(gl=...).
"""

from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np
import pytest

from locator.core import Locator


def _write_bam_list(path: Path, ids: list[str]) -> None:
    path.write_text("\n".join(f"/data/{sid}.bam" for sid in ids) + "\n")


def _write_synthetic_beagle(
    path: Path, n_sites: int = 20, n_samples: int = 4, seed: int = 0
) -> None:
    rng = np.random.default_rng(seed)
    header = ["marker", "allele1", "allele2"]
    for s in range(n_samples):
        header += [f"Ind{s}", f"Ind{s}", f"Ind{s}"]
    lines = ["\t".join(header)]
    for i in range(n_sites):
        triplets = rng.dirichlet(np.ones(3), size=n_samples)
        flat = triplets.flatten().tolist()
        row = [f"chr1_{i}", "A", "C"] + [f"{v:.6f}" for v in flat]
        lines.append("\t".join(row))
    raw = ("\n".join(lines) + "\n").encode()
    with gzip.open(path, "wb") as fh:
        fh.write(raw)


def _write_sample_data(path: Path, ids: list[str]) -> None:
    rows = [["sampleID", "x", "y"]]
    for i, sid in enumerate(ids):
        rows.append([sid, str(float(i)), str(float(i * 2))])
    path.write_text("\n".join("\t".join(r) for r in rows) + "\n")


def test_load_from_gl_dosage_mode_returns_float_matrix(tmp_path):
    beagle = tmp_path / "out.beagle.gz"
    bam_list = tmp_path / "bams.txt"
    sample_data = tmp_path / "samples.tsv"
    _write_synthetic_beagle(beagle, n_sites=20, n_samples=4, seed=0)
    _write_bam_list(bam_list, ["Ind0", "Ind1", "Ind2", "Ind3"])
    _write_sample_data(sample_data, ["Ind0", "Ind1", "Ind2", "Ind3"])

    loc = Locator({"sample_data": str(sample_data)})
    genotypes, samples = loc.load_genotypes(
        gl=str(beagle), bam_list=str(bam_list), gl_mode="dosage"
    )

    assert isinstance(genotypes, np.ndarray)
    assert genotypes.ndim == 2
    assert np.issubdtype(genotypes.dtype, np.floating)
    n_sites, n_samples = genotypes.shape
    assert n_samples == 4
    assert n_sites > 0
    assert n_sites <= 20  # MAF filter may drop some
    assert ((genotypes >= 0.0) & (genotypes <= 2.0)).all()
    assert list(samples) == ["Ind0", "Ind1", "Ind2", "Ind3"]


def test_load_from_gl_dosage_default_mode_is_dosage(tmp_path):
    beagle = tmp_path / "out.beagle.gz"
    bam_list = tmp_path / "bams.txt"
    sample_data = tmp_path / "samples.tsv"
    _write_synthetic_beagle(beagle, n_sites=20, n_samples=4, seed=1)
    _write_bam_list(bam_list, ["Ind0", "Ind1", "Ind2", "Ind3"])
    _write_sample_data(sample_data, ["Ind0", "Ind1", "Ind2", "Ind3"])

    loc = Locator({"sample_data": str(sample_data)})
    genotypes_default, _ = loc.load_genotypes(gl=str(beagle), bam_list=str(bam_list))
    genotypes_explicit, _ = loc.load_genotypes(
        gl=str(beagle), bam_list=str(bam_list), gl_mode="dosage"
    )
    np.testing.assert_array_equal(genotypes_default, genotypes_explicit)


def test_load_from_gl_full_gl_mode_returns_three_rows_per_site(tmp_path):
    beagle = tmp_path / "out.beagle.gz"
    bam_list = tmp_path / "bams.txt"
    sample_data = tmp_path / "samples.tsv"
    _write_synthetic_beagle(beagle, n_sites=20, n_samples=4, seed=2)
    _write_bam_list(bam_list, ["Ind0", "Ind1", "Ind2", "Ind3"])
    _write_sample_data(sample_data, ["Ind0", "Ind1", "Ind2", "Ind3"])

    loc = Locator({"sample_data": str(sample_data)})
    genotypes_full, _ = loc.load_genotypes(
        gl=str(beagle), bam_list=str(bam_list), gl_mode="full_gl"
    )
    genotypes_dose, _ = loc.load_genotypes(
        gl=str(beagle), bam_list=str(bam_list), gl_mode="dosage"
    )

    # full_gl produces 3 rows per kept site; dosage produces 1 row per kept site
    assert genotypes_full.shape[0] == 3 * genotypes_dose.shape[0]
    assert genotypes_full.shape[1] == 4
    assert ((genotypes_full >= 0.0) & (genotypes_full <= 1.0)).all()


def test_load_from_gl_requires_bam_list(tmp_path):
    beagle = tmp_path / "out.beagle.gz"
    sample_data = tmp_path / "samples.tsv"
    _write_synthetic_beagle(beagle, n_sites=5, n_samples=2, seed=3)
    _write_sample_data(sample_data, ["Ind0", "Ind1"])

    loc = Locator({"sample_data": str(sample_data)})
    with pytest.raises(ValueError, match="bam_list"):
        loc.load_genotypes(gl=str(beagle))


def test_load_from_gl_invalid_mode_raises(tmp_path):
    beagle = tmp_path / "out.beagle.gz"
    bam_list = tmp_path / "bams.txt"
    sample_data = tmp_path / "samples.tsv"
    _write_synthetic_beagle(beagle, n_sites=5, n_samples=2, seed=4)
    _write_bam_list(bam_list, ["Ind0", "Ind1"])
    _write_sample_data(sample_data, ["Ind0", "Ind1"])

    loc = Locator({"sample_data": str(sample_data)})
    with pytest.raises(ValueError, match="gl_mode"):
        loc.load_genotypes(gl=str(beagle), bam_list=str(bam_list), gl_mode="bogus")


def test_load_from_gl_dimension_mismatch_raises(tmp_path):
    beagle = tmp_path / "out.beagle.gz"
    bam_list = tmp_path / "bams.txt"
    sample_data = tmp_path / "samples.tsv"
    # Beagle has 4 samples; bam_list has 3 → 12 vs 9 cols mismatch
    _write_synthetic_beagle(beagle, n_sites=5, n_samples=4, seed=5)
    _write_bam_list(bam_list, ["Ind0", "Ind1", "Ind2"])
    _write_sample_data(sample_data, ["Ind0", "Ind1", "Ind2"])

    loc = Locator({"sample_data": str(sample_data)})
    with pytest.raises(ValueError, match="GL columns"):
        loc.load_genotypes(gl=str(beagle), bam_list=str(bam_list))


def test_load_from_gl_all_sites_filtered_raises(tmp_path):
    """Synthetic case: tiny beagle with monomorphic GL → all sites fail MAF."""
    beagle = tmp_path / "out.beagle.gz"
    bam_list = tmp_path / "bams.txt"
    sample_data = tmp_path / "samples.tsv"
    # Manually write a beagle where every sample at every site is confidently AA.
    n_sites = 3
    n_samples = 4
    header = ["marker", "allele1", "allele2"]
    for s in range(n_samples):
        header += [f"Ind{s}", f"Ind{s}", f"Ind{s}"]
    lines = ["\t".join(header)]
    for i in range(n_sites):
        triplet_str = ["1.000000", "0.000000", "0.000000"] * n_samples
        lines.append("\t".join([f"chr1_{i}", "A", "C"] + triplet_str))
    raw = ("\n".join(lines) + "\n").encode()
    with gzip.open(beagle, "wb") as fh:
        fh.write(raw)
    _write_bam_list(bam_list, [f"Ind{i}" for i in range(n_samples)])
    _write_sample_data(sample_data, [f"Ind{i}" for i in range(n_samples)])

    loc = Locator({"sample_data": str(sample_data)})
    with pytest.raises(ValueError, match="[Nn]o sites"):
        loc.load_genotypes(gl=str(beagle), bam_list=str(bam_list))
