"""Tests for DataLoaderMixin._load_from_microsat and the load_genotypes plumbing."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from locator.core import Locator


def _write_pair_tsv(path: Path) -> int:
    rows = [
        ["sampleID", "L1", "L2"],
        ["s1", "10,11", "20,22"],
        ["s2", "10,10", "20,20"],
        ["s3", "11,12", "22,24"],
        ["s4", "10,12", "NA"],
    ]
    path.write_text("\n".join("\t".join(r) for r in rows) + "\n")
    return len(rows) - 1


def _write_sample_data(path: Path, samples: list[str]) -> None:
    rows = [["sampleID", "x", "y"]]
    for i, sid in enumerate(samples):
        rows.append([sid, str(float(i)), str(float(i * 2))])
    path.write_text("\n".join("\t".join(r) for r in rows) + "\n")


def test_load_from_microsat_returns_float_dosage_matrix(tmp_path):
    inp = tmp_path / "ms.tsv"
    _write_pair_tsv(inp)
    sample_data = tmp_path / "samples.tsv"
    _write_sample_data(sample_data, ["s1", "s2", "s3", "s4"])

    loc = Locator({"sample_data": str(sample_data)})
    genotypes, samples = loc.load_genotypes(microsat=str(inp))

    assert isinstance(genotypes, np.ndarray)
    assert genotypes.ndim == 2
    assert np.issubdtype(genotypes.dtype, np.floating)
    n_sites, n_samples = genotypes.shape
    assert n_samples == 4
    assert n_sites > 0
    assert ((genotypes >= 0.0) & (genotypes <= 2.0)).all()
    assert list(samples) == ["s1", "s2", "s3", "s4"]


def test_load_from_microsat_imputes_missing_with_site_mean(tmp_path):
    inp = tmp_path / "ms.tsv"
    rows = [
        ["sampleID", "L1"],
        ["s1", "10,10"],
        ["s2", "10,11"],
        ["s3", "NA"],
    ]
    inp.write_text("\n".join("\t".join(r) for r in rows) + "\n")
    sample_data = tmp_path / "samples.tsv"
    _write_sample_data(sample_data, ["s1", "s2", "s3"])

    loc = Locator({"sample_data": str(sample_data)})
    genotypes, _ = loc.load_genotypes(microsat=str(inp))

    # genotypes shape is (n_sites, n_samples) per the continuous-dosage contract.
    # Two allele columns for L1: allele 10 mean = (2+1)/2 = 1.5;
    # allele 11 mean = (0+1)/2 = 0.5. s3 is column index 2.
    np.testing.assert_allclose(genotypes[:, 2], np.array([1.5, 0.5], dtype=np.float32))


def test_load_from_microsat_two_column_format(tmp_path):
    inp = tmp_path / "ms.tsv"
    rows = [
        ["sampleID", "variant_0", "variant_1", "variant_2", "variant_3"],
        ["s1", "10", "11", "20", "22"],
        ["s2", "10", "10", "20", "20"],
        ["s3", "11", "12", "22", "24"],
    ]
    inp.write_text("\n".join("\t".join(r) for r in rows) + "\n")
    sample_data = tmp_path / "samples.tsv"
    _write_sample_data(sample_data, ["s1", "s2", "s3"])

    loc = Locator({"sample_data": str(sample_data)})
    genotypes, samples = loc.load_genotypes(microsat=str(inp))

    # Two reconstructed loci × 3 alleles each = 6 dosage rows
    assert genotypes.shape == (6, 3)


def test_load_from_microsat_raises_on_missing_sampleID_column(tmp_path):
    inp = tmp_path / "ms.tsv"
    inp.write_text("badheader\tL1\nfoo\t10,11\n")
    sample_data = tmp_path / "samples.tsv"
    _write_sample_data(sample_data, ["foo"])

    loc = Locator({"sample_data": str(sample_data)})
    with pytest.raises(ValueError, match="sampleID"):
        loc.load_genotypes(microsat=str(inp))


def test_load_from_microsat_raises_on_duplicate_sampleIDs(tmp_path):
    inp = tmp_path / "ms.tsv"
    rows = [["sampleID", "L1"], ["s1", "10,11"], ["s1", "10,12"]]
    inp.write_text("\n".join("\t".join(r) for r in rows) + "\n")
    sample_data = tmp_path / "samples.tsv"
    _write_sample_data(sample_data, ["s1"])

    loc = Locator({"sample_data": str(sample_data)})
    with pytest.raises(ValueError, match="[Dd]uplicate"):
        loc.load_genotypes(microsat=str(inp))


def test_load_from_microsat_raises_when_all_loci_filtered_out(tmp_path):
    inp = tmp_path / "ms.tsv"
    rows = [["sampleID", "L1"], ["s1", "NA,NA"], ["s2", "NA,NA"]]
    inp.write_text("\n".join("\t".join(r) for r in rows) + "\n")
    sample_data = tmp_path / "samples.tsv"
    _write_sample_data(sample_data, ["s1", "s2"])

    loc = Locator({"sample_data": str(sample_data)})
    with pytest.raises(ValueError, match="[Nn]o loci"):
        loc.load_genotypes(microsat=str(inp))


def test_load_from_microsat_two_column_odd_columns_raises(tmp_path):
    inp = tmp_path / "ms.tsv"
    rows = [["sampleID", "v0", "v1", "v2"], ["s1", "10", "11", "12"]]
    inp.write_text("\n".join("\t".join(r) for r in rows) + "\n")
    sample_data = tmp_path / "samples.tsv"
    _write_sample_data(sample_data, ["s1"])

    loc = Locator({"sample_data": str(sample_data)})
    with pytest.raises(ValueError, match="odd"):
        loc.load_genotypes(microsat=str(inp))
