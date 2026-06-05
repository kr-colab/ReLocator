"""Unit tests for locator._gl parsing/encoding helpers."""

from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np
import pytest

from locator import _gl


def _write_bam_list(path: Path, ids: list[str]) -> None:
    path.write_text("\n".join(f"/data/{sid}.bam" for sid in ids) + "\n")


def _write_synthetic_beagle(
    path: Path, n_sites: int, n_samples: int, seed: int = 0
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


def test_sample_ids_from_bam_list(tmp_path):
    bam_list = tmp_path / "bams.txt"
    _write_bam_list(bam_list, ["sampleA", "sampleB", "sampleC"])
    ids = _gl.sample_ids_from_bam_list(bam_list)
    assert ids == ["sampleA", "sampleB", "sampleC"]


def test_sample_ids_from_bam_list_strips_extensions(tmp_path):
    bam_list = tmp_path / "bams.txt"
    bam_list.write_text("/data/sampleA.bam\n/data/sampleB.bam\n")
    ids = _gl.sample_ids_from_bam_list(bam_list)
    assert ids == ["sampleA", "sampleB"]


def test_sample_ids_from_bam_list_empty_raises(tmp_path):
    bam_list = tmp_path / "bams.txt"
    bam_list.write_text("\n\n")
    with pytest.raises(ValueError, match="No BAM paths"):
        _gl.sample_ids_from_bam_list(bam_list)


def test_load_beagle_returns_markers_and_flat_gl(tmp_path):
    beagle = tmp_path / "out.beagle.gz"
    _write_synthetic_beagle(beagle, n_sites=5, n_samples=3, seed=42)
    markers, gl_flat = _gl.load_beagle(beagle)
    assert len(markers) == 5
    assert gl_flat.shape == (5, 9)  # 3 samples * 3 GLs
    assert gl_flat.dtype == np.float32


def test_load_beagle_empty_raises(tmp_path):
    beagle = tmp_path / "empty.beagle.gz"
    with gzip.open(beagle, "wb") as fh:
        fh.write(b"marker\tallele1\tallele2\tInd0\tInd0\tInd0\n")
    with pytest.raises(ValueError, match="No data rows"):
        _gl.load_beagle(beagle)


def test_validate_dimensions_pass():
    gl_flat = np.zeros((10, 9), dtype=np.float32)
    _gl.validate_dimensions(
        gl_flat, n_samples=3, beagle_path="x.beagle.gz", bam_list_path="x.txt"
    )


def test_validate_dimensions_mismatch_raises():
    gl_flat = np.zeros((10, 9), dtype=np.float32)
    with pytest.raises(ValueError, match="6.*2 samples"):
        _gl.validate_dimensions(
            gl_flat, n_samples=2, beagle_path="x.beagle.gz", bam_list_path="x.txt"
        )


def test_reshape_gl():
    gl_flat = np.arange(18, dtype=np.float32).reshape(2, 9)
    out = _gl.reshape_gl(gl_flat, n_samples=3)
    assert out.shape == (2, 3, 3)
    np.testing.assert_array_equal(out[0, 0], [0, 1, 2])
    np.testing.assert_array_equal(out[0, 2], [6, 7, 8])


def test_expected_dosage():
    gl = np.array(
        [
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        ],
        dtype=np.float32,
    )
    out = _gl.expected_dosage(gl)
    np.testing.assert_allclose(out, np.array([[0.0, 1.0, 2.0]]))


def test_detect_missing():
    gl = np.array(
        [
            [[0.9, 0.05, 0.05], [0.34, 0.33, 0.33], [0.0, 1.0, 0.0]],
        ],
        dtype=np.float32,
    )
    mask = _gl.detect_missing(gl, gl_missing_threshold=0.4)
    np.testing.assert_array_equal(mask, np.array([[False, True, False]]))


def test_impute_dosage_with_site_mean_fills_missing():
    dosage = np.array([[0.0, 1.0, 2.0, 99.0]], dtype=np.float32)
    missing = np.array([[False, False, False, True]])
    out = _gl.impute_dosage_with_site_mean(dosage, missing)
    # mean of non-missing 0/1/2 = 1.0
    assert out[0, 3] == pytest.approx(1.0)


def test_impute_gl_with_site_mean_fills_missing_triplet():
    gl = np.array(
        [
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [99.0, 99.0, 99.0]],
        ],
        dtype=np.float32,
    )
    missing = np.array([[False, False, True]])
    out = _gl.impute_gl_with_site_mean(gl, missing)
    # mean triplet of first two = (0.5, 0.5, 0.0)
    np.testing.assert_allclose(out[0, 2], np.array([0.5, 0.5, 0.0]))


def _impute_dosage_reference(dosage, missing_mask):
    """Original per-site loop, kept here as the numerical reference."""
    for i in range(dosage.shape[0]):
        mask = missing_mask[i]
        if not mask.any():
            continue
        present = dosage[i, ~mask]
        dosage[i, mask] = present.mean() if present.size > 0 else 0.0
    return dosage


def _impute_gl_reference(gl, missing_mask):
    """Original per-site loop, kept here as the numerical reference."""
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


def test_impute_dosage_matches_reference_loop():
    rng = np.random.default_rng(0)
    dosage = rng.uniform(0.0, 2.0, size=(50, 8)).astype(np.float32)
    missing = rng.random((50, 8)) < 0.3
    missing[3, :] = True  # a fully-missing site exercises the 0.0 fallback
    missing[7, :] = False  # a site with no missing samples is a no-op
    got = _gl.impute_dosage_with_site_mean(dosage.copy(), missing)
    expected = _impute_dosage_reference(dosage.copy(), missing)
    np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-6)


def test_impute_gl_matches_reference_loop():
    rng = np.random.default_rng(1)
    gl = rng.dirichlet(np.ones(3), size=(50, 8)).astype(np.float32)
    missing = rng.random((50, 8)) < 0.3
    missing[3, :] = True  # fully-missing site exercises the [1/3,1/3,1/3] fallback
    missing[7, :] = False  # no-missing site is a no-op
    got = _gl.impute_gl_with_site_mean(gl.copy(), missing)
    expected = _impute_gl_reference(gl.copy(), missing)
    np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-6)


def test_impute_dosage_all_missing_site_falls_back_to_zero():
    dosage = np.array([[5.0, 6.0, 7.0]], dtype=np.float32)
    missing = np.array([[True, True, True]])
    out = _gl.impute_dosage_with_site_mean(dosage, missing)
    np.testing.assert_array_equal(out, np.zeros((1, 3), dtype=np.float32))


def test_impute_gl_all_missing_site_falls_back_to_uniform():
    gl = np.full((1, 3, 3), 9.0, dtype=np.float32)
    missing = np.array([[True, True, True]])
    out = _gl.impute_gl_with_site_mean(gl, missing)
    np.testing.assert_allclose(out, np.full((1, 3, 3), 1.0 / 3, dtype=np.float32))


def test_impute_dosage_mutates_in_place():
    dosage = np.array([[0.0, 1.0, 2.0, 99.0]], dtype=np.float32)
    missing = np.array([[False, False, False, True]])
    out = _gl.impute_dosage_with_site_mean(dosage, missing)
    assert out is dosage


def test_impute_gl_mutates_in_place():
    gl = np.zeros((1, 3, 3), dtype=np.float32)
    missing = np.array([[False, False, True]])
    out = _gl.impute_gl_with_site_mean(gl, missing)
    assert out is gl


def test_load_beagle_malformed_value_raises(tmp_path):
    beagle = tmp_path / "bad.beagle.gz"
    with gzip.open(beagle, "wt") as fh:
        fh.write("marker\tallele1\tallele2\tInd0\tInd0\tInd0\n")
        fh.write("chr1_0\tA\tC\t0.5\tnot_a_float\t0.2\n")
    with pytest.raises(ValueError, match="Failed to parse"):
        _gl.load_beagle(beagle)


def test_load_beagle_uncompressed_plain_file(tmp_path):
    """compression is inferred from the suffix, so a plain .beagle loads too."""
    beagle = tmp_path / "out.beagle"
    lines = ["marker\tallele1\tallele2\tInd0\tInd0\tInd0\tInd1\tInd1\tInd1"]
    for i in range(4):
        lines.append(f"chr1_{i}\tA\tC\t0.8\t0.1\t0.1\t0.2\t0.5\t0.3")
    beagle.write_text("\n".join(lines) + "\n")
    markers, gl_flat = _gl.load_beagle(beagle)
    assert len(markers) == 4
    assert gl_flat.shape == (4, 6)  # 2 samples * 3 GLs
    assert gl_flat.dtype == np.float32


def test_filter_sites_min_maf_drops_invariant():
    # site 0: all dosage = 0 → MAF = 0; site 1: 50/50 → MAF = 0.5
    dosage = np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 2.0]], dtype=np.float32)
    missing = np.zeros((2, 3), dtype=bool)
    keep, reasons = _gl.filter_sites(dosage, missing, min_maf=0.05, max_missing_frac=1.0)
    np.testing.assert_array_equal(keep, [False, True])
    assert reasons["min_maf"] == 1
    assert reasons["total_kept"] == 1


def test_filter_sites_max_missing_frac_drops_high_missingness():
    dosage = np.array([[1.0, 1.0, 1.0]], dtype=np.float32)
    missing = np.array([[True, True, False]])  # missing_frac = 2/3
    keep, _reasons = _gl.filter_sites(dosage, missing, min_maf=0.0, max_missing_frac=0.5)
    np.testing.assert_array_equal(keep, [False])
