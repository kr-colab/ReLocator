"""Tests for DataLoaderMixin._load_from_gl and the load_genotypes plumbing.

Tests exercise the native GL loader public surface via Locator(...).load_genotypes(gl=...).
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


def test_load_from_gl_defaults_match_explicit_defaults(tmp_path):
    """Calling with no threshold kwargs must equal passing the documented defaults."""
    beagle = tmp_path / "out.beagle.gz"
    bam_list = tmp_path / "bams.txt"
    sample_data = tmp_path / "samples.tsv"
    _write_synthetic_beagle(beagle, n_sites=30, n_samples=5, seed=7)
    _write_bam_list(bam_list, [f"Ind{i}" for i in range(5)])
    _write_sample_data(sample_data, [f"Ind{i}" for i in range(5)])

    loc = Locator({"sample_data": str(sample_data)})
    implicit, _ = loc.load_genotypes(gl=str(beagle), bam_list=str(bam_list))
    explicit, _ = loc.load_genotypes(
        gl=str(beagle),
        bam_list=str(bam_list),
        gl_missing_threshold=0.4,
        gl_min_maf=0.01,
        gl_max_missing_frac=0.10,
    )
    np.testing.assert_array_equal(implicit, explicit)


@pytest.mark.parametrize(
    "seed, permissive_kwargs, strict_kwargs, strictly_fewer",
    [
        # A higher gl_min_maf removes more low-frequency sites; with
        # Dirichlet-uniform GLs the strict cutoff drops at least one.
        (11, {"gl_min_maf": 0.0}, {"gl_min_maf": 0.4}, True),
        # gl_missing_threshold=0.6 forces near-uniform GLs to count as missing
        # so gl_max_missing_frac actually bites. Monotone, but not guaranteed
        # to drop a site on this synthetic data, so only assert <=.
        (
            13,
            {"gl_missing_threshold": 0.6, "gl_min_maf": 0.0, "gl_max_missing_frac": 1.0},
            {"gl_missing_threshold": 0.6, "gl_min_maf": 0.0, "gl_max_missing_frac": 0.0},
            False,
        ),
    ],
)
def test_load_from_gl_stricter_filter_drops_more_sites(
    tmp_path, seed, permissive_kwargs, strict_kwargs, strictly_fewer
):
    """A stricter threshold keeps no more sites than a permissive one."""
    beagle = tmp_path / "out.beagle.gz"
    bam_list = tmp_path / "bams.txt"
    sample_data = tmp_path / "samples.tsv"
    _write_synthetic_beagle(beagle, n_sites=40, n_samples=6, seed=seed)
    _write_bam_list(bam_list, [f"Ind{i}" for i in range(6)])
    _write_sample_data(sample_data, [f"Ind{i}" for i in range(6)])

    loc = Locator({"sample_data": str(sample_data)})
    permissive, _ = loc.load_genotypes(
        gl=str(beagle), bam_list=str(bam_list), **permissive_kwargs
    )
    strict, _ = loc.load_genotypes(
        gl=str(beagle), bam_list=str(bam_list), **strict_kwargs
    )
    assert strict.shape[0] <= permissive.shape[0]
    if strictly_fewer:
        assert strict.shape[0] < permissive.shape[0]


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"gl_min_maf": -0.1}, "gl_min_maf"),
        ({"gl_max_missing_frac": 2.0}, "gl_max_missing_frac"),
        ({"gl_max_missing_frac": -0.5}, "gl_max_missing_frac"),
        ({"gl_missing_threshold": -1.0}, "gl_missing_threshold"),
        ({"gl_missing_threshold": 1.5}, "gl_missing_threshold"),
    ],
)
def test_load_from_gl_out_of_range_thresholds_raise(tmp_path, kwargs, match):
    beagle = tmp_path / "out.beagle.gz"
    bam_list = tmp_path / "bams.txt"
    sample_data = tmp_path / "samples.tsv"
    _write_synthetic_beagle(beagle, n_sites=5, n_samples=3, seed=17)
    _write_bam_list(bam_list, [f"Ind{i}" for i in range(3)])
    _write_sample_data(sample_data, [f"Ind{i}" for i in range(3)])

    loc = Locator({"sample_data": str(sample_data)})
    with pytest.raises(ValueError, match=match):
        loc.load_genotypes(gl=str(beagle), bam_list=str(bam_list), **kwargs)


def _write_controlled_beagle(path: Path, rows: list[tuple[str, list[float]]]) -> None:
    """Write a beagle with explicit marker names and GL triplets per sample.

    Each row is (marker, flat_gl) where flat_gl is n_samples*3 values.
    """
    n_cols = len(rows[0][1])
    n_samples = n_cols // 3
    header = ["marker", "allele1", "allele2"]
    for s in range(n_samples):
        header += [f"Ind{s}", f"Ind{s}", f"Ind{s}"]
    lines = ["\t".join(header)]
    for marker, gl in rows:
        lines.append("\t".join([marker, "A", "C"] + [f"{v:.6f}" for v in gl]))
    with gzip.open(path, "wb") as fh:
        fh.write(("\n".join(lines) + "\n").encode())


def test_load_from_gl_dosage_sets_positions_for_kept_sites(tmp_path):
    beagle = tmp_path / "out.beagle.gz"
    bam_list = tmp_path / "bams.txt"
    sample_data = tmp_path / "samples.tsv"
    ids = ["Ind0", "Ind1", "Ind2"]
    het = [0.0, 1.0, 0.0] * 3  # all-AB: dosage 1, MAF 0.5 -> kept
    mono = [1.0, 0.0, 0.0] * 3  # all-AA: dosage 0, MAF 0 -> dropped
    _write_controlled_beagle(
        beagle,
        [("chr1_10", het), ("chr1_20", het), ("chr1_30", mono), ("chr2_5", het)],
    )
    _write_bam_list(bam_list, ids)
    _write_sample_data(sample_data, ids)

    loc = Locator({"sample_data": str(sample_data)})
    genotypes, _ = loc.load_genotypes(gl=str(beagle), bam_list=str(bam_list))

    # site chr1_30 is monomorphic and filtered out; positions track kept sites.
    assert loc.positions is not None
    assert len(loc.positions) == genotypes.shape[0]
    np.testing.assert_array_equal(loc.positions, np.array([10, 20, 5]))
    np.testing.assert_array_equal(
        np.asarray(loc.chromosomes), np.array(["chr1", "chr1", "chr2"], dtype=object)
    )
    assert 30 not in loc.positions


def test_load_from_gl_full_gl_leaves_positions_unset(tmp_path):
    beagle = tmp_path / "out.beagle.gz"
    bam_list = tmp_path / "bams.txt"
    sample_data = tmp_path / "samples.tsv"
    _write_synthetic_beagle(beagle, n_sites=10, n_samples=4, seed=3)
    _write_bam_list(bam_list, [f"Ind{i}" for i in range(4)])
    _write_sample_data(sample_data, [f"Ind{i}" for i in range(4)])

    loc = Locator({"sample_data": str(sample_data)})
    loc.load_genotypes(gl=str(beagle), bam_list=str(bam_list), gl_mode="full_gl")
    # full_gl emits 3 rows per site, so per-site positions cannot align 1:1.
    assert getattr(loc, "positions", None) is None


@pytest.mark.slow
def test_run_windows_holdouts_on_gl_dosage(tmp_path):
    """End-to-end: GL dosage flows through sequential windowed holdouts."""
    beagle = tmp_path / "out.beagle.gz"
    bam_list = tmp_path / "bams.txt"
    sample_data = tmp_path / "samples.tsv"
    ids = [f"Ind{i}" for i in range(8)]
    _write_synthetic_beagle(beagle, n_sites=30, n_samples=8, seed=5)
    _write_bam_list(bam_list, ids)
    _write_sample_data(sample_data, ids)

    loc = Locator(
        {
            "sample_data": str(sample_data),
            "keras_verbose": 0,
            "max_epochs": 2,
            "patience": 1,
            "out": str(tmp_path / "gl_win"),
        }
    )
    genotypes, samples = loc.load_genotypes(gl=str(beagle), bam_list=str(bam_list))
    assert loc.positions is not None  # set by the GL loader from markers

    # positions are 0..29 (chr1_<i>), so a 10bp window yields multiple windows.
    result = loc.run_windows_holdouts(
        genotypes=genotypes,
        samples=samples,
        k=2,
        window_size=10,
        return_df=True,
        save_full_pred_matrix=False,
    )
    assert result is not None
    x_cols = [c for c in result.columns if c.startswith("x_")]
    assert len(x_cols) > 0
    assert len(result) == 2  # k holdout samples
