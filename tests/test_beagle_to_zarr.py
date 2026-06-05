"""Tests for the beagle -> dosage-zarr converter and the zarr dosage loader."""

from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np
import pytest

from locator.core import Locator
from locator.data.filters import is_dosage_matrix
from scripts.beagle_to_zarr import beagle_to_zarr


def _write_bam_list(path: Path, ids: list[str]) -> None:
    path.write_text("\n".join(f"/data/{sid}.bam" for sid in ids) + "\n")


def _write_synthetic_beagle(path: Path, n_sites: int, n_samples: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    header = ["marker", "allele1", "allele2"]
    for s in range(n_samples):
        header += [f"Ind{s}", f"Ind{s}", f"Ind{s}"]
    lines = ["\t".join(header)]
    for i in range(n_sites):
        flat = rng.dirichlet(np.ones(3), size=n_samples).flatten()
        lines.append(
            "\t".join([f"chr1_{i * 100}", "A", "C"] + [f"{v:.6f}" for v in flat])
        )
    with gzip.open(path, "wb") as fh:
        fh.write(("\n".join(lines) + "\n").encode())


def _write_sample_data(path: Path, ids: list[str]) -> None:
    rows = ["sampleID\tx\ty"]
    for i, sid in enumerate(ids):
        rows.append(f"{sid}\t{float(i)}\t{float(i * 2)}")
    path.write_text("\n".join(rows) + "\n")


@pytest.fixture
def beagle_inputs(tmp_path):
    beagle = tmp_path / "out.beagle.gz"
    bam_list = tmp_path / "bams.txt"
    sample_data = tmp_path / "samples.tsv"
    ids = [f"Ind{i}" for i in range(5)]
    _write_synthetic_beagle(beagle, n_sites=30, n_samples=5, seed=7)
    _write_bam_list(bam_list, ids)
    _write_sample_data(sample_data, ids)
    return beagle, bam_list, sample_data, ids


def test_converter_writes_expected_layout(tmp_path, beagle_inputs):
    import zarr

    beagle, bam_list, _sd, ids = beagle_inputs
    out = tmp_path / "store.zarr"
    beagle_to_zarr(str(beagle), str(bam_list), str(out), chunk_sites=8)

    root = zarr.open_group(str(out), mode="r")
    assert root.attrs.get("locator_format") == "dosage"
    dosage = root["dosage"]
    n_sites, n_samples = dosage.shape
    assert n_samples == 5
    assert dosage.dtype == np.float32
    assert dosage.chunks == (8, 5)  # chunked along the sites axis only
    assert root["variants/POS"].shape == (n_sites,)
    assert np.issubdtype(np.asarray(root["variants/POS"][:]).dtype, np.integer)
    assert root["variants/CHROM"].shape == (n_sites,)
    assert [str(s) for s in root["samples"][:]] == ids


def test_converter_matches_native_gl_loader(tmp_path, beagle_inputs):
    """Converter dosage must equal load_genotypes(gl=..., gl_mode='dosage')."""
    beagle, bam_list, sample_data, _ids = beagle_inputs
    out = tmp_path / "store.zarr"
    beagle_to_zarr(str(beagle), str(bam_list), str(out))

    loc_zarr = Locator({"sample_data": str(sample_data)})
    g_zarr, s_zarr = loc_zarr.load_genotypes(zarr=str(out))
    loc_gl = Locator({"sample_data": str(sample_data)})
    g_gl, s_gl = loc_gl.load_genotypes(gl=str(beagle), bam_list=str(bam_list))

    np.testing.assert_array_equal(g_zarr, g_gl)
    assert list(s_zarr) == list(s_gl)


def test_zarr_dosage_loads_as_dosage_with_positions(tmp_path, beagle_inputs):
    beagle, bam_list, sample_data, _ids = beagle_inputs
    out = tmp_path / "store.zarr"
    beagle_to_zarr(str(beagle), str(bam_list), str(out))

    loc = Locator({"sample_data": str(sample_data)})
    genotypes, _ = loc.load_genotypes(zarr=str(out))
    assert is_dosage_matrix(genotypes)
    assert genotypes.dtype == np.float32
    assert loc.positions is not None
    assert len(loc.positions) == genotypes.shape[0]
    assert loc.chromosomes is not None
    assert len(loc.chromosomes) == genotypes.shape[0]


def test_converter_refuses_overwrite_without_flag(tmp_path, beagle_inputs):
    beagle, bam_list, _sd, _ids = beagle_inputs
    out = tmp_path / "store.zarr"
    beagle_to_zarr(str(beagle), str(bam_list), str(out))
    # Second write to the same path without overwrite must fail.
    with pytest.raises(FileExistsError):
        beagle_to_zarr(str(beagle), str(bam_list), str(out), overwrite=False)
    # With overwrite it succeeds.
    beagle_to_zarr(str(beagle), str(bam_list), str(out), overwrite=True)
