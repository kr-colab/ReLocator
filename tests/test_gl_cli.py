"""End-to-end CLI test for `locator --gl --bam_list`."""

from __future__ import annotations

import gzip
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np


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


def _write_sample_data(path: Path, ids: list[str]) -> None:
    rows = [["sampleID", "x", "y"]]
    for i, sid in enumerate(ids):
        rows.append([sid, str(float(i)), str(float(i * 2))])
    path.write_text("\n".join("\t".join(r) for r in rows) + "\n")


def _cmd_prefix() -> list[str]:
    if shutil.which("locator") is None:
        return [sys.executable, "-m", "locator.cli"]
    return ["locator"]


def test_cli_gl_dosage_runs_end_to_end(tmp_path: Path):
    beagle = tmp_path / "out.beagle.gz"
    bam_list = tmp_path / "bams.txt"
    sample_data = tmp_path / "samples.tsv"
    _write_synthetic_beagle(beagle, n_sites=100, n_samples=6, seed=0)
    _write_bam_list(bam_list, [f"Ind{i}" for i in range(6)])
    _write_sample_data(sample_data, [f"Ind{i}" for i in range(6)])

    out_prefix = tmp_path / "out" / "run"

    proc = subprocess.run(
        _cmd_prefix()
        + [
            "--gl",
            str(beagle),
            "--bam_list",
            str(bam_list),
            "--gl_mode",
            "dosage",
            "--sample_data",
            str(sample_data),
            "--out",
            str(out_prefix),
            "--max_epochs",
            "2",
            "--patience",
            "2",
            "--train_split",
            "0.5",
            "--seed",
            "1",
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode == 0, f"stderr:\n{proc.stderr}\nstdout:\n{proc.stdout}"
    expected_predlocs = out_prefix.parent / "run_predlocs.txt"
    assert expected_predlocs.exists(), (
        f"Expected {expected_predlocs} — stderr:\n{proc.stderr}\nstdout:\n{proc.stdout}"
    )


def test_cli_gl_without_bam_list_errors(tmp_path: Path):
    beagle = tmp_path / "out.beagle.gz"
    sample_data = tmp_path / "samples.tsv"
    _write_synthetic_beagle(beagle, n_sites=20, n_samples=4, seed=1)
    _write_sample_data(sample_data, [f"Ind{i}" for i in range(4)])

    out_prefix = tmp_path / "out" / "run"

    proc = subprocess.run(
        _cmd_prefix()
        + [
            "--gl",
            str(beagle),
            "--sample_data",
            str(sample_data),
            "--out",
            str(out_prefix),
            "--max_epochs",
            "2",
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode != 0
    assert "bam_list" in (proc.stderr + proc.stdout).lower()
