"""End-to-end CLI test for `locator --microsat`."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest


def _write_pair_tsv(path: Path) -> None:
    rows = [
        ["sampleID", "L1", "L2"],
        ["s1", "10,11", "20,22"],
        ["s2", "10,10", "20,20"],
        ["s3", "11,12", "22,24"],
        ["s4", "10,12", "20,24"],
        ["s5", "10,11", "22,24"],
        ["s6", "11,11", "20,22"],
    ]
    path.write_text("\n".join("\t".join(r) for r in rows) + "\n")


def _write_sample_data(path: Path, ids: list[str]) -> None:
    rows = [["sampleID", "x", "y"]]
    for i, sid in enumerate(ids):
        rows.append([sid, str(float(i)), str(float(i * 2))])
    path.write_text("\n".join("\t".join(r) for r in rows) + "\n")


@pytest.mark.slow
def test_cli_microsat_runs_end_to_end(tmp_path: Path):
    if shutil.which("locator") is None:
        cmd_prefix = [sys.executable, "-m", "locator.cli"]
    else:
        cmd_prefix = ["locator"]

    inp = tmp_path / "ms.tsv"
    _write_pair_tsv(inp)
    sample_data = tmp_path / "samples.tsv"
    _write_sample_data(sample_data, [f"s{i}" for i in range(1, 7)])
    out_prefix = tmp_path / "out" / "run"

    proc = subprocess.run(
        cmd_prefix
        + [
            "--microsat",
            str(inp),
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
    # Locator writes <out>_predlocs.txt on successful training.
    expected_predlocs = out_prefix.parent / "run_predlocs.txt"
    assert expected_predlocs.exists(), (
        f"Expected {expected_predlocs} — stderr:\n{proc.stderr}\nstdout:\n{proc.stdout}"
    )
