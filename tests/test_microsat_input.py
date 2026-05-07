"""Tests for scripts/microsat_to_locator.py.

Helper functions are tested by direct import; full-script behavior is
tested through subprocess invocations against synthetic fixtures.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
SCRIPT_PATH = SCRIPTS_DIR / "microsat_to_locator.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("microsat_to_locator", SCRIPT_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def msl():
    return _load_module()


def run_script(*args):
    cmd = [sys.executable, str(SCRIPT_PATH), *map(str, args)]
    return subprocess.run(cmd, capture_output=True, text=True, check=False)


@pytest.mark.parametrize(
    "cell, expected",
    [
        ("12,14", (12, 14)),
        ("12/14", (12, 14)),
        ("12 14", (12, 14)),
        ("12|14", (12, 14)),
        (" 12 , 14 ", (12, 14)),
        ("14", (14, 14)),
        ("NA", (None, None)),
        ("nan", (None, None)),
        (".", (None, None)),
        ("", (None, None)),
        ("0,0", (None, None)),
        ("not_a_number", (None, None)),
        ("12,not_a_number", (None, None)),
    ],
)
def test_parse_genotype_variants(msl, cell, expected):
    assert msl.parse_genotype(cell) == expected


def _df_pairs(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows).set_index("sampleID")


def test_catalog_keeps_all_alleles_above_maf(msl):
    df = _df_pairs(
        [
            {"sampleID": "s1", "L1": "10,11"},
            {"sampleID": "s2", "L1": "10,12"},
            {"sampleID": "s3", "L1": "11,12"},
            {"sampleID": "s4", "L1": "10,11"},
        ]
    )
    catalog = msl.build_allele_catalog(
        df, ["L1"], min_allele_freq=0.0, max_locus_missing=1.0
    )
    assert catalog["L1"] == [10, 11, 12]


def test_catalog_drops_rare_alleles(msl):
    rows = [{"sampleID": f"s{i}", "L1": "10,11"} for i in range(99)]
    rows.append({"sampleID": "s99", "L1": "10,99"})
    df = _df_pairs(rows)
    catalog = msl.build_allele_catalog(
        df, ["L1"], min_allele_freq=0.05, max_locus_missing=1.0
    )
    assert 99 not in catalog["L1"]
    assert 10 in catalog["L1"] and 11 in catalog["L1"]


def test_catalog_handles_all_missing_locus(msl):
    df = _df_pairs(
        [
            {"sampleID": "s1", "L1": "NA"},
            {"sampleID": "s2", "L1": "NA"},
        ]
    )
    catalog = msl.build_allele_catalog(
        df, ["L1"], min_allele_freq=0.0, max_locus_missing=1.0
    )
    assert catalog["L1"] == []


def test_catalog_warns_above_missing_threshold(msl, capsys):
    df = _df_pairs(
        [
            {"sampleID": "s1", "L1": "10,11"},
            {"sampleID": "s2", "L1": "NA"},
            {"sampleID": "s3", "L1": "NA"},
        ]
    )
    catalog = msl.build_allele_catalog(
        df, ["L1"], min_allele_freq=0.0, max_locus_missing=0.5
    )
    captured = capsys.readouterr()
    assert "WARNING" in captured.err
    assert "L1" in captured.err
    assert catalog["L1"] == [10, 11]


def test_catalog_does_not_warn_at_default_threshold(msl, capsys):
    """max_locus_missing=1.0 default should never warn."""
    df = _df_pairs(
        [
            {"sampleID": "s1", "L1": "NA"},
            {"sampleID": "s2", "L1": "NA"},
        ]
    )
    msl.build_allele_catalog(df, ["L1"], min_allele_freq=0.0, max_locus_missing=1.0)
    captured = capsys.readouterr()
    assert "WARNING" not in captured.err


def test_dosage_block_basic(msl):
    df = _df_pairs(
        [
            {"sampleID": "s1", "L1": "10,11"},
            {"sampleID": "s2", "L1": "10,10"},
            {"sampleID": "s3", "L1": "11,11"},
        ]
    )
    catalog = {"L1": [10, 11]}
    matrix, col_names = msl.encode_dosage_block(df, ["L1"], catalog)
    assert col_names == ["dosage_L1_10", "dosage_L1_11"]
    np.testing.assert_array_equal(
        matrix, np.array([[1, 1], [2, 0], [0, 2]], dtype=np.float32)
    )


def test_dosage_block_multi_locus(msl):
    df = _df_pairs(
        [
            {"sampleID": "s1", "L1": "10,11", "L2": "20,21"},
            {"sampleID": "s2", "L1": "11,11", "L2": "20,20"},
        ]
    )
    catalog = {"L1": [10, 11], "L2": [20, 21]}
    matrix, col_names = msl.encode_dosage_block(df, ["L1", "L2"], catalog)
    assert col_names == [
        "dosage_L1_10",
        "dosage_L1_11",
        "dosage_L2_20",
        "dosage_L2_21",
    ]
    expected = np.array([[1, 1, 1, 1], [0, 2, 2, 0]], dtype=np.float32)
    np.testing.assert_array_equal(matrix, expected)


def test_dosage_block_imputes_missing_with_site_mean(msl):
    df = _df_pairs(
        [
            {"sampleID": "s1", "L1": "10,10"},
            {"sampleID": "s2", "L1": "10,11"},
            {"sampleID": "s3", "L1": "NA"},
        ]
    )
    catalog = {"L1": [10, 11]}
    matrix, _ = msl.encode_dosage_block(df, ["L1"], catalog)
    # site mean across non-missing: col 0 = (2+1)/2 = 1.5; col 1 = (0+1)/2 = 0.5
    np.testing.assert_allclose(matrix[2], np.array([1.5, 0.5], dtype=np.float32))


def test_dosage_block_drops_alleles_outside_catalog(msl):
    df = _df_pairs(
        [
            {"sampleID": "s1", "L1": "10,99"},
            {"sampleID": "s2", "L1": "10,10"},
        ]
    )
    catalog = {"L1": [10]}
    matrix, col_names = msl.encode_dosage_block(df, ["L1"], catalog)
    assert col_names == ["dosage_L1_10"]
    # s1 has only one in-catalog allele (10), so dosage = 1
    np.testing.assert_allclose(matrix, np.array([[1.0], [2.0]], dtype=np.float32))


def test_detect_format_pair(msl):
    df = pd.DataFrame(
        {
            "sampleID": ["s1", "s2"],
            "L1": ["10,11", "12,13"],
        }
    )
    assert msl.detect_format(df) == "pair"


def test_detect_format_two_column(msl):
    df = pd.DataFrame(
        {
            "sampleID": ["s1", "s2"],
            "variant_0": ["10", "12"],
            "variant_1": ["11", "13"],
        }
    )
    assert msl.detect_format(df) == "two_column"


def test_detect_format_two_column_odd_columns_raises(msl):
    df = pd.DataFrame(
        {
            "sampleID": ["s1"],
            "variant_0": ["10"],
            "variant_1": ["11"],
            "variant_2": ["12"],
        }
    )
    with pytest.raises(ValueError, match="odd"):
        msl.detect_format(df)


def test_convert_two_column_pairs_alleles(msl):
    df = pd.DataFrame(
        {
            "sampleID": ["s1", "s2"],
            "variant_0": ["10", "14"],
            "variant_1": ["11", "16"],
            "variant_2": ["20", "22"],
            "variant_3": ["21", "24"],
        }
    )
    out = msl.convert_two_column_to_pair(df)
    assert list(out.columns) == ["sampleID", "locus_0", "locus_1"]
    assert list(out["locus_0"]) == ["10,11", "14,16"]
    assert list(out["locus_1"]) == ["20,21", "22,24"]


def _write_pair_tsv(path: Path) -> int:
    """Write a small pair-format input. Returns n_samples."""
    rows = [
        ["sampleID", "L1", "L2"],
        ["s1", "10,11", "20,22"],
        ["s2", "10,10", "20,20"],
        ["s3", "11,12", "22,24"],
        ["s4", "10,12", "NA"],
    ]
    path.write_text("\n".join("\t".join(r) for r in rows) + "\n")
    return len(rows) - 1


def test_cli_dosage_basic(tmp_path: Path):
    inp = tmp_path / "ms.tsv"
    _write_pair_tsv(inp)
    out = tmp_path / "feat.tsv"
    proc = run_script("--microsat", inp, "--out", out)
    assert proc.returncode == 0, proc.stderr
    df = pd.read_csv(out, sep="\t")
    cols = list(df.columns)
    assert cols[0] == "sampleID"
    for c in cols[1:]:
        assert c.startswith("dosage_"), c


def test_cli_two_column_format_works(tmp_path: Path):
    inp = tmp_path / "ms.tsv"
    rows = [
        ["sampleID", "variant_0", "variant_1", "variant_2", "variant_3"],
        ["s1", "10", "11", "20", "22"],
        ["s2", "10", "10", "20", "20"],
        ["s3", "11", "12", "22", "24"],
    ]
    inp.write_text("\n".join("\t".join(r) for r in rows) + "\n")
    out = tmp_path / "feat.tsv"
    proc = run_script("--microsat", inp, "--out", out)
    assert proc.returncode == 0, proc.stderr
    df = pd.read_csv(out, sep="\t")
    assert len(df) == 3
    # 2 reconstructed loci with alleles {10,11,12} and {20,22,24} → 6 dosage cols
    assert df.shape[1] == 7  # 6 features + sampleID


def test_cli_writes_encoding_report(tmp_path: Path):
    inp = tmp_path / "ms.tsv"
    _write_pair_tsv(inp)
    out = tmp_path / "feat.tsv"
    enc = tmp_path / "encoding.tsv"
    proc = run_script(
        "--microsat",
        inp,
        "--out",
        out,
        "--report_encoding",
        enc,
    )
    assert proc.returncode == 0, proc.stderr
    enc_df = pd.read_csv(enc, sep="\t")
    assert {"locus", "allele", "column_name"}.issubset(enc_df.columns)
    out_df = pd.read_csv(out, sep="\t")
    feature_cols = [c for c in out_df.columns if c != "sampleID"]
    assert len(enc_df) == len(feature_cols)


def test_cli_missing_sampleID_column_returns_1(tmp_path: Path):
    inp = tmp_path / "ms.tsv"
    inp.write_text("badheader\tL1\nfoo\t10,11\n")
    out = tmp_path / "feat.tsv"
    proc = run_script("--microsat", inp, "--out", out)
    assert proc.returncode == 1
    assert "sampleID" in (proc.stderr + proc.stdout)


def test_cli_duplicate_sampleIDs_returns_2(tmp_path: Path):
    inp = tmp_path / "ms.tsv"
    rows = [
        ["sampleID", "L1"],
        ["s1", "10,11"],
        ["s1", "10,12"],
    ]
    inp.write_text("\n".join("\t".join(r) for r in rows) + "\n")
    out = tmp_path / "feat.tsv"
    proc = run_script("--microsat", inp, "--out", out)
    assert proc.returncode == 2
    assert "duplicate" in (proc.stderr + proc.stdout).lower()


def test_cli_two_column_odd_columns_returns_3_clean_error(tmp_path: Path):
    """Odd column count in two-column format produces a clean error, not a traceback."""
    inp = tmp_path / "ms.tsv"
    rows = [
        ["sampleID", "v0", "v1", "v2"],
        ["s1", "10", "11", "12"],
    ]
    inp.write_text("\n".join("\t".join(r) for r in rows) + "\n")
    out = tmp_path / "feat.tsv"
    proc = run_script("--microsat", inp, "--out", out)
    assert proc.returncode == 3
    assert "odd" in (proc.stderr + proc.stdout).lower()
    assert "Traceback" not in proc.stderr


def test_cli_no_active_loci_returns_4(tmp_path: Path):
    inp = tmp_path / "ms.tsv"
    rows = [
        ["sampleID", "L1"],
        ["s1", "NA,NA"],
        ["s2", "NA,NA"],
    ]
    inp.write_text("\n".join("\t".join(r) for r in rows) + "\n")
    out = tmp_path / "feat.tsv"
    proc = run_script("--microsat", inp, "--out", out)
    assert proc.returncode == 4
    assert "no loci" in (proc.stderr + proc.stdout).lower()
