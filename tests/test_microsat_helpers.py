"""Unit tests for locator._microsat parsing/encoding helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from locator import _microsat as ms


def _df_pairs(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows).set_index("sampleID")


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
def test_parse_genotype_variants(cell, expected):
    assert ms.parse_genotype(cell) == expected


def test_catalog_keeps_all_alleles_above_maf():
    df = _df_pairs(
        [
            {"sampleID": "s1", "L1": "10,11"},
            {"sampleID": "s2", "L1": "10,12"},
            {"sampleID": "s3", "L1": "11,12"},
            {"sampleID": "s4", "L1": "10,11"},
        ]
    )
    catalog = ms.build_allele_catalog(
        df, ["L1"], min_allele_freq=0.0, max_locus_missing=1.0
    )
    assert catalog["L1"] == [10, 11, 12]


def test_catalog_drops_rare_alleles():
    rows = [{"sampleID": f"s{i}", "L1": "10,11"} for i in range(99)]
    rows.append({"sampleID": "s99", "L1": "10,99"})
    df = _df_pairs(rows)
    catalog = ms.build_allele_catalog(
        df, ["L1"], min_allele_freq=0.05, max_locus_missing=1.0
    )
    assert 99 not in catalog["L1"]
    assert 10 in catalog["L1"] and 11 in catalog["L1"]


def test_dosage_block_basic():
    df = _df_pairs(
        [
            {"sampleID": "s1", "L1": "10,11"},
            {"sampleID": "s2", "L1": "10,10"},
            {"sampleID": "s3", "L1": "11,11"},
        ]
    )
    catalog = {"L1": [10, 11]}
    matrix, col_names = ms.encode_dosage_block(df, ["L1"], catalog)
    assert col_names == ["dosage_L1_10", "dosage_L1_11"]
    np.testing.assert_array_equal(
        matrix, np.array([[1, 1], [2, 0], [0, 2]], dtype=np.float32)
    )


def test_dosage_block_imputes_missing_with_site_mean():
    df = _df_pairs(
        [
            {"sampleID": "s1", "L1": "10,10"},
            {"sampleID": "s2", "L1": "10,11"},
            {"sampleID": "s3", "L1": "NA"},
        ]
    )
    catalog = {"L1": [10, 11]}
    matrix, _ = ms.encode_dosage_block(df, ["L1"], catalog)
    np.testing.assert_allclose(matrix[2], np.array([1.5, 0.5], dtype=np.float32))


def test_detect_format_pair():
    df = pd.DataFrame({"sampleID": ["s1", "s2"], "L1": ["10,11", "12,13"]})
    assert ms.detect_format(df) == "pair"


def test_detect_format_two_column():
    df = pd.DataFrame(
        {
            "sampleID": ["s1", "s2"],
            "variant_0": ["10", "12"],
            "variant_1": ["11", "13"],
        }
    )
    assert ms.detect_format(df) == "two_column"


def test_detect_format_two_column_odd_columns_raises():
    df = pd.DataFrame(
        {
            "sampleID": ["s1"],
            "variant_0": ["10"],
            "variant_1": ["11"],
            "variant_2": ["12"],
        }
    )
    with pytest.raises(ValueError, match="odd"):
        ms.detect_format(df)


def test_convert_two_column_pairs_alleles():
    df = pd.DataFrame(
        {
            "sampleID": ["s1", "s2"],
            "variant_0": ["10", "14"],
            "variant_1": ["11", "16"],
            "variant_2": ["20", "22"],
            "variant_3": ["21", "24"],
        }
    )
    out = ms.convert_two_column_to_pair(df)
    assert list(out.columns) == ["sampleID", "locus_0", "locus_1"]
    assert list(out["locus_0"]) == ["10,11", "14,16"]
    assert list(out["locus_1"]) == ["20,21", "22,24"]
