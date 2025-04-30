# tests/test_core.py

import pytest
from unittest.mock import patch, MagicMock
import tensorflow as tf  # Import tensorflow to check its attributes
from locator.core import setup_gpu, Locator
import pandas as pd
import numpy as np
import allel
from pathlib import Path
import tempfile


@patch("locator.core.tf.config.list_physical_devices")
@patch("locator.core.tf.config.set_visible_devices")
@patch("locator.core.tf.config.experimental.set_memory_growth")
@patch("builtins.print")
def test_setup_gpu_no_gpus_available(
    mock_print,
    mock_set_memory_growth,
    mock_set_visible_devices,
    mock_list_physical_devices,
):
    """
    Tests setup_gpu behavior when no GPUs are detected.
    """
    # Simulate no GPUs being found
    mock_list_physical_devices.return_value = []

    # Call the function
    result = setup_gpu()

    # Assertions
    assert result is False
    mock_list_physical_devices.assert_called_once_with("GPU")
    mock_print.assert_called_with("No GPU devices available. Running on CPU.")
    # Ensure configuration functions were NOT called
    mock_set_visible_devices.assert_not_called()
    mock_set_memory_growth.assert_not_called()


# Mock setup_gpu for Locator tests
@patch("locator.core.tf.config.list_physical_devices")
@patch("locator.core.tf.config.experimental.set_memory_growth")
@patch("locator.core.setup_gpu")
def test_locator_init_defaults(
    mock_setup_gpu, mock_set_memory_growth, mock_list_physical_devices
):
    """Tests Locator initialization with default configuration."""
    locator = Locator()
    assert isinstance(locator.config, dict)
    # Check some default values
    assert locator.config["train_split"] == 0.9
    assert locator.config["width"] == 256
    assert locator.config["optimizer_algo"] == "adam"
    assert not hasattr(locator, "_sample_data_df")  # No df provided
    assert not hasattr(locator, "_genotype_df")  # No df provided
    mock_setup_gpu.assert_called_once_with(None)  # Default GPU


@patch("locator.core.tf.config.list_physical_devices")
@patch("locator.core.tf.config.experimental.set_memory_growth")
@patch("locator.core.setup_gpu")
def test_locator_init_custom_config(
    mock_setup_gpu, mock_set_memory_growth, mock_list_physical_devices
):
    """Tests Locator initialization with a custom configuration dictionary."""
    custom_config = {
        "out": "test_run",
        "train_split": 0.8,
        "width": 512,
        "max_epochs": 100,
        "disable_gpu": True,
    }
    locator = Locator(config=custom_config)
    assert locator.config["out"] == "test_run"
    assert locator.config["train_split"] == 0.8  # Overridden
    assert locator.config["width"] == 512  # Overridden
    assert locator.config["max_epochs"] == 100  # Overridden
    assert locator.config["batch_size"] == 32  # Default preserved
    assert locator.config["disable_gpu"] is True
    mock_setup_gpu.assert_not_called()  # GPU setup disabled


@patch("locator.core.tf.config.list_physical_devices")
@patch("locator.core.tf.config.experimental.set_memory_growth")
@patch("locator.core.setup_gpu")
def test_locator_init_with_sample_dataframe(
    mock_setup_gpu, mock_set_memory_growth, mock_list_physical_devices
):
    """Tests Locator initialization with a valid sample_data DataFrame."""
    coords_df = pd.DataFrame(
        {
            "sampleID": ["s1", "s2", "s3"],
            "x": [10.0, 20.0, 30.0],
            "y": [-5.0, -15.0, -25.0],
            "other_col": [1, 2, 3],
        }
    )
    config = {"sample_data": coords_df}
    locator = Locator(config=config)
    assert hasattr(locator, "_sample_data_df")
    pd.testing.assert_frame_equal(locator._sample_data_df, coords_df)
    mock_setup_gpu.assert_called_once_with(None)


@patch("locator.core.tf.config.list_physical_devices")
@patch("locator.core.tf.config.experimental.set_memory_growth")
@patch("locator.core.setup_gpu")
def test_locator_init_with_invalid_sample_dataframe(
    mock_setup_gpu, mock_set_memory_growth, mock_list_physical_devices
):
    """Tests Locator initialization with an invalid sample_data DataFrame."""
    invalid_coords_df = pd.DataFrame(
        {
            "sampleID": ["s1", "s2"],
            "longitude": [10.0, 20.0],  # Missing 'x' and 'y'
            "latitude": [-5.0, -15.0],
        }
    )
    config = {"sample_data": invalid_coords_df}
    with pytest.raises(
        ValueError, match="sample_data DataFrame must contain columns:.*'x'.*'y'"
    ):
        Locator(config=config)


@patch("locator.core.tf.config.list_physical_devices")
@patch("locator.core.tf.config.experimental.set_memory_growth")
@patch("locator.core.setup_gpu")
def test_locator_init_with_genotype_dataframe(
    mock_setup_gpu, mock_set_memory_growth, mock_list_physical_devices
):
    """Tests Locator initialization with a valid genotype_data DataFrame."""
    geno_df = pd.DataFrame(
        {
            1001: [0, 1],
            2001: [1, 2],
            3005: [2, 0],
        },
        index=["s1", "s2"],
    )
    config = {"genotype_data": geno_df}
    locator = Locator(config=config)
    assert hasattr(locator, "_genotype_df")
    pd.testing.assert_frame_equal(locator._genotype_df, geno_df)
    assert hasattr(locator, "positions")
    np.testing.assert_array_equal(locator.positions, np.array([1001.0, 2001.0, 3005.0]))
    # mock_setup_gpu.assert_called_once_with(None)


@patch("locator.core.tf.config.list_physical_devices")
@patch("locator.core.tf.config.experimental.set_memory_growth")
@patch("locator.core.setup_gpu")
def test_locator_init_with_invalid_genotype_values(
    mock_setup_gpu, mock_set_memory_growth, mock_list_physical_devices
):
    """Tests Locator initialization with invalid genotype values in DataFrame."""
    invalid_geno_df = pd.DataFrame(
        {
            1001: [0, 3],  # Contains '3'
            2001: [1, -1],  # Contains '-1'
        },
        index=["s1", "s2"],
    )
    config = {"genotype_data": invalid_geno_df}
    with pytest.raises(ValueError, match="Genotype values must be 0, 1, or 2"):
        Locator(config=config)


@patch("locator.core.tf.config.list_physical_devices")
@patch("locator.core.tf.config.experimental.set_memory_growth")
@patch("locator.core.setup_gpu")
def test_locator_init_with_non_numeric_genotype_columns(
    mock_setup_gpu, mock_set_memory_growth, mock_list_physical_devices
):
    """Tests Locator initialization with non-numeric SNP positions in DataFrame."""
    invalid_geno_df = pd.DataFrame(
        {
            "SNP1": [0, 1],
            "SNP2": [1, 2],
        },
        index=["s1", "s2"],
    )
    config = {"genotype_data": invalid_geno_df}
    with pytest.raises(
        ValueError, match="Column names must be convertible to integers"
    ):
        Locator(config=config)


@patch("locator.core.tf.config.list_physical_devices")
@patch("locator.core.tf.config.experimental.set_memory_growth")
@patch("locator.core.setup_gpu")
def test_locator_init_gpu_selection(
    mock_setup_gpu, mock_set_memory_growth, mock_list_physical_devices
):
    """Tests GPU selection via config."""
    # Test selection by integer
    config_gpu_1 = {"gpu_number": 1}
    Locator(config=config_gpu_1)
    mock_setup_gpu.assert_called_with(1)

    # Test selection by string
    mock_setup_gpu.reset_mock()
    config_gpu_str = {"gpu_number": "0"}
    Locator(config=config_gpu_str)
    mock_setup_gpu.assert_called_with(0)

    # Test invalid string (should use default)
    mock_setup_gpu.reset_mock()
    config_gpu_invalid = {"gpu_number": "abc"}
    Locator(config=config_gpu_invalid)
    mock_setup_gpu.assert_called_with(None)


@patch("locator.core.allel.read_vcf")
def test_load_from_vcf_success(mock_read_vcf):
    """
    Tests that _load_from_vcf correctly loads genotype and sample data from a VCF file.
    """
    # Mock VCF data structure as returned by allel.read_vcf
    mock_vcf = {
        "calldata/GT": np.array(
            [[[0, 0], [1, 0]], [[1, 1], [0, 1]]]
        ),  # shape: (variants, samples, ploidy)
        "samples": np.array(["sample1", "sample2"]),
    }
    mock_read_vcf.return_value = mock_vcf

    locator = Locator()
    genotypes, samples = locator._load_from_vcf("dummy.vcf")

    # Assertions
    assert isinstance(samples, np.ndarray)
    np.testing.assert_array_equal(samples, np.array(["sample1", "sample2"]))
    # Check that genotypes is an allel.GenotypeArray and matches the mock data
    assert hasattr(genotypes, "shape")
    np.testing.assert_array_equal(genotypes, mock_vcf["calldata/GT"])

    mock_read_vcf.assert_called_once_with("dummy.vcf")


@patch("locator.core.allel.read_vcf")
def test_load_from_vcf_failure(mock_read_vcf):
    """
    Tests that _load_from_vcf raises ValueError if the VCF file cannot be read.
    """
    mock_read_vcf.return_value = None
    locator = Locator()
    with pytest.raises(ValueError, match="Could not read VCF file: dummy.vcf"):
        locator._load_from_vcf("dummy.vcf")
    mock_read_vcf.assert_called_once_with("dummy.vcf")


@patch("locator.core.pd.read_csv")
def test_load_from_matrix_success(mock_read_csv):
    """
    Tests that _load_from_matrix correctly loads genotype and sample data from a matrix file.
    """
    # Mock DataFrame as would be read from a matrix file
    mock_df = MagicMock()
    mock_df.__getitem__.side_effect = lambda key: {
        "sampleID": ["s1", "s2"],
    }[key]
    mock_df.drop.return_value = pd.DataFrame(
        {
            1001: [0, 1],
            2001: [1, 2],
            3005: [2, 0],
        },
        index=[0, 1],
    )
    mock_read_csv.return_value = mock_df

    locator = Locator()
    genotypes, samples = locator._load_from_matrix("dummy_matrix.txt")

    # Assertions
    assert isinstance(samples, np.ndarray)
    np.testing.assert_array_equal(samples, np.array(["s1", "s2"]))
    # Check that genotypes is an allel.GenotypeArray and has the expected shape
    assert hasattr(genotypes, "shape")
    # The shape should be (n_sites, n_samples, 2)
    assert genotypes.shape[1] == 2  # 2 samples
    assert genotypes.shape[2] == 2  # ploidy=2

    mock_read_csv.assert_called_once_with("dummy_matrix.txt", sep="\t")


@patch("locator.core.pd.read_csv")
def test_load_from_matrix_invalid_file(mock_read_csv):
    """
    Tests that _load_from_matrix raises a KeyError if 'sampleID' column is missing.
    """
    # Simulate missing 'sampleID' column
    mock_df = pd.DataFrame(
        {
            1001: [0, 1],
            2001: [1, 2],
        }
    )
    mock_read_csv.return_value = mock_df

    locator = Locator()
    with pytest.raises(KeyError):
        locator._load_from_matrix("dummy_matrix.txt")
    mock_read_csv.assert_called_once_with("dummy_matrix.txt", sep="\t")


@patch("locator.core.pd.read_csv")
def test_load_from_matrix_with_invalid_genotypes_raises(mock_read_csv):
    """
    Tests that _load_from_matrix raises ValueError if invalid genotype values are present.
    """
    mock_df = MagicMock()
    mock_df.__getitem__.side_effect = lambda key: {
        "sampleID": ["s1", "s2"],
    }[key]
    mock_df.drop.return_value = pd.DataFrame(
        {
            1001: [0, 3],  # 3 is invalid
            2001: [1, -1],  # -1 is invalid
            3005: [2, 0],
        },
        index=[0, 1],
    )
    mock_read_csv.return_value = mock_df

    locator = Locator()
    with pytest.raises(ValueError, match="Genotype values must be 0, 1, or 2"):
        locator._load_from_matrix("dummy_matrix.txt")


# Test load_genotypes when genotype_data is provided via config as a DataFrame.
def test_load_genotypes_from_dataframe():
    # Create a small genotype DataFrame with SNP positions as columns (convertible to float)
    geno_df = pd.DataFrame({1001: [0, 1], 2005: [1, 2]}, index=["s1", "s2"])
    sample_df = pd.DataFrame(
        {"sampleID": ["s1", "s2"], "x": [10.0, 20.0], "y": [5.0, 15.0]}
    )
    config = {"genotype_data": geno_df, "sample_data": sample_df}
    locator = Locator(config=config)
    # When genotype_data is provided in config, Locator.__init__ stores _genotype_df.
    genotypes, samples = locator.load_genotypes()
    # Check that samples are strings
    np.testing.assert_array_equal(samples, np.array(["s1", "s2"], dtype=object))
    # Expect genotype array shape: (n_sites, n_samples, 2) -> (2, 2, 2)
    assert genotypes.shape == (2, 2, 2)
    assert isinstance(genotypes, allel.GenotypeArray)


# Test load_genotypes using a matrix file.
def test_load_genotypes_from_matrix(tmp_path):
    # Create a temporary matrix file with valid data.
    data = {"sampleID": ["s1", "s2"], "1001": [0, 1], "2005": [1, 2]}
    df = pd.DataFrame(data)
    matrix_file = tmp_path / "geno_matrix.txt"
    df.to_csv(matrix_file, sep="\t", index=False)

    sample_df = pd.DataFrame(
        {"sampleID": ["s1", "s2"], "x": [10.0, 20.0], "y": [5.0, 15.0]}
    )
    config = {"sample_data": sample_df}
    locator = Locator(config=config)
    # Do not provide genotype_data so that the matrix branch is executed.
    genotypes, samples = locator.load_genotypes(matrix=str(matrix_file))

    # Expected shape: (n_sites, n_samples, 2). In this case, n_sites = 2, n_samples = 2.
    assert genotypes.shape == (2, 2, 2)
    np.testing.assert_array_equal(samples, np.array(["s1", "s2"], dtype=object))
    assert isinstance(genotypes, allel.GenotypeArray)


# Test load_genotypes using a VCF file by patching allel.read_vcf.
def test_load_genotypes_from_vcf(monkeypatch):
    dummy_vcf_data = {
        "calldata/GT": np.array([[[0, 0], [1, 0]], [[1, 1], [0, 1]]]),  # shape: (2,2,2)
        "samples": np.array(["s1", "s2"]),
    }

    def dummy_read_vcf(vcf, log):
        return dummy_vcf_data

    monkeypatch.setattr("locator.core.allel.read_vcf", dummy_read_vcf)

    sample_df = pd.DataFrame(
        {"sampleID": ["s1", "s2"], "x": [10.0, 20.0], "y": [5.0, 15.0]}
    )
    config = {"sample_data": sample_df}
    locator = Locator(config=config)
    genotypes, samples = locator.load_genotypes(vcf="dummy.vcf")

    np.testing.assert_array_equal(samples, np.array(["s1", "s2"]))
    assert genotypes.shape == (2, 2, 2)
    assert isinstance(genotypes, allel.GenotypeArray)


# Test load_genotypes using a zarr input by patching the _load_from_zarr method.
def test_load_genotypes_from_zarr(monkeypatch):
    def dummy_load_from_zarr(self, zarr_path):
        # Return a dummy genotype array and samples.
        dummy_genotypes = np.array([[[0, 0], [1, 0]], [[1, 1], [0, 1]]])
        return allel.GenotypeArray(dummy_genotypes), np.array(["s1", "s2"])

    monkeypatch.setattr(Locator, "_load_from_zarr", dummy_load_from_zarr)

    sample_df = pd.DataFrame(
        {"sampleID": ["s1", "s2"], "x": [10.0, 20.0], "y": [5.0, 15.0]}
    )
    config = {"sample_data": sample_df}
    locator = Locator(config=config)
    genotypes, samples = locator.load_genotypes(zarr="dummy.zarr")

    np.testing.assert_array_equal(samples, np.array(["s1", "s2"]))
    assert genotypes.shape == (2, 2, 2)
    assert isinstance(genotypes, allel.GenotypeArray)


# Test error case: no genotype data provided.
def test_load_genotypes_no_data():
    sample_df = pd.DataFrame(
        {"sampleID": ["s1", "s2"], "x": [10.0, 20.0], "y": [5.0, 15.0]}
    )
    config = {"sample_data": sample_df}
    locator = Locator(config=config)
    with pytest.raises(ValueError, match="No genotype data provided"):
        locator.load_genotypes()


def dummy_create_network(
    input_shape, width, n_layers, dropout_prop, optimizer_config, loss_fn=None
):
    """Dummy replacement for create_network for testing Locator.train."""

    class DummyModel:
        def fit(self, dataset, epochs, verbose, validation_data, callbacks):
            # Return a dummy history object with a 'history' dict.
            DummyHistory = type(
                "DummyHistory", (), {"history": {"loss": [0.5], "val_loss": [0.6]}}
            )
            return DummyHistory()

    return DummyModel()
