# tests/test_core.py

import pytest
from unittest.mock import patch, MagicMock
import tensorflow as tf  # Import tensorflow to check its attributes
from locator.core import setup_gpu, Locator
import pandas as pd
import numpy as np


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
