"""Tests for GPU optimization features."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import tensorflow as tf

from locator.core import Locator
from locator.gpu_optimizer import GPUOptimizer


class TestGPUOptimizer:
    """Test GPU optimization utilities."""

    def test_mixed_precision_setup(self):
        """Test mixed precision setup."""
        original_policy = tf.keras.mixed_precision.global_policy()

        try:
            with patch("tensorflow.config.list_physical_devices") as mock_devices:
                with patch(
                    "tensorflow.config.experimental.get_device_details"
                ) as mock_details:
                    mock_gpu = MagicMock()
                    mock_devices.return_value = [mock_gpu]
                    mock_details.return_value = {"compute_capability": (7, 0)}

                    result = GPUOptimizer.setup_mixed_precision()
                    assert result is True

            with patch("tensorflow.config.list_physical_devices") as mock_devices:
                mock_devices.return_value = []
                result = GPUOptimizer.setup_mixed_precision()
                assert result is False

        finally:
            tf.keras.mixed_precision.set_global_policy(original_policy)

    def test_optimize_gpu_memory(self):
        """Test GPU memory optimization modes."""
        GPUOptimizer.optimize_gpu_memory("growth")
        GPUOptimizer.optimize_gpu_memory("preallocate")
        GPUOptimizer.optimize_gpu_memory("limit", memory_limit=4096)


class TestLocatorGPUIntegration:
    """Test GPU optimizations integrated with Locator."""

    def test_locator_gpu_config(self):
        """Test Locator with GPU optimization config."""
        config = {
            "out": "test",
            "use_mixed_precision": True,
            "gpu_batch_size": "auto",
            "gpu_memory_mode": "growth",
            "disable_gpu": True,
        }

        locator = Locator(config)

        assert locator.config["use_mixed_precision"] is False
        assert locator.config["gpu_batch_size"] == "auto"
        assert locator.config["gpu_memory_mode"] == "growth"


class TestBatchSizeOptimization:
    """Test dynamic batch size optimization."""

    @pytest.mark.skipif(
        not tf.config.list_physical_devices("GPU"), reason="GPU not available"
    )
    def test_get_optimal_batch_size(self):
        """Test optimal batch size determination."""
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(256, input_shape=(1000,)),
                tf.keras.layers.Dense(256),
                tf.keras.layers.Dense(2),
            ]
        )

        optimal_size = GPUOptimizer.get_optimal_batch_size(
            model,
            input_shape=(1000,),
            target_memory_usage=0.8,
            min_batch_size=16,
            max_batch_size=512,
        )

        assert optimal_size & (optimal_size - 1) == 0
        assert 16 <= optimal_size <= 512

    def test_batch_size_with_small_dataset(self):
        """Test batch size optimization with small dataset."""
        model = tf.keras.Sequential(
            [tf.keras.layers.Dense(10, input_shape=(100,)), tf.keras.layers.Dense(2)]
        )

        batch_size = GPUOptimizer.get_optimal_batch_size(
            model,
            input_shape=(100,),
            min_batch_size=32,
            max_batch_size=2048,
            dataset_size=100,
        )

        assert batch_size == 32

    def test_batch_size_with_medium_dataset(self):
        """Test batch size optimization with medium dataset."""
        model = tf.keras.Sequential(
            [tf.keras.layers.Dense(10, input_shape=(100,)), tf.keras.layers.Dense(2)]
        )

        batch_size = GPUOptimizer.get_optimal_batch_size(
            model,
            input_shape=(100,),
            min_batch_size=32,
            max_batch_size=2048,
            dataset_size=1000,
        )

        assert batch_size <= 128

    def test_batch_size_no_gpu(self):
        """Test batch size optimization without GPU."""
        model = tf.keras.Sequential(
            [tf.keras.layers.Dense(10, input_shape=(100,)), tf.keras.layers.Dense(2)]
        )

        with patch("tensorflow.config.list_physical_devices") as mock_devices:
            mock_devices.return_value = []

            batch_size = GPUOptimizer.get_optimal_batch_size(
                model, input_shape=(100,), min_batch_size=32, max_batch_size=2048
            )

            assert batch_size == 32


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
