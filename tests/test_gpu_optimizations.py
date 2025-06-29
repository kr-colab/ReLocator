"""Tests for GPU optimization features."""

import pytest
import numpy as np
import tensorflow as tf
from unittest.mock import patch, MagicMock

from locator.gpu_optimizer import (
    GPUOptimizer, 
    GradientAccumulator,
    create_optimized_training_config
)
from locator.core import Locator


class TestGPUOptimizer:
    """Test GPU optimization utilities."""
    
    def test_mixed_precision_setup(self):
        """Test mixed precision setup."""
        # Save current policy
        original_policy = tf.keras.mixed_precision.global_policy()
        
        try:
            # Mock GPU with compute capability 7.0 (supports mixed precision)
            with patch('tensorflow.config.list_physical_devices') as mock_devices:
                with patch('tensorflow.config.experimental.get_device_details') as mock_details:
                    mock_gpu = MagicMock()
                    mock_devices.return_value = [mock_gpu]
                    mock_details.return_value = {'compute_capability': (7, 0)}
                    
                    result = GPUOptimizer.setup_mixed_precision()
                    
                    # Should return True for GPU with compute capability >= 7
                    assert result == True
                    
            # Test with no GPU
            with patch('tensorflow.config.list_physical_devices') as mock_devices:
                mock_devices.return_value = []
                result = GPUOptimizer.setup_mixed_precision()
                assert result == False
                
        finally:
            # Restore original policy
            tf.keras.mixed_precision.set_global_policy(original_policy)
    
    def test_create_efficient_dataset(self):
        """Test efficient dataset creation."""
        # Create dummy data
        X = np.random.randn(100, 10).astype(np.float32)
        y = np.random.randn(100, 2).astype(np.float32)
        
        # Create dataset
        dataset = GPUOptimizer.create_efficient_dataset(
            X, y, batch_size=32, training=True
        )
        
        # Check dataset properties
        assert isinstance(dataset, tf.data.Dataset)
        
        # Get first batch
        for batch_x, batch_y in dataset.take(1):
            assert batch_x.shape == (32, 10)  # batch_size x features
            assert batch_y.shape == (32, 2)   # batch_size x outputs
    
    def test_optimize_gpu_memory(self):
        """Test GPU memory optimization modes."""
        # Test growth mode (should not raise)
        GPUOptimizer.optimize_gpu_memory("growth")
        
        # Test preallocate mode (should not raise)
        GPUOptimizer.optimize_gpu_memory("preallocate")
        
        # Test limit mode with memory limit
        GPUOptimizer.optimize_gpu_memory("limit", memory_limit=4096)
    
    def test_get_gpu_info(self):
        """Test GPU info retrieval."""
        info = GPUOptimizer.get_gpu_info()
        
        assert isinstance(info, dict)
        assert 'gpu_count' in info
        assert 'gpus' in info
        assert isinstance(info['gpus'], list)


class TestGradientAccumulator:
    """Test gradient accumulation functionality."""
    
    def test_gradient_accumulator_init(self):
        """Test gradient accumulator initialization."""
        # Create simple model
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(10, input_shape=(5,)),
            tf.keras.layers.Dense(1)
        ])
        
        # Create accumulator
        accumulator = GradientAccumulator(model, accumulation_steps=4)
        
        assert accumulator.model == model
        assert accumulator.accumulation_steps == 4
        assert len(accumulator.accumulated_gradients) == len(model.trainable_variables)
    
    def test_gradient_accumulation_step(self):
        """Test single accumulation step."""
        # Create simple model
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(10, input_shape=(5,)),
            tf.keras.layers.Dense(1)
        ])
        
        accumulator = GradientAccumulator(model, accumulation_steps=2)
        
        # Create dummy data
        X = tf.random.normal((8, 5))
        y = tf.random.normal((8, 1))
        
        # Define loss function
        loss_fn = tf.keras.losses.MeanSquaredError()
        
        # Perform accumulation step
        loss = accumulator.accumulate_step(X, y, loss_fn)
        
        assert isinstance(loss.numpy(), (float, np.float32))
        assert accumulator.step_count.numpy() == 1


class TestLocatorGPUIntegration:
    """Test GPU optimizations integrated with Locator."""
    
    def test_locator_gpu_config(self):
        """Test Locator with GPU optimization config."""
        config = {
            "out": "test",
            "use_mixed_precision": True,
            "gpu_batch_size": "auto",
            "gpu_memory_mode": "growth",
            "disable_gpu": True  # Disable for testing
        }
        
        locator = Locator(config)
        
        # Check that GPU options are set
        assert locator.config["use_mixed_precision"] == False  # Should be False when GPU disabled
        assert locator.config["gpu_batch_size"] == "auto"
        # use_efficient_pipeline option removed - always uses tf.data
        assert locator.config["gpu_memory_mode"] == "growth"
    
    def test_optimized_training_config(self):
        """Test optimized configuration creation."""
        base_config = {
            "out": "test",
            "batch_size": 32
        }
        
        optimized = create_optimized_training_config(base_config)
        
        # Check that GPU optimizations are added
        assert "use_mixed_precision" in optimized
        assert "gpu_batch_size" in optimized
        # use_efficient_pipeline option removed
        assert optimized["gpu_batch_size"] == "auto"
        # Always uses tf.data pipeline now
        
        # Check that original config is preserved
        assert optimized["out"] == "test"
        assert optimized["batch_size"] == 32


class TestBatchSizeOptimization:
    """Test dynamic batch size optimization."""
    
    @pytest.mark.skipif(not tf.config.list_physical_devices('GPU'), 
                        reason="GPU not available")
    def test_get_optimal_batch_size(self):
        """Test optimal batch size determination."""
        # Create simple model
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(256, input_shape=(1000,)),
            tf.keras.layers.Dense(256),
            tf.keras.layers.Dense(2)
        ])
        
        # Get optimal batch size
        optimal_size = GPUOptimizer.get_optimal_batch_size(
            model,
            input_shape=(1000,),
            target_memory_usage=0.8,
            min_batch_size=16,
            max_batch_size=512
        )
        
        # Should return a power of 2
        assert optimal_size & (optimal_size - 1) == 0  # Check if power of 2
        assert 16 <= optimal_size <= 512
    
    def test_batch_size_with_small_dataset(self):
        """Test batch size optimization with small dataset."""
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(10, input_shape=(100,)),
            tf.keras.layers.Dense(2)
        ])
        
        # Test with very small dataset (100 samples)
        batch_size = GPUOptimizer.get_optimal_batch_size(
            model,
            input_shape=(100,),
            min_batch_size=32,
            max_batch_size=2048,
            dataset_size=100
        )
        
        # Should be limited by dataset size (10% of 100 = 10, but min is 32)
        assert batch_size == 32
        
    def test_batch_size_with_medium_dataset(self):
        """Test batch size optimization with medium dataset."""
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(10, input_shape=(100,)),
            tf.keras.layers.Dense(2)
        ])
        
        # Test with medium dataset (1000 samples)
        batch_size = GPUOptimizer.get_optimal_batch_size(
            model,
            input_shape=(100,),
            min_batch_size=32,
            max_batch_size=2048,
            dataset_size=1000
        )
        
        # Should be limited to reasonable size (10% of 1000 = 100)
        assert batch_size <= 128  # Next power of 2 after 100
        
    def test_batch_size_no_gpu(self):
        """Test batch size optimization without GPU."""
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(10, input_shape=(100,)),
            tf.keras.layers.Dense(2)
        ])
        
        # Mock no GPU available
        with patch('tensorflow.config.list_physical_devices') as mock_devices:
            mock_devices.return_value = []
            
            batch_size = GPUOptimizer.get_optimal_batch_size(
                model,
                input_shape=(100,),
                min_batch_size=32,
                max_batch_size=2048
            )
            
            # Should return minimum batch size when no GPU
            assert batch_size == 32


if __name__ == "__main__":
    pytest.main([__file__, "-v"])