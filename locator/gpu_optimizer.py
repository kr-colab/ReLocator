"""GPU optimization utilities for Locator.

This module provides utilities to maximize GPU efficiency and speed for
deep learning genomic predictions.
"""

import os
import warnings
from typing import Optional, Tuple, Dict, Any
import numpy as np
import tensorflow as tf


class GPUOptimizer:
    """Utilities for optimizing GPU performance in TensorFlow."""
    
    @staticmethod
    def setup_mixed_precision():
        """Enable mixed precision training for 2x speedup on modern GPUs.
        
        Returns:
            bool: True if mixed precision was enabled successfully
        """
        try:
            # Check if GPU supports mixed precision (compute capability >= 7.0)
            gpus = tf.config.list_physical_devices('GPU')
            if not gpus:
                return False
                
            # Get compute capability
            gpu_details = tf.config.experimental.get_device_details(gpus[0])
            compute_capability = gpu_details.get('compute_capability', (0, 0))
            
            if compute_capability[0] >= 7:  # Tensor Core support
                policy = tf.keras.mixed_precision.Policy('mixed_float16')
                tf.keras.mixed_precision.set_global_policy(policy)
                print(f"Mixed precision training enabled (compute capability {compute_capability})")
                return True
            else:
                print(f"GPU compute capability {compute_capability} doesn't support mixed precision efficiently")
                return False
                
        except Exception as e:
            warnings.warn(f"Failed to enable mixed precision: {e}")
            return False
    
    @staticmethod
    def get_optimal_batch_size(model: tf.keras.Model, 
                             input_shape: Tuple[int, ...],
                             target_memory_usage: float = 0.9,
                             min_batch_size: int = 32,
                             max_batch_size: int = 2048) -> int:
        """Dynamically determine optimal batch size for GPU memory.
        
        Args:
            model: Keras model to optimize for
            input_shape: Shape of single input sample (excluding batch dimension)
            target_memory_usage: Target GPU memory usage (0.0-1.0)
            min_batch_size: Minimum batch size to test
            max_batch_size: Maximum batch size to test
            
        Returns:
            int: Optimal batch size for current GPU
        """
        gpus = tf.config.list_physical_devices('GPU')
        if not gpus:
            return min_batch_size
            
        # Get available GPU memory
        try:
            gpu_memory = tf.config.experimental.get_memory_info('GPU:0')
            available_memory = gpu_memory['current'] * target_memory_usage
        except:
            # Fallback: estimate based on typical GPU sizes
            available_memory = 8 * 1024 * 1024 * 1024 * target_memory_usage  # 8GB default
        
        # Binary search for optimal batch size
        left, right = min_batch_size, max_batch_size
        optimal_batch_size = min_batch_size
        
        while left <= right:
            test_batch_size = (left + right) // 2
            
            try:
                # Create dummy data and test forward pass
                dummy_input = tf.random.normal((test_batch_size,) + input_shape)
                
                # Clear any previous allocations
                tf.keras.backend.clear_session()
                
                # Test forward and backward pass
                with tf.GradientTape() as tape:
                    output = model(dummy_input, training=True)
                    loss = tf.reduce_mean(output)
                
                # Test gradient computation
                gradients = tape.gradient(loss, model.trainable_variables)
                
                # If successful, try larger batch
                optimal_batch_size = test_batch_size
                left = test_batch_size + 1
                
            except tf.errors.ResourceExhaustedError:
                # If OOM, try smaller batch
                right = test_batch_size - 1
            except Exception as e:
                # Other errors, try smaller batch
                warnings.warn(f"Error testing batch size {test_batch_size}: {e}")
                right = test_batch_size - 1
        
        # Clear session after testing
        tf.keras.backend.clear_session()
        
        # Round to nearest power of 2 for efficiency
        optimal_batch_size = 2 ** int(np.log2(optimal_batch_size))
        
        print(f"Optimal batch size determined: {optimal_batch_size}")
        return optimal_batch_size
    
    @staticmethod
    def create_efficient_dataset(X: np.ndarray,
                               y: np.ndarray,
                               batch_size: int,
                               training: bool = True,
                               cache: bool = True,
                               num_parallel_calls: int = tf.data.AUTOTUNE) -> tf.data.Dataset:
        """Create an efficient tf.data pipeline with GPU optimization.
        
        Args:
            X: Input features
            y: Target values
            batch_size: Batch size
            training: Whether this is for training (enables shuffling)
            cache: Whether to cache data in memory
            num_parallel_calls: Number of parallel preprocessing calls
            
        Returns:
            tf.data.Dataset: Optimized dataset
        """
        # Convert to float32 (or float16 if mixed precision is enabled)
        policy = tf.keras.mixed_precision.global_policy()
        dtype = policy.compute_dtype
        
        # Create dataset from tensor slices
        dataset = tf.data.Dataset.from_tensor_slices((
            tf.constant(X, dtype=dtype),
            tf.constant(y, dtype=tf.float32)  # Always float32 for coordinates
        ))
        
        # Cache before shuffling for better performance
        if cache:
            dataset = dataset.cache()
        
        # Shuffle with appropriate buffer size
        if training:
            buffer_size = min(len(X), 10000)
            dataset = dataset.shuffle(buffer_size, reshuffle_each_iteration=True)
        
        # Batch with drop_remainder for consistent batch sizes (better GPU utilization)
        dataset = dataset.batch(batch_size, drop_remainder=training)
        
        # Prefetch with autotune for optimal performance
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        # Enable parallel processing options
        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
        options.experimental_optimization.apply_default_optimizations = True
        options.experimental_optimization.map_parallelization = True
        dataset = dataset.with_options(options)
        
        return dataset
    
    @staticmethod
    def optimize_gpu_memory(mode: str = "growth", memory_limit: Optional[int] = None):
        """Configure GPU memory allocation strategy.
        
        Args:
            mode: Memory allocation mode ('growth', 'preallocate', 'limit')
            memory_limit: Memory limit in MB (only used with mode='limit')
        """
        gpus = tf.config.list_physical_devices('GPU')
        if not gpus:
            return
            
        for gpu in gpus:
            try:
                if mode == "growth":
                    tf.config.experimental.set_memory_growth(gpu, True)
                elif mode == "preallocate":
                    tf.config.experimental.set_memory_growth(gpu, False)
                elif mode == "limit" and memory_limit:
                    tf.config.set_logical_device_configuration(
                        gpu,
                        [tf.config.LogicalDeviceConfiguration(memory_limit=memory_limit)]
                    )
            except RuntimeError as e:
                warnings.warn(f"GPU memory configuration failed: {e}")
    
    @staticmethod
    def enable_xla_compilation():
        """Enable XLA compilation for additional performance.
        
        Note: This is experimental and may not work with all operations.
        """
        tf.config.optimizer.set_jit(True)
        print("XLA compilation enabled (experimental)")
    
    @staticmethod
    def get_gpu_info() -> Dict[str, Any]:
        """Get information about available GPUs.
        
        Returns:
            Dict containing GPU information
        """
        gpus = tf.config.list_physical_devices('GPU')
        info = {
            'gpu_count': len(gpus),
            'gpus': []
        }
        
        for i, gpu in enumerate(gpus):
            gpu_info = {
                'index': i,
                'name': gpu.name,
            }
            
            try:
                details = tf.config.experimental.get_device_details(gpu)
                gpu_info.update(details)
                
                # Get memory info if available
                memory_info = tf.config.experimental.get_memory_info(f'GPU:{i}')
                gpu_info['memory'] = memory_info
            except:
                pass
                
            info['gpus'].append(gpu_info)
            
        return info


class GradientAccumulator:
    """Helper class for gradient accumulation to simulate larger batch sizes."""
    
    def __init__(self, model: tf.keras.Model, accumulation_steps: int = 4):
        """Initialize gradient accumulator.
        
        Args:
            model: Keras model to accumulate gradients for
            accumulation_steps: Number of steps to accumulate before updating
        """
        self.model = model
        self.accumulation_steps = accumulation_steps
        self.reset()
    
    def reset(self):
        """Reset accumulated gradients."""
        self.accumulated_gradients = [
            tf.Variable(tf.zeros_like(var), trainable=False)
            for var in self.model.trainable_variables
        ]
        self.accumulated_loss = tf.Variable(0.0, trainable=False)
        self.step_count = tf.Variable(0, trainable=False)
    
    @tf.function
    def accumulate_step(self, X, y, loss_fn):
        """Perform one accumulation step.
        
        Args:
            X: Input batch
            y: Target batch
            loss_fn: Loss function to use
            
        Returns:
            Current batch loss
        """
        with tf.GradientTape() as tape:
            predictions = self.model(X, training=True)
            loss = loss_fn(y, predictions)
            
            # Scale loss by accumulation steps
            scaled_loss = loss / self.accumulation_steps
        
        # Calculate gradients
        gradients = tape.gradient(scaled_loss, self.model.trainable_variables)
        
        # Accumulate gradients
        for acc_grad, grad in zip(self.accumulated_gradients, gradients):
            if grad is not None:
                acc_grad.assign_add(grad)
        
        # Accumulate loss
        self.accumulated_loss.assign_add(loss)
        self.step_count.assign_add(1)
        
        return loss
    
    def apply_gradients(self, optimizer):
        """Apply accumulated gradients if ready.
        
        Args:
            optimizer: Keras optimizer to use
            
        Returns:
            bool: Whether gradients were applied
        """
        if self.step_count >= self.accumulation_steps:
            # Apply accumulated gradients
            optimizer.apply_gradients(
                zip(self.accumulated_gradients, self.model.trainable_variables)
            )
            
            # Reset accumulator
            self.reset()
            return True
        
        return False


def create_optimized_training_config(base_config: Dict[str, Any]) -> Dict[str, Any]:
    """Create an optimized configuration for GPU training.
    
    Args:
        base_config: Base configuration dictionary
        
    Returns:
        Optimized configuration dictionary
    """
    optimized_config = base_config.copy()
    
    # GPU optimization defaults
    gpu_defaults = {
        'use_mixed_precision': True,
        'gpu_batch_size': 'auto',  # Will be determined dynamically
        'use_efficient_pipeline': True,
        'gradient_accumulation_steps': 1,  # Increase for larger effective batch size
        'gpu_memory_mode': 'growth',
        'enable_xla': False,  # Experimental
        'prefetch_buffer': tf.data.AUTOTUNE,
        'shuffle_buffer': 10000,
    }
    
    # Update config with GPU optimizations
    for key, value in gpu_defaults.items():
        if key not in optimized_config:
            optimized_config[key] = value
    
    return optimized_config