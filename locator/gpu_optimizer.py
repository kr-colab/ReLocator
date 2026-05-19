"""GPU optimization utilities for Locator.

This module provides utilities to maximize GPU efficiency and speed for
deep learning genomic predictions.
"""

import warnings
from typing import Optional, Tuple

import numpy as np
import tensorflow as tf


class GPUOptimizer:
    """Utilities for optimizing GPU performance in TensorFlow."""

    @staticmethod
    def setup_mixed_precision():
        """Enable mixed precision training for 2x speedup on modern GPUs.

        Returns
        -------
            bool: True if mixed precision was enabled successfully
        """
        try:
            # Check if GPU supports mixed precision (compute capability >= 7.0)
            gpus = tf.config.list_physical_devices("GPU")
            if not gpus:
                return False

            # Get compute capability
            gpu_details = tf.config.experimental.get_device_details(gpus[0])
            compute_capability = gpu_details.get("compute_capability", (0, 0))

            if compute_capability[0] >= 7:  # Tensor Core support
                policy = tf.keras.mixed_precision.Policy("mixed_float16")
                tf.keras.mixed_precision.set_global_policy(policy)
                print(
                    f"Mixed precision training enabled (compute capability {compute_capability})"
                )
                return True
            else:
                print(
                    f"GPU compute capability {compute_capability} doesn't support mixed precision efficiently"
                )
                return False

        except Exception as e:
            warnings.warn(f"Failed to enable mixed precision: {e}")
            return False

    @staticmethod
    def get_optimal_batch_size(  # noqa: C901
        model: tf.keras.Model,
        input_shape: Tuple[int, ...],
        target_memory_usage: float = 0.9,
        min_batch_size: int = 32,
        max_batch_size: int = 2048,
        dataset_size: Optional[int] = None,
        verbose: bool = True,
    ) -> int:
        """Dynamically determine optimal batch size for GPU memory.

        Args:
            model: Keras model to optimize for
            input_shape: Shape of single input sample (excluding batch dimension)
            target_memory_usage: Target GPU memory usage (0.0-1.0)
            min_batch_size: Minimum batch size to test
            max_batch_size: Maximum batch size to test
            dataset_size: Size of the dataset (if provided, limits max batch size)

        Returns
        -------
            int: Optimal batch size for current GPU
        """
        gpus = tf.config.list_physical_devices("GPU")
        if not gpus:
            return min_batch_size

        # Limit max batch size based on dataset size
        if dataset_size is not None:
            # Don't use batch size larger than 10% of dataset
            max_reasonable_batch = max(min_batch_size, dataset_size // 10)
            max_batch_size = min(max_batch_size, max_reasonable_batch)
            if max_batch_size < 2048 and verbose:
                print(
                    f"Limiting max batch size to {max_batch_size} based on dataset size {dataset_size}"
                )

        # Get available GPU memory
        # Note: The available_memory calculation is commented out but preserved
        # for future use. It would estimate GPU memory for batch size optimization.
        # After tf.config.set_visible_devices() or CUDA_VISIBLE_DEVICES is set,
        # the selected GPU is always accessible as 'GPU:0' from TensorFlow's perspective.

        # try:
        #     gpu_memory = tf.config.experimental.get_memory_info("GPU:0")
        #     available_memory = gpu_memory["current"] * target_memory_usage
        # except Exception:
        #     # Fallback: use conservative estimate
        #     # Most consumer GPUs have 8-24GB, datacenter GPUs 40-80GB
        #     gpu_name = gpus[0].name.lower()
        #     if "a100" in gpu_name or "a6000" in gpu_name:
        #         available_memory = 40 * 1024 * 1024 * 1024 * target_memory_usage  # 40GB
        #     elif "v100" in gpu_name or "3090" in gpu_name or "4090" in gpu_name:
        #         available_memory = 24 * 1024 * 1024 * 1024 * target_memory_usage  # 24GB
        #     else:
        #         available_memory = 8 * 1024 * 1024 * 1024 * target_memory_usage  # 8GB default
        #     if verbose:
        #         print(f"Using estimated GPU memory for {gpus[0].name}")

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
                _ = tape.gradient(loss, model.trainable_variables)

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
        optimal_batch_size = max(optimal_batch_size, 1)
        optimal_batch_size = 2 ** int(np.log2(optimal_batch_size))

        # Final check against dataset size
        if dataset_size is not None and optimal_batch_size > dataset_size // 10:
            # For small datasets, use a more conservative batch size
            optimal_batch_size = min(optimal_batch_size, max(32, dataset_size // 16))
            if verbose:
                print(f"Adjusted batch size for small dataset: {optimal_batch_size}")

        if verbose:
            print(f"Optimal batch size determined: {optimal_batch_size}")
        return optimal_batch_size

    @staticmethod
    def optimize_gpu_memory(mode: str = "growth", memory_limit: Optional[int] = None):
        """Configure GPU memory allocation strategy.

        Args:
            mode: Memory allocation mode ('growth', 'preallocate', 'limit')
            memory_limit: Memory limit in MB (only used with mode='limit')
        """
        gpus = tf.config.list_physical_devices("GPU")
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
                        [
                            tf.config.LogicalDeviceConfiguration(
                                memory_limit=memory_limit
                            )
                        ],
                    )
            except (RuntimeError, ValueError) as e:
                # Both can fire when GPU is already configured — e.g. a Ray
                # actor that installed a memory cap before Locator init runs.
                warnings.warn(f"GPU memory configuration failed: {e}")

    @staticmethod
    def enable_xla_compilation():
        """Enable XLA compilation for additional performance.

        Note: This is experimental and may not work with all operations.
        """
        tf.config.optimizer.set_jit(True)
        print("XLA compilation enabled (experimental)")
