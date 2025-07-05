"""
Parallel analysis methods using Ray for multi-GPU execution.

This module provides Ray-based parallel implementations of analysis methods
from locator.analysis, enabling efficient multi-GPU utilization.
"""

import os
import sys
import tempfile
import pickle
import time
from typing import List, Optional, Dict, Any, Union
import numpy as np
import pandas as pd
from pathlib import Path

# Ray imports
import ray

# Import types for annotation
if sys.version_info >= (3, 8):
    from typing import TypedDict
else:
    from typing_extensions import TypedDict


@ray.remote(num_gpus=0.25)
def _ray_kfold_worker(
    fold_idx: int,
    gpu_id: int,
    data_file: str
) -> Dict[str, Any]:
    """
    Ray worker function that runs a single k-fold on a specific GPU.
    
    Args:
        fold_idx: Fold index
        gpu_id: GPU ID to use
        data_file: Path to pickled data file
    
    Returns:
        Dictionary with predictions and metadata
    """
    # Set GPU before importing TensorFlow
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    # Import inside worker to ensure proper GPU setup
    import tensorflow as tf
    from locator import Locator
    import allel
    
    # Suppress TF warnings
    tf.get_logger().setLevel('ERROR')
    
    print(f"Worker processing fold {fold_idx} on GPU {gpu_id}")
    
    # Load data from pickle file
    with open(data_file, 'rb') as f:
        data = pickle.load(f)
    
    # Reconstruct GenotypeArray
    gt_array = data['genotypes_array']
    shape = data['genotypes_shape']
    # FIXED: Simply reconstruct from the raw values
    genotypes = allel.GenotypeArray(gt_array)
    
    # Get fold's IndexSet
    index_set = data['fold_index_sets'][fold_idx]
    holdout_indices = index_set.test
    
    # Create Locator instance
    locator_config = data['config'].copy()
    locator_config['out'] = f"{locator_config['out']}_fold{fold_idx}"
    locator_config['disable_gpu'] = False
    locator_config['gpu_number'] = 0  # Use first visible GPU
    locator_config['keras_verbose'] = 0  # Suppress keras output
    
    # CRITICAL FIX: Store the sample data DataFrame in the config
    # This ensures sort_samples works correctly
    if '_sample_data_df' not in locator_config:
        locator_config['_sample_data_df'] = data['sample_data']
    
    locator = Locator(locator_config)  # Pass as dictionary
    
    # CRITICAL FIX: Ensure samples are set correctly BEFORE train_holdout
    # This must match the exact order used when creating the IndexSets
    locator.samples = data['samples']
    
    # Train with holdout
    start_time = time.time()
    history = locator.train_holdout(
        genotypes=genotypes,
        samples=data['samples'],  # Pass the same samples list
        holdout_indices=holdout_indices
    )
    train_time = time.time() - start_time
    
    # Make predictions
    predictions = locator.predict_holdout(
        verbose=False,
        return_df=True,
        save_preds_to_disk=False,
        plot_summary=False,
        plot_map=False
    )
    
    # CRITICAL: Verify sample IDs match expected holdout samples
    expected_samples = [data['samples'][i] for i in holdout_indices]
    actual_samples = predictions['sampleID'].tolist()
    
    if set(expected_samples) != set(actual_samples):
        print(f"WARNING: Sample mismatch in fold {fold_idx}!")
        print(f"Expected {len(expected_samples)} samples, got {len(actual_samples)}")
        print(f"First 5 expected: {expected_samples[:5]}")
        print(f"First 5 actual: {actual_samples[:5]}")
    
    # Clear keras session
    tf.keras.backend.clear_session()
    
    return {
        'fold': fold_idx,
        'gpu_id': gpu_id,
        'train_time': train_time,
        'predictions': predictions.to_dict('records'),
        'holdout_indices': holdout_indices.tolist(),
        'final_loss': float(history.history['loss'][-1]) if history else None
    }


def parallel_k_fold_holdouts(
    locator,
    genotypes,
    samples,
    k: int = 10,
    gpu_ids: List[int] = [0, 1],
    return_df: bool = True,
    save_full_pred_matrix: bool = True,
    verbose: bool = True,
    na_action: Optional[str] = None
) -> Union[pd.DataFrame, None]:
    """
    Run true k-fold cross-validation in parallel across multiple GPUs using Ray.
    
    This is a parallel version of AnalysisMixin.run_k_fold_holdouts() that distributes
    folds across available GPUs.
    
    Args:
        locator: Locator instance (for configuration and methods)
        genotypes: GenotypeArray
        samples: List of sample IDs
        k: Number of folds (holdout sets)
        gpu_ids: List of GPU IDs to use
        return_df: Whether to return DataFrame with all predictions
        save_full_pred_matrix: Whether to save full prediction matrix to disk
        verbose: Whether to show training progress and intermediate output
        na_action: How to handle NA samples ('separate', 'exclude', 'fail'). 
            If None, uses locator.na_action
    
    Returns:
        pandas.DataFrame or None: If return_df=True, returns DataFrame with one prediction 
            per held-out sample containing columns:
            - sampleID: Sample identifier
            - x_pred: Predicted longitude
            - y_pred: Predicted latitude
            - fold: Fold number (0 to k-1)
            
            Note: True locations are not included. To calculate prediction errors, merge
            the returned DataFrame with your sample metadata using the sampleID column.
    """
    # Initialize Ray if not already initialized
    if not ray.is_initialized():
        ray.init()
    
    # Use instance default if na_action not specified
    if na_action is None:
        na_action = locator.na_action
    
    # Get sample status
    status = locator.get_sample_status(samples)
    
    # Report status
    if verbose:
        print(f"K-fold CV: {status['n_known']} samples with coordinates, {status['n_na']} without")
        if status['n_na'] > 0:
            print(f"NA handling mode: {na_action}")
            if na_action == 'separate':
                print("Note: K-fold CV requires known locations; 'separate' behaves like 'exclude'")
    
    # Apply NA action
    if na_action == 'fail' and status['n_na'] > 0:
        raise ValueError(
            f"Found {status['n_na']} samples without coordinates. "
            f"Set na_action='separate' or 'exclude' to proceed."
        )
    
    # Get sample data and locations
    # CRITICAL: Use the same method as non-parallel version
    if hasattr(locator, "_sample_data_df"):
        sample_data, locs = locator.sort_samples(samples)
    else:
        sample_data_path = locator.config.get("sample_data")
        if not sample_data_path:
            raise ValueError("sample_data file path must be provided in config")
        sample_data, locs = locator.sort_samples(samples, sample_data_path)
    
    # Create NA mask
    na_mask = np.isnan(locs[:, 0])
    n_total_samples = len(locs)
    n_samples_with_coords = np.sum(~na_mask)
    
    if k > n_samples_with_coords:
        raise ValueError(
            f"k ({k}) must be less than or equal to number of samples with known locations ({n_samples_with_coords})"
        )
    
    # Import IndexSet
    from locator.data.indexset import IndexSet
    
    # Create list to store IndexSets for each fold
    # Use a fixed seed based on config seed or numpy's current state
    if 'seed' in locator.config and locator.config['seed'] is not None:
        kfold_seed = locator.config['seed']
    else:
        # Generate a seed from current numpy state to ensure consistency
        kfold_seed = np.random.randint(0, 2**31)
    
    fold_index_sets = []
    for fold_idx in range(k):
        index_set = IndexSet.from_k_fold(
            n=n_total_samples,
            k=k,
            fold=fold_idx,
            seed=kfold_seed,  # Use consistent seed for all folds
            na_mask=na_mask
        )
        fold_index_sets.append(index_set)
    
    # Pre-calculate KDE bandwidth if needed
    bandwidth_calculated = False
    original_bandwidth = None
    
    if (locator.config.get("weight_samples", {}).get("enabled", False) and
        locator.config.get("weight_samples", {}).get("method") == "KD"):
        
        existing_bandwidth = locator.config.get("weight_samples", {}).get("bandwidth")
        
        if existing_bandwidth is None:
            # Get all samples with coordinates for bandwidth calculation
            coords_mask = ~na_mask
            all_train_locs = locs[coords_mask]
            
            if len(all_train_locs) > 1:
                if verbose:
                    print("Pre-calculating optimal KDE bandwidth for k-fold CV...")
                
                from locator.sample_weights import get_global_bandwidth_optimizer
                optimizer = get_global_bandwidth_optimizer()
                
                optimal_bandwidth = optimizer.get_bandwidth(
                    all_train_locs,
                    cache_key=f"kfold_k{k}_n{len(all_train_locs)}",
                    n_bandwidths=locator.config.get("weight_samples", {}).get("n_bandwidths", 100),
                    verbose=verbose
                )
                
                # Store original value
                original_bandwidth = existing_bandwidth
                # Set in config
                locator.config["weight_samples"]["bandwidth"] = optimal_bandwidth
                bandwidth_calculated = True
                
                if verbose:
                    print(f"Using bandwidth: {optimal_bandwidth:.3f}")
    
    # Save data to temporary file
    with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.pkl') as f:
        data = {
            'genotypes_array': genotypes.values,  # FIXED: Save raw values, not to_n_alt()
            'genotypes_shape': genotypes.shape,
            'samples': samples,  # CRITICAL: Pass the original samples list
            'sample_data': sample_data,  # Pass the sorted sample data
            'locs': locs,
            'config': locator.config,
            'fold_index_sets': fold_index_sets,
            'na_mask': na_mask
        }
        pickle.dump(data, f)
        data_file = f.name
    
    if verbose:
        print(f"Running true {k}-fold cross-validation across GPUs {gpu_ids} using Ray...")
    
    start_time = time.time()
    
    # Submit all folds to Ray
    futures = []
    for fold_idx in range(k):
        gpu_id = gpu_ids[fold_idx % len(gpu_ids)]
        future = _ray_kfold_worker.remote(
            fold_idx=fold_idx,
            gpu_id=gpu_id,
            data_file=data_file
        )
        futures.append(future)
        if verbose:
            print(f"Submitted fold {fold_idx} to GPU {gpu_id}")
    
    # Wait for all folds to complete
    if verbose:
        print("\nWaiting for all folds to complete...")
    results = ray.get(futures)
    
    total_time = time.time() - start_time
    
    # Clean up
    os.unlink(data_file)
    
    if verbose:
        print(f"\nCompleted {k}-fold CV in {total_time:.1f}s ({total_time/k:.1f}s per fold)")
    
    # Restore original bandwidth setting if we changed it
    if bandwidth_calculated:
        if original_bandwidth is None:
            # Remove the key if it wasn't there originally
            locator.config.get("weight_samples", {}).pop("bandwidth", None)
        else:
            locator.config["weight_samples"]["bandwidth"] = original_bandwidth
    
    if return_df:
        # Build predictions DataFrame
        pred_rows = []
        for result in results:
            for pred in result['predictions']:
                pred_rows.append({
                    "sampleID": pred['sampleID'],
                    "x_pred": pred['x_pred'],
                    "y_pred": pred['y_pred'],
                    "fold": result['fold']
                })
        
        all_predictions = pd.DataFrame(pred_rows)
        
        # Verify we have predictions for all expected samples
        expected_samples = set(samples[i] for i in range(len(samples)) if not na_mask[i])
        actual_samples = set(all_predictions['sampleID'].unique())
        
        if expected_samples != actual_samples:
            print(f"WARNING: Sample mismatch in final results!")
            print(f"Expected {len(expected_samples)} unique samples")
            print(f"Got {len(actual_samples)} unique samples")
            missing = expected_samples - actual_samples
            extra = actual_samples - expected_samples
            if missing:
                print(f"Missing samples: {list(missing)[:10]}...")
            if extra:
                print(f"Extra samples: {list(extra)[:10]}...")
        
        if save_full_pred_matrix:
            all_predictions.to_csv(
                f"{locator.config['out']}_kfold_holdouts_predlocs.csv", index=False
            )
        
        return all_predictions
    
    return None




def parallel_leave_one_out(
    locator,
    genotypes,
    samples,
    gpu_ids: List[int] = [0, 1],
    return_df: bool = True,
    save_full_pred_matrix: bool = True,
    na_action: Optional[str] = None
) -> Union[pd.DataFrame, None]:
    """
    Perform leave-one-out cross-validation in parallel across multiple GPUs.
    
    This is a parallel version of AnalysisMixin.run_leave_one_out() that uses
    Ray to distribute the computation. It's a convenience wrapper around 
    parallel_k_fold_holdouts with k equal to the number of samples with known locations.
    
    Args:
        locator: Locator instance (for configuration and methods)
        genotypes: Array of genotype data
        samples: Sample IDs corresponding to genotypes
        gpu_ids: List of GPU IDs to use
        return_df: Whether to return DataFrame with all predictions
        save_full_pred_matrix: Whether to save full prediction matrix to disk
        na_action: How to handle NA samples ('separate', 'exclude', 'fail'). 
            If None, uses locator.na_action
    
    Returns:
        pandas.DataFrame or None: DataFrame with predictions for each left-out sample
    """
    # Get sample status to determine k
    status = locator.get_sample_status(samples)
    n_known = status['n_known']
    
    if n_known == 0:
        raise ValueError("No samples with known coordinates for leave-one-out CV")
    
    print(f"Running leave-one-out cross-validation for {n_known} samples across GPUs {gpu_ids}")
    
    # Run k-fold with k equal to number of known samples
    # This will create folds with exactly 1 sample each
    result = parallel_k_fold_holdouts(
        locator=locator,
        genotypes=genotypes,
        samples=samples,
        k=n_known,
        gpu_ids=gpu_ids,
        return_df=return_df,
        save_full_pred_matrix=False,  # We'll save with our own name
        verbose=False,  # We already printed our message
        na_action=na_action
    )
    
    # Save with leave-one-out specific filename if requested
    if result is not None and save_full_pred_matrix:
        result.to_csv(
            f"{locator.config['out']}_leave_one_out_predlocs.csv", index=False
        )
    
    return result


# Additional parallel methods that could be implemented:
# - parallel_holdouts() - for run_holdouts() with random replicates
# - parallel_jacknife_holdouts() - for run_jacknife_holdouts()
# - parallel_windows_holdouts() - for run_windows_holdouts()
# 
# These would follow the same pattern:
# 1. Use locator methods for data validation and preprocessing
# 2. Create IndexSets or fold specifications
# 3. Pre-calculate KDE bandwidth if needed
# 4. Distribute work across GPUs using Ray remote functions
# 5. Restore configuration and return results in expected format