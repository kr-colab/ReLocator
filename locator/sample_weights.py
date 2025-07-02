"""
Sample weighting functionality for Locator.

This module provides various methods for calculating sample weights based on
geographic distribution, including KDE-based weights with bandwidth optimization,
histogram-based weights, and loading pre-calculated weights.
"""

import numpy as np
import pandas as pd
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
import warnings
from typing import Optional, Dict, Union, Tuple, Any


class BandwidthOptimizer:
    """
    Manages bandwidth calculation and caching for KDE weights.
    
    This class provides optimized bandwidth selection by caching results
    to avoid redundant grid searches across multiple analyses.
    """
    
    def __init__(self):
        """Initialize the bandwidth optimizer with empty cache."""
        self._cache = {}
        
    def get_bandwidth(
        self, 
        locations: np.ndarray,
        cache_key: Optional[str] = None,
        bandwidth: Optional[float] = None,
        n_bandwidths: int = 100,
        min_bw: float = 0.1,
        max_bw: float = 10.0,
        cv: int = 5,
        n_jobs: int = -1,
        verbose: bool = False
    ) -> float:
        """
        Get optimal bandwidth, using cache if available.
        
        Args:
            locations: Array of shape (n_samples, 2) with x, y coordinates
            cache_key: Key for caching results. If None, creates key from data hash
            bandwidth: Pre-specified bandwidth. If provided, returns this value
            n_bandwidths: Number of bandwidth values to test
            min_bw: Minimum bandwidth value
            max_bw: Maximum bandwidth value  
            cv: Number of cross-validation folds
            n_jobs: Number of parallel jobs (-1 for all cores)
            verbose: Whether to print progress
            
        Returns:
            Optimal bandwidth value
        """
        # Return pre-specified bandwidth if provided
        if bandwidth is not None:
            return bandwidth
            
        # Create cache key if not provided
        if cache_key is None:
            # Use data characteristics for cache key
            cache_key = f"n={len(locations)}_mean={locations.mean():.3f}_std={locations.std():.3f}"
            
        # Check cache
        if cache_key in self._cache:
            if verbose:
                print(f"Using cached bandwidth for key '{cache_key}': {self._cache[cache_key]:.3f}")
            return self._cache[cache_key]
            
        # Calculate optimal bandwidth
        if verbose:
            print(f"Calculating optimal bandwidth ({n_bandwidths} values from {min_bw} to {max_bw})...")
            
        bandwidths = np.linspace(min_bw, max_bw, n_bandwidths)
        grid = GridSearchCV(
            KernelDensity(kernel='gaussian'),
            {'bandwidth': bandwidths},
            cv=cv,
            n_jobs=n_jobs
        )
        grid.fit(locations)
        
        optimal_bw = grid.best_params_['bandwidth']
        
        # Cache result
        self._cache[cache_key] = optimal_bw
        
        if verbose:
            print(f"Optimal bandwidth: {optimal_bw:.3f} (CV score: {grid.best_score_:.3f})")
            
        return optimal_bw
        
    def clear_cache(self, cache_key: Optional[str] = None):
        """
        Clear bandwidth cache.
        
        Args:
            cache_key: Specific key to clear. If None, clears entire cache
        """
        if cache_key is None:
            self._cache.clear()
        elif cache_key in self._cache:
            del self._cache[cache_key]


# Global optimizer instance for shared caching across analyses
_global_optimizer = None


def get_global_bandwidth_optimizer() -> BandwidthOptimizer:
    """Get or create the global bandwidth optimizer instance."""
    global _global_optimizer
    if _global_optimizer is None:
        _global_optimizer = BandwidthOptimizer()
    return _global_optimizer


def calculate_optimal_bandwidth(
    locations: np.ndarray,
    n_bandwidths: int = 100,
    min_bw: float = 0.1,
    max_bw: float = 10.0,
    cv: int = 5,
    n_jobs: int = -1,
    verbose: bool = False
) -> Tuple[float, Dict[str, Any]]:
    """
    Calculate the optimal KDE bandwidth for a set of locations.
    
    This is a standalone function for one-off bandwidth calculation without caching.
    
    Args:
        locations: Array of shape (n_samples, 2) with x, y coordinates
        n_bandwidths: Number of bandwidth values to test
        min_bw: Minimum bandwidth value
        max_bw: Maximum bandwidth value
        cv: Number of cross-validation folds
        n_jobs: Number of parallel jobs (-1 for all cores)
        verbose: Whether to print progress
        
    Returns:
        Tuple of (optimal_bandwidth, info_dict) where info_dict contains:
            - 'bandwidth': optimal bandwidth value
            - 'cv_scores': cross-validation scores for all tested bandwidths
            - 'bandwidths_tested': array of bandwidth values tested
            - 'best_score': best cross-validation score
    """
    if len(locations) < 2:
        raise ValueError("Need at least 2 locations to calculate bandwidth")
        
    if verbose:
        print(f"Calculating optimal KDE bandwidth using {n_bandwidths} values from {min_bw} to {max_bw}")
    
    bandwidths = np.linspace(min_bw, max_bw, n_bandwidths)
    
    grid = GridSearchCV(
        KernelDensity(kernel='gaussian'),
        {'bandwidth': bandwidths},
        cv=cv,
        n_jobs=n_jobs
    )
    grid.fit(locations)
    
    optimal_bandwidth = grid.best_params_['bandwidth']
    best_score = grid.best_score_
    
    if verbose:
        print(f"Optimal bandwidth: {optimal_bandwidth:.3f} (CV score: {best_score:.3f})")
    
    return optimal_bandwidth, {
        'bandwidth': optimal_bandwidth,
        'cv_scores': grid.cv_results_['mean_test_score'],
        'bandwidths_tested': bandwidths,
        'best_score': best_score
    }


def weight_samples(
    method: str,
    trainlocs: Optional[np.ndarray] = None,
    trainsamps: Optional[np.ndarray] = None,
    weightdf: Optional[pd.DataFrame] = None,
    xbins: Optional[int] = None,
    ybins: Optional[int] = None,
    lam: Optional[float] = None,
    bandwidth: Optional[float] = None,
    cache_bandwidth: bool = True,
    n_bandwidths: int = 100
) -> Dict[str, Any]:
    """
    Calculate weights for training data based on the specified method.
    
    Args:
        method: Method for calculating weights ('KD', 'histogram', or 'load')
        trainlocs: Training locations (required for KD and histogram methods)
        trainsamps: Training sample IDs
        weightdf: DataFrame containing pre-calculated sample weights
        xbins: Number of bins in x direction for histogram method
        ybins: Number of bins in y direction for histogram method
        lam: Exponent for KDE weights
        bandwidth: Bandwidth for KDE (if None, will be calculated)
        cache_bandwidth: Whether to use bandwidth caching for KDE
        n_bandwidths: Number of bandwidth values to test if calculating
        
    Returns:
        Dictionary containing:
            - 'method': weighting method used
            - 'sample_weights': array of weights
            - 'sample_weights_df': DataFrame with sampleID and weights
            - method-specific parameters
    """
    if method == 'KD':
        if trainlocs is None:
            raise ValueError("trainlocs required for KD method")
            
        weights = _make_kd_weights(
            trainlocs, 
            lam=1.0 if lam is None else lam,
            bandwidth=bandwidth,
            cache_bandwidth=cache_bandwidth,
            n_bandwidths=n_bandwidths
        )
        df = pd.DataFrame({
            'sampleID': trainsamps,
            'sample_weight': weights
        })
        
    elif method == 'histogram':
        if trainlocs is None:
            raise ValueError("trainlocs required for histogram method")
            
        weights = _make_histogram_weights(
            trainlocs, 
            xbins=10 if xbins is None else xbins,
            ybins=10 if ybins is None else ybins
        )
        df = pd.DataFrame({
            'sampleID': trainsamps,
            'sample_weight': weights
        })
        
    elif method == 'load':
        if weightdf is None:
            raise ValueError("weightdf required for load method")
            
        df = _load_sample_weights(weightdf, trainsamps)
        weights = df['sample_weight'].values
        
    else:
        raise ValueError("Invalid method. Choose 'KD', 'histogram', or 'load'.")
        
    return {
        'method': method,
        'sample_weights': weights,
        'sample_weights_df': df,
        'xbins': xbins,
        'ybins': ybins,
        'lam': lam,
        'bandwidth': bandwidth,
    }


def _make_kd_weights(
    trainlocs: np.ndarray,
    lam: float = 1.0,
    bandwidth: Optional[float] = None,
    cache_bandwidth: bool = True,
    n_bandwidths: int = 100
) -> np.ndarray:
    """
    Calculate weights using Kernel Density Estimation with optimized bandwidth selection.
    
    Args:
        trainlocs: Training locations, shape (n_samples, 2)
        lam: Exponent for weights
        bandwidth: Pre-specified bandwidth. If None, will be calculated
        cache_bandwidth: Whether to use bandwidth caching
        n_bandwidths: Number of bandwidth values to test
        
    Returns:
        Array of normalized weights
    """
    # Get bandwidth (from cache, parameter, or calculate)
    if bandwidth is None and cache_bandwidth:
        optimizer = get_global_bandwidth_optimizer()
        bw = optimizer.get_bandwidth(trainlocs, n_bandwidths=n_bandwidths)
    elif bandwidth is None:
        # Calculate without caching
        bw, _ = calculate_optimal_bandwidth(trainlocs, n_bandwidths=n_bandwidths, verbose=False)
    else:
        bw = bandwidth
    
    # Fit kernel with determined bandwidth
    kde = KernelDensity(bandwidth=bw, kernel='gaussian')
    kde.fit(trainlocs)
    
    # Calculate weights
    weights = kde.score_samples(trainlocs)
    weights = 1.0 / np.exp(weights)
    weights /= min(weights)
    weights = np.power(weights, lam)
    weights /= sum(weights)
    
    return weights


def _make_histogram_weights(trainlocs: np.ndarray, xbins: int = 10, ybins: int = 10) -> np.ndarray:
    """
    Calculate weights using 2D histogram binning.
    
    Args:
        trainlocs: Training locations, shape (n_samples, 2)
        xbins: Number of bins in x direction
        ybins: Number of bins in y direction
        
    Returns:
        Array of weights based on inverse bin density
    """
    bincount = [xbins, ybins]
    
    # Make 2D histogram
    H, xedges, yedges = np.histogram2d(trainlocs[:, 0], trainlocs[:, 1], bins=bincount)
    
    # Sort trainlocs into bins
    xbin = np.digitize(trainlocs[:, 0], xedges[1:], right=True)
    ybin = np.digitize(trainlocs[:, 1], yedges[1:], right=True)
    
    # Assign sample weights
    weights = np.empty(len(trainlocs), dtype='float')
    for i in range(len(trainlocs)):
        weights[i] = 1 / (H[xbin[i]][ybin[i]])
        
    weights /= min(weights)
    
    return weights


def _load_sample_weights(weightdf: pd.DataFrame, trainsamps: list) -> pd.DataFrame:
    """
    Load pre-calculated sample weights from a DataFrame.
    
    Args:
        weightdf: DataFrame with columns 'sampleID' and 'sample_weight'
        trainsamps: List of training sample IDs
        
    Returns:
        DataFrame with sample weights for training samples
    """
    if 'sampleID' not in weightdf.columns or 'sample_weight' not in weightdf.columns:
        raise ValueError("weightdf must contain 'sampleID' and 'sample_weight' columns")
        
    # Create a copy to avoid modifying original
    df = weightdf.copy()
    df.set_index('sampleID', inplace=True)
    
    # Extract weights for training samples
    weights = []
    for samp in trainsamps:
        if samp not in df.index:
            raise ValueError(f"Sample '{samp}' not found in weight DataFrame")
        w = df.loc[samp, 'sample_weight']
        if isinstance(w, pd.Series):
            weights.append(w.iloc[0])
        else:
            weights.append(w)
            
    return pd.DataFrame({
        'sampleID': trainsamps,
        'sample_weight': weights
    })