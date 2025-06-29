"""
Unit tests for sample_weights module.
"""

import unittest
import numpy as np
import pandas as pd
import time

from locator.sample_weights import (
    BandwidthOptimizer,
    calculate_optimal_bandwidth,
    get_global_bandwidth_optimizer,
    weight_samples,
    _make_kd_weights,
    _make_histogram_weights,
    _load_sample_weights
)


class TestBandwidthOptimizer(unittest.TestCase):
    """Test the BandwidthOptimizer class."""
    
    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        self.locations = np.random.randn(100, 2) * 10 + [30, 40]
        
    def test_bandwidth_calculation(self):
        """Test basic bandwidth calculation."""
        optimizer = BandwidthOptimizer()
        bandwidth = optimizer.get_bandwidth(
            self.locations,
            n_bandwidths=10,  # Small for fast testing
            min_bw=0.1,
            max_bw=5.0
        )
        
        self.assertGreaterEqual(bandwidth, 0.1)
        self.assertLessEqual(bandwidth, 5.0)
        
    def test_caching(self):
        """Test that bandwidth is cached properly."""
        optimizer = BandwidthOptimizer()
        
        # First call - should calculate
        start = time.time()
        bw1 = optimizer.get_bandwidth(self.locations, cache_key='test', n_bandwidths=20)
        calc_time = time.time() - start
        
        # Second call - should use cache
        start = time.time()
        bw2 = optimizer.get_bandwidth(self.locations, cache_key='test', n_bandwidths=20)
        cache_time = time.time() - start
        
        self.assertEqual(bw1, bw2)
        self.assertLess(cache_time * 10, calc_time)
        
    def test_manual_bandwidth_override(self):
        """Test that manual bandwidth specification works."""
        optimizer = BandwidthOptimizer()
        manual_bw = 2.5
        
        bw = optimizer.get_bandwidth(self.locations, bandwidth=manual_bw)
        self.assertEqual(bw, manual_bw)
        
    def test_clear_cache(self):
        """Test cache clearing."""
        optimizer = BandwidthOptimizer()
        
        # Add to cache
        optimizer.get_bandwidth(self.locations, cache_key='test1')
        optimizer.get_bandwidth(self.locations * 2, cache_key='test2')
        
        # Clear specific key
        optimizer.clear_cache('test1')
        self.assertNotIn('test1', optimizer._cache)
        self.assertIn('test2', optimizer._cache)
        
        # Clear all
        optimizer.clear_cache()
        self.assertEqual(len(optimizer._cache), 0)


class TestCalculateOptimalBandwidth(unittest.TestCase):
    """Test the standalone calculate_optimal_bandwidth function."""
    
    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        self.locations = np.random.randn(50, 2) * 5 + [20, 30]
        
    def test_basic_calculation(self):
        """Test basic bandwidth calculation."""
        bandwidth, info = calculate_optimal_bandwidth(
            self.locations,
            n_bandwidths=10,
            min_bw=0.5,
            max_bw=3.0
        )
        
        self.assertGreater(bandwidth, 0.5)
        self.assertLess(bandwidth, 3.0)
        self.assertIn('cv_scores', info)
        self.assertIn('bandwidths_tested', info)
        self.assertEqual(len(info['bandwidths_tested']), 10)
        
    def test_insufficient_data(self):
        """Test error with insufficient data."""
        with self.assertRaises(ValueError):
            calculate_optimal_bandwidth(np.array([[1, 2]]))


class TestWeightSamples(unittest.TestCase):
    """Test the main weight_samples function."""
    
    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        self.n_samples = 50
        self.locations = np.random.randn(self.n_samples, 2) * 10 + [30, 40]
        self.sample_ids = [f"sample_{i}" for i in range(self.n_samples)]
        
    def test_kd_weights(self):
        """Test KDE weight calculation."""
        result = weight_samples(
            method='KD',
            trainlocs=self.locations,
            trainsamps=self.sample_ids,
            bandwidth=2.0  # Fixed for reproducibility
        )
        
        self.assertEqual(result['method'], 'KD')
        self.assertEqual(len(result['sample_weights']), self.n_samples)
        self.assertAlmostEqual(np.sum(result['sample_weights']), 1.0, places=5)
        
    def test_histogram_weights(self):
        """Test histogram weight calculation."""
        result = weight_samples(
            method='histogram',
            trainlocs=self.locations,
            trainsamps=self.sample_ids,
            xbins=5,
            ybins=5
        )
        
        self.assertEqual(result['method'], 'histogram')
        self.assertEqual(len(result['sample_weights']), self.n_samples)
        self.assertGreater(np.min(result['sample_weights']), 0)
        
    def test_load_weights(self):
        """Test loading pre-calculated weights."""
        # Create weight DataFrame
        weights_df = pd.DataFrame({
            'sampleID': self.sample_ids,
            'sample_weight': np.random.rand(self.n_samples)
        })
        
        result = weight_samples(
            method='load',
            trainsamps=self.sample_ids,
            weightdf=weights_df
        )
        
        self.assertEqual(result['method'], 'load')
        self.assertEqual(len(result['sample_weights']), self.n_samples)
        
    def test_invalid_method(self):
        """Test error with invalid method."""
        with self.assertRaises(ValueError):
            weight_samples(method='invalid', trainlocs=self.locations)
            
    def test_missing_required_args(self):
        """Test errors when required arguments are missing."""
        # KD without locations
        with self.assertRaises(ValueError):
            weight_samples(method='KD', trainsamps=self.sample_ids)
            
        # Load without DataFrame
        with self.assertRaises(ValueError):
            weight_samples(method='load', trainsamps=self.sample_ids)


class TestMakeKDWeights(unittest.TestCase):
    """Test the _make_kd_weights function."""
    
    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        self.locations = np.random.randn(30, 2) * 5 + [10, 20]
        
    def test_with_bandwidth(self):
        """Test with specified bandwidth."""
        weights = _make_kd_weights(self.locations, bandwidth=2.0)
        
        self.assertEqual(len(weights), len(self.locations))
        self.assertAlmostEqual(np.sum(weights), 1.0, places=5)
        
    def test_without_bandwidth(self):
        """Test automatic bandwidth calculation."""
        weights = _make_kd_weights(
            self.locations,
            cache_bandwidth=False,  # Don't cache for testing
            n_bandwidths=10  # Small for speed
        )
        
        self.assertEqual(len(weights), len(self.locations))
        self.assertAlmostEqual(np.sum(weights), 1.0, places=5)
        
    def test_with_caching(self):
        """Test that caching is used when enabled."""
        # Clear global cache first
        optimizer = get_global_bandwidth_optimizer()
        optimizer.clear_cache()
        
        # First call
        weights1 = _make_kd_weights(self.locations, cache_bandwidth=True, n_bandwidths=10)
        
        # Check that bandwidth was cached
        self.assertEqual(len(optimizer._cache), 1)
        
        # Second call should use cache
        weights2 = _make_kd_weights(self.locations, cache_bandwidth=True, n_bandwidths=10)
        
        # Weights should be identical
        np.testing.assert_array_almost_equal(weights1, weights2)


class TestMakeHistogramWeights(unittest.TestCase):
    """Test the _make_histogram_weights function."""
    
    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        # Create clustered data for better histogram testing
        cluster1 = np.random.randn(20, 2) + [0, 0]
        cluster2 = np.random.randn(10, 2) + [5, 5]
        self.locations = np.vstack([cluster1, cluster2])
        
    def test_basic_histogram(self):
        """Test basic histogram weight calculation."""
        weights = _make_histogram_weights(self.locations, xbins=3, ybins=3)
        
        self.assertEqual(len(weights), len(self.locations))
        self.assertGreater(np.min(weights), 0)
        
        # Samples in less dense bins should have higher weights
        self.assertGreater(np.max(weights), np.min(weights))


class TestLoadSampleWeights(unittest.TestCase):
    """Test the _load_sample_weights function."""
    
    def test_basic_loading(self):
        """Test basic weight loading."""
        sample_ids = ['A', 'B', 'C', 'D']
        weights_df = pd.DataFrame({
            'sampleID': sample_ids,
            'sample_weight': [0.1, 0.2, 0.3, 0.4]
        })
        
        result_df = _load_sample_weights(weights_df, ['B', 'D', 'A'])
        
        self.assertEqual(len(result_df), 3)
        self.assertEqual(result_df.iloc[0]['sample_weight'], 0.2)  # B
        self.assertEqual(result_df.iloc[1]['sample_weight'], 0.4)  # D
        self.assertEqual(result_df.iloc[2]['sample_weight'], 0.1)  # A
        
    def test_missing_sample(self):
        """Test error when sample is missing."""
        weights_df = pd.DataFrame({
            'sampleID': ['A', 'B'],
            'sample_weight': [0.5, 0.5]
        })
        
        with self.assertRaises(ValueError):
            _load_sample_weights(weights_df, ['A', 'C'])
            
    def test_invalid_dataframe(self):
        """Test error with invalid DataFrame structure."""
        weights_df = pd.DataFrame({
            'id': ['A', 'B'],  # Wrong column name
            'weight': [0.5, 0.5]
        })
        
        with self.assertRaises(ValueError):
            _load_sample_weights(weights_df, ['A'])


class TestGlobalOptimizer(unittest.TestCase):
    """Test the global optimizer functionality."""
    
    def test_singleton(self):
        """Test that global optimizer is a singleton."""
        opt1 = get_global_bandwidth_optimizer()
        opt2 = get_global_bandwidth_optimizer()
        
        self.assertIs(opt1, opt2)
        
    def test_shared_cache(self):
        """Test that cache is shared across calls."""
        np.random.seed(42)
        locations = np.random.randn(50, 2)
        
        # Clear any existing cache
        optimizer = get_global_bandwidth_optimizer()
        optimizer.clear_cache()
        
        # Calculate bandwidth through weight function
        _make_kd_weights(locations, cache_bandwidth=True, n_bandwidths=10)
        
        # Check that global optimizer has cached result
        self.assertEqual(len(optimizer._cache), 1)


if __name__ == '__main__':
    unittest.main()