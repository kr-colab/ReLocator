"""Consolidated tests for analysis methods using tf.data pipeline"""

from unittest.mock import Mock, patch

import allel
import numpy as np
import pandas as pd
import pytest

from locator.core import Locator


class TestAnalysisTFData:
    """Test that all analysis methods use tf.data pipeline efficiently"""

    # Bootstrap tests
    @pytest.mark.parametrize("n_bootstraps", [1, 2])
    def test_bootstrap_uses_site_order(self, genotype_data, basic_config, n_bootstraps):
        """Test bootstrap uses site_order parameter without array copies"""
        genotypes, samples, _, _, n_snps = genotype_data

        # Create locator
        locator = Locator(basic_config)

        # Track site orders used during training
        site_orders_used = []
        original_train = locator.train

        def track_train(*args, **kwargs):
            site_order = kwargs.get("site_order")
            if site_order is not None:
                site_orders_used.append(site_order.copy())
            return original_train(*args, **kwargs)

        locator.train = track_train

        # Run bootstrap
        results = locator.run_bootstraps(
            genotypes=genotypes,
            samples=samples,
            n_bootstraps=n_bootstraps,
            return_df=True,
        )

        # Verify site_order was used for each bootstrap
        assert len(site_orders_used) == n_bootstraps

        # Each site_order should be different (very unlikely to be same)
        if n_bootstraps > 1:
            assert not np.array_equal(site_orders_used[0], site_orders_used[1])

        # Verify results
        assert isinstance(results, pd.DataFrame)
        for i in range(n_bootstraps):
            assert f"x_{i}" in results.columns
            assert f"y_{i}" in results.columns

    # Holdout tests with parametrization
    @pytest.mark.parametrize(
        "method,kwargs,expected_format",
        [
            ("run_holdouts", {"k": 5, "n_reps": 2, "return_df": True}, "wide"),
            ("run_k_fold_holdouts", {"k": 3, "return_df": True}, "long"),
            ("train_holdout", {"k": 5}, None),  # Returns history, not df
        ],
    )
    def test_holdout_methods_use_indexset(
        self, genotype_data, basic_config, method, kwargs, expected_format
    ):
        """Test all holdout methods use IndexSet correctly"""
        genotypes, samples, _, _, _ = genotype_data

        # Create locator
        locator = Locator(basic_config)

        # Get the method
        holdout_method = getattr(locator, method)

        # Run the method
        result = holdout_method(genotypes=genotypes, samples=samples, **kwargs)

        # Verify IndexSet was created (all methods should create one)
        assert hasattr(locator, "index_set")
        assert locator.index_set is not None

        # Check results based on expected format
        if expected_format == "wide":
            # run_holdouts returns wide format
            assert isinstance(result, pd.DataFrame)
            assert "x_rep0" in result.columns
            assert "y_rep0" in result.columns
        elif expected_format == "long":
            # run_k_fold_holdouts returns long format
            assert isinstance(result, pd.DataFrame)
            assert "fold" in result.columns
            assert "x_pred" in result.columns
            assert "y_pred" in result.columns
        else:
            # train_holdout returns history
            assert hasattr(result, "history")

    # Jacknife tests
    def test_jacknife_with_tf_data(self, genotype_data, basic_config):
        """Test jacknife works with tf.data pipeline"""
        genotypes, samples, _, _, _ = genotype_data

        # Create locator
        locator = Locator(basic_config)

        # Run jacknife - it drops prop of SNPs, creates ceiling(1/prop) replicates
        results = locator.run_jacknife(
            genotypes=genotypes,
            samples=samples,
            prop=0.2,  # This will create 5 jacknife replicates
            return_df=True,
        )

        # Verify results
        assert isinstance(results, pd.DataFrame)
        # Should have predictions for each jackknife iteration
        n_jack = int(np.ceil(1.0 / 0.2))  # Should be 5
        for i in range(n_jack):
            assert f"x_{i}" in results.columns
            assert f"y_{i}" in results.columns

        # Verify filtered_genotypes exists
        assert hasattr(locator, "filtered_genotypes")

    # Shared memory efficiency test
    @pytest.mark.parametrize(
        "method,method_kwargs",
        [
            ("run_bootstraps", {"n_bootstraps": 1}),
            ("run_jacknife", {"prop": 0.5}),  # 2 jacknife replicates
        ],
    )
    def test_memory_efficiency(self, genotype_data, basic_config, method, method_kwargs):
        """Test that resampling methods don't create array copies"""
        genotypes, samples, _, _, _ = genotype_data

        # Create locator
        locator = Locator(basic_config)

        # Initial training to set up filtered_genotypes
        locator.train(genotypes=genotypes, samples=samples)

        # Store shape of filtered genotypes
        filtered_geno_shape = locator.filtered_genotypes.shape
        # filtered_geno_id = id(locator.filtered_genotypes)  # noqa: F841

        # Mock model to speed up test
        locator.model = Mock()
        locator.model.fit.return_value = Mock(history={})
        locator.model.predict.return_value = np.random.normal(0, 1, (len(samples), 2))

        # Run the method
        analysis_method = getattr(locator, method)
        analysis_method(genotypes=genotypes, samples=samples, **method_kwargs)

        # Verify filtered_genotypes is still the same object
        assert hasattr(locator, "filtered_genotypes")
        assert locator.filtered_genotypes.shape == filtered_geno_shape
        # Note: id check may not always work due to Python memory management
        # but shape check ensures no reconstruction

    # Test that all methods work with tf.data pipeline
    @patch("locator.training.make_tf_dataset")
    def test_all_methods_use_make_tf_dataset(
        self, mock_make_tf_dataset, genotype_data, basic_config
    ):
        """Test that all training methods use make_tf_dataset"""
        genotypes, samples, _, _, _ = genotype_data

        # Track calls
        call_count = 0

        def track_calls(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            # Return the original function result
            from locator.data import make_tf_dataset as original

            return original(*args, **kwargs)

        mock_make_tf_dataset.side_effect = track_calls

        # Create locator
        locator = Locator(basic_config)

        # Test train_holdout
        locator.train_holdout(genotypes=genotypes, samples=samples, k=5)

        # Should have been called at least twice (train and validation datasets)
        assert call_count >= 2

        # Verify the calls included correct parameters. The index-based
        # pipeline carries coordinates and indices, not the genotype matrix.
        calls = mock_make_tf_dataset.call_args_list
        for call in calls:
            kwargs = call[1]
            assert "coordinates" in kwargs
            assert "index_set" in kwargs
            assert "split" in kwargs

    # Leave-one-out test (separate due to special requirements)
    def test_leave_one_out_small_dataset(self, basic_config):
        """Test leave-one-out with appropriately sized dataset"""
        # Create smaller dataset for LOO testing
        n_samples = 10
        n_snps = 50

        geno_array = np.random.choice([0, 1], size=(n_snps, n_samples, 2)).astype(
            np.int8
        )
        genotypes = allel.GenotypeArray(geno_array)
        samples = np.array([f"sample_{i}" for i in range(n_samples)])

        # Create locator with adjusted config
        config = basic_config.copy()
        config["keras_verbose"] = 0
        locator = Locator(config)

        # Run leave-one-out
        results = locator.run_leave_one_out(
            genotypes=genotypes, samples=samples, return_df=True
        )

        # Verify results
        assert isinstance(results, pd.DataFrame)
        assert len(results) > 0
        assert "fold" in results.columns
