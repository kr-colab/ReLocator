"""Tests for the PCA-initialized projection + two-phase fine-tune mode."""

import numpy as np
import pytest
from sklearn.decomposition import PCA

from locator import Locator
from locator.models import PCA_LAYER_NAME
from locator.pca import compute_pca_projection, compute_pca_projection_gram


def _pca_config(basic_config, **overrides):
    """Base config for a fast, deterministic CPU training run."""
    config = {
        **basic_config,
        "min_mac": 0,
        "use_mixed_precision": False,
        "disable_gpu": True,
    }
    config.update(overrides)
    return config


def test_compute_pca_projection_matches_sklearn():
    """A Dense layer with the returned weights reproduces PCA.transform."""
    rng = np.random.default_rng(0)
    X = rng.integers(0, 3, size=(30, 80)).astype(np.float32)
    n_components = 5

    W, bias = compute_pca_projection(X, n_components)
    assert W.shape == (80, n_components)
    assert bias.shape == (n_components,)

    reference = (
        PCA(n_components=n_components)
        .fit(X.astype(np.float64))
        .transform(X.astype(np.float64))
    )
    projected = X @ W + bias
    assert np.allclose(projected, reference, atol=1e-2)


def test_train_builds_pca_projection(genotype_data, basic_config):
    """Setting pca_components inserts a pca_projection layer of that width."""
    genotypes, samples, _, _, _ = genotype_data
    loc = Locator(_pca_config(basic_config, pca_components=8, max_epochs=2))
    loc.train(genotypes=genotypes, samples=samples)

    layer = loc.model.get_layer(PCA_LAYER_NAME)
    assert layer.units == 8


def test_frozen_projection_keeps_pca_loadings(genotype_data, basic_config):
    """With pca_finetune=False the projection stays at its PCA initialization."""
    genotypes, samples, _, _, _ = genotype_data
    loc = Locator(
        _pca_config(basic_config, pca_components=8, max_epochs=3, pca_finetune=False)
    )
    loc.train(genotypes=genotypes, samples=samples)

    kernel = loc.model.get_layer(PCA_LAYER_NAME).get_weights()[0]
    train_geno = np.asarray(
        loc.filtered_genotypes[:, loc.index_set.train].T, dtype=np.float32
    )
    # Training fits PCA via the Gram-matrix path, so compare against that.
    expected, _ = compute_pca_projection_gram(train_geno, 8)
    assert np.allclose(kernel, expected, atol=1e-5)


def test_finetune_moves_projection(genotype_data, basic_config):
    """With pca_finetune=True phase 2 moves the projection off the PCA init."""
    genotypes, samples, _, _, _ = genotype_data
    loc = Locator(
        _pca_config(basic_config, pca_components=8, max_epochs=5, pca_finetune=True)
    )
    loc.train(genotypes=genotypes, samples=samples)

    kernel = loc.model.get_layer(PCA_LAYER_NAME).get_weights()[0]
    train_geno = np.asarray(
        loc.filtered_genotypes[:, loc.index_set.train].T, dtype=np.float32
    )
    pca_loadings, _ = compute_pca_projection_gram(train_geno, 8)
    assert not np.allclose(kernel, pca_loadings, atol=1e-6)


def test_pca_components_too_large_raises(genotype_data, basic_config):
    """pca_components exceeding min(n_train, n_snps) raises a clear error."""
    genotypes, samples, _, _, _ = genotype_data
    loc = Locator(_pca_config(basic_config, pca_components=999, max_epochs=1))
    with pytest.raises(ValueError, match="pca_components"):
        loc.train(genotypes=genotypes, samples=samples)


def test_no_pca_projection_when_disabled(genotype_data, basic_config):
    """Without pca_components the model has no pca_projection layer."""
    genotypes, samples, _, _, _ = genotype_data
    loc = Locator(_pca_config(basic_config, max_epochs=1))
    loc.train(genotypes=genotypes, samples=samples)

    layer_names = [layer.name for layer in loc.model.layers]
    assert PCA_LAYER_NAME not in layer_names


def test_pca_projection_is_float32_under_mixed_precision():
    """The pca_projection layer stays float32 even under a mixed_float16 policy.

    Its forward pass nearly cancels two large terms, so float16 would destroy
    the PCA scores.
    """
    from tensorflow import keras

    from locator.models import create_network

    original = keras.mixed_precision.global_policy()
    keras.mixed_precision.set_global_policy("mixed_float16")
    try:
        model = create_network(input_shape=200, pca_components=8)
        assert model.get_layer(PCA_LAYER_NAME).compute_dtype == "float32"
    finally:
        keras.mixed_precision.set_global_policy(original)


def test_pca_components_rejects_site_order(genotype_data, basic_config):
    """pca_components is incompatible with bootstrap/jacknife SNP resampling."""
    genotypes, samples, _, _, n_snps = genotype_data
    loc = Locator(_pca_config(basic_config, pca_components=8, max_epochs=1))
    with pytest.raises(ValueError, match="bootstrap"):
        loc.train(genotypes=genotypes, samples=samples, site_order=np.arange(n_snps))
