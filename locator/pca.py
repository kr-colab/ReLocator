"""PCA loadings for the optional PCA-initialized projection layer."""

import numpy as np
from sklearn.decomposition import PCA


def compute_pca_projection(genotype_matrix, n_components):
    """Fit PCA and return weights for a linear layer that maps raw genotype
    counts to PCA scores.

    The returned ``(W, bias)`` satisfy ``X @ W + bias == (X - mean) @ W`` for
    raw genotype counts ``X``, so a ``Dense`` layer initialized with them
    reproduces ``PCA.transform(X)`` exactly without pre-centering the input.

    Parameters
    ----------
    genotype_matrix : np.ndarray
        Genotype matrix of shape ``(n_samples, n_snps)``. Pass training
        samples only to avoid leaking held-out samples into the subspace.
    n_components : int
        Number of PCA components (the projection width).

    Returns
    -------
    W : np.ndarray
        Loadings of shape ``(n_snps, n_components)``, float32.
    bias : np.ndarray
        Bias of shape ``(n_components,)``, float32, equal to ``-(mean @ W)``.
    """
    X = np.asarray(genotype_matrix, dtype=np.float32)
    pca = PCA(n_components=n_components)
    pca.fit(X)
    W = pca.components_.T
    bias = -(pca.mean_ @ W)
    return W.astype(np.float32), bias.astype(np.float32)
