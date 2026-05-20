"""PCA loadings for the optional PCA-initialized projection layer."""

import numpy as np
import tensorflow as tf
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


def compute_pca_projection_gram(genotype_matrix, n_components):
    """Fit PCA via the Gram-matrix eigendecomposition; a fast path for the
    PCA-initialized projection layer.

    Returns the same ``(W, bias)`` contract as :func:`compute_pca_projection`,
    but built entirely from TensorFlow ops, so it runs on the GPU when the
    input tensor is GPU-resident. With far more SNPs than samples the Gram
    matrix ``Xc @ Xcᵀ`` is only ``(n_samples, n_samples)``, so the fit reduces
    to one small eigendecomposition plus two matmuls -- orders of magnitude
    cheaper than an SVD over the full ``(n_samples, n_snps)`` matrix.

    The Gram matrix's eigenvectors are the left singular vectors of ``Xc`` and
    its eigenvalues the squared singular values, so the loadings are recovered
    as ``V = Xcᵀ @ U / singular_value``.

    Parameters
    ----------
    genotype_matrix : tf.Tensor or np.ndarray
        Genotype matrix of shape ``(n_samples, n_snps)``, any numeric dtype.
        Pass training samples only to avoid leaking held-out samples.
    n_components : int
        Number of PCA components (the projection width).

    Returns
    -------
    W : tf.Tensor
        Loadings of shape ``(n_snps, n_components)``, float32. Returned as a
        device tensor so the caller can assign it to the projection layer
        without a host round trip.
    bias : tf.Tensor
        Bias of shape ``(n_components,)``, float32, equal to ``-(mean @ W)``.
    """
    X = tf.cast(genotype_matrix, tf.float32)
    mean = tf.reduce_mean(X, axis=0, keepdims=True)
    Xc = X - mean
    gram = tf.linalg.matmul(Xc, Xc, transpose_b=True)
    evals, evecs = tf.linalg.eigh(gram)  # ascending eigenvalues
    evals = evals[::-1][:n_components]
    evecs = evecs[:, ::-1][:, :n_components]
    # Guard against tiny negative eigenvalues from float round-off.
    scale = tf.sqrt(tf.maximum(evals, 1e-12))
    W = tf.linalg.matmul(Xc, evecs, transpose_a=True) / scale
    bias = tf.reshape(-tf.linalg.matmul(mean, W), [-1])
    return W, bias


def scree_elbow(genotype_matrix):
    """Pick a PCA rank from the genotype-PCA scree elbow.

    Computes the explained-variance spectrum (Gram-matrix eigenvalues over the
    samples) and returns the rank at the chord-distance elbow: the point of the
    explained-variance curve farthest below the straight line joining its first
    and last points. For genotype data the spectrum drops steeply then flattens,
    so this is a stable, data-driven projection width.

    Parameters
    ----------
    genotype_matrix : tf.Tensor or np.ndarray
        Genotype matrix of shape ``(n_samples, n_snps)``; training samples only.

    Returns
    -------
    int
        The elbow rank, at least 1.
    """
    X = tf.cast(genotype_matrix, tf.float32)
    mean = tf.reduce_mean(X, axis=0, keepdims=True)
    Xc = X - mean
    gram = tf.linalg.matmul(Xc, Xc, transpose_b=True)
    evals = np.clip(tf.linalg.eigvalsh(gram).numpy()[::-1], 0.0, None)
    total = evals.sum()
    if total <= 0.0:
        return 1
    # Keep the meaningful components; a centred n-sample matrix has rank n-1.
    evr = evals[evals > total * 1e-9] / total
    if len(evr) < 3 or evr[0] == evr[-1]:
        return max(1, len(evr))
    x = np.arange(len(evr), dtype=np.float64) / (len(evr) - 1)
    y = (evr - evr.min()) / (evr.max() - evr.min())
    chord = y[0] + x * (y[-1] - y[0])
    return int(np.argmax(chord - y)) + 1
