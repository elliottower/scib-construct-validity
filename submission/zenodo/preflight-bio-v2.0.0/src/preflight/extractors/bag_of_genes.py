"""Bag-of-genes embedding extractor: log1p normalization + PCA on raw counts.

A simple, model-free baseline for single-cell embeddings. Takes raw expression
counts from an AnnData object, applies log1p normalization, and reduces
dimensionality via PCA.

Usage:
    from preflight.extractors.bag_of_genes import extract_bag_of_genes

    # 512-dimensional PCA embedding
    X_pca = extract_bag_of_genes(adata, n_components=512)

    # Lower-dimensional for comparison with scVI (50d)
    X_pca_50 = extract_bag_of_genes(adata, n_components=50)
"""
from typing import Any

import anndata as ad
import numpy as np
from scipy import sparse
from sklearn.decomposition import PCA


def extract_bag_of_genes(
    adata: ad.AnnData,
    n_components: int = 512,
    seed: int = 0,
) -> np.ndarray:
    """Extract bag-of-genes embedding from raw expression.

    Pipeline: raw counts -> log1p normalization -> PCA.

    Args:
        adata: AnnData with .X containing raw counts (sparse or dense).
        n_components: Number of PCA dimensions.
        seed: Random seed for reproducible PCA.

    Returns:
        (n_cells, n_components) float64 array. n_components may be smaller
        than requested if the data has fewer cells or genes.

    Raises:
        ValueError: If adata.X is None.
    """
    if adata.X is None:
        raise ValueError("adata.X is None -- no expression data to extract")

    X = adata.X
    if sparse.issparse(X):
        X = np.asarray(X.todense())
    X = np.asarray(X, dtype=np.float64)
    X = np.log1p(X)

    n_components_actual = min(n_components, X.shape[0], X.shape[1])
    pca = PCA(n_components=n_components_actual, random_state=seed)
    return pca.fit_transform(X)


def extract_bag_of_genes_with_variance(
    adata: ad.AnnData,
    n_components: int = 512,
    seed: int = 0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Extract bag-of-genes embedding and return variance diagnostics.

    Same pipeline as extract_bag_of_genes, but also returns the PCA
    explained variance ratios for quality assessment.

    Args:
        adata: AnnData with .X containing raw counts.
        n_components: Number of PCA dimensions.
        seed: Random seed for reproducible PCA.

    Returns:
        (X_pca, info) where X_pca is (n_cells, n_components) and info
        contains explained_variance_ratio, cumulative_variance, and
        n_components_actual.
    """
    if adata.X is None:
        raise ValueError("adata.X is None -- no expression data to extract")

    X = adata.X
    if sparse.issparse(X):
        X = np.asarray(X.todense())
    X = np.asarray(X, dtype=np.float64)
    X = np.log1p(X)

    n_components_actual = min(n_components, X.shape[0], X.shape[1])
    pca = PCA(n_components=n_components_actual, random_state=seed)
    X_pca = pca.fit_transform(X)

    cumulative = np.cumsum(pca.explained_variance_ratio_)
    info = {
        "n_components_actual": n_components_actual,
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "cumulative_variance": cumulative.tolist(),
        "total_variance_explained": float(cumulative[-1]),
        "n_cells": X.shape[0],
        "n_genes": X.shape[1],
    }
    return X_pca, info
