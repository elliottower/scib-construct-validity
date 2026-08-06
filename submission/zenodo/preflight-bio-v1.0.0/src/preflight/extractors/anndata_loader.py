"""Load AnnData (.h5ad) files into arrays for the Preflight pipeline.

Bridges the gap between AnnData objects (the standard single-cell container)
and run()'s (n, d) array contract.

Usage:
    from preflight.extractors.anndata_loader import load_h5ad, extract_arrays

    # From file
    X, labels, metadata = load_h5ad("data.h5ad", obsm_key="X_geneformer")

    # From an existing AnnData
    X, labels, metadata = extract_arrays(adata, obsm_key="X_pca")
"""
import warnings
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
from scipy import sparse


def _to_dense(X: Any) -> np.ndarray:
    if sparse.issparse(X):
        return np.asarray(X.todense())
    return np.asarray(X, dtype=np.float64)


def extract_arrays(
    adata: ad.AnnData,
    obsm_key: str | None = None,
    obs_label_key: str | None = None,
    obs_batch_key: str | None = None,
    n_hvg: int | None = None,
) -> tuple[np.ndarray, np.ndarray | None, dict[str, Any]]:
    """Extract embedding matrix, labels, and metadata from an AnnData object.

    Args:
        adata: An AnnData object.
        obsm_key: Key in adata.obsm for pre-computed embeddings (e.g.
            "X_pca", "X_geneformer", "X_scgpt"). If None, uses adata.X.
        obs_label_key: Column in adata.obs for cell-type or class labels.
        obs_batch_key: Column in adata.obs for batch/site labels (used for
            Module 7 ecological bias metadata).
        n_hvg: If set and obsm_key is None, subset to the top n_hvg highly
            variable genes (requires adata.var["highly_variable"]).

    Returns:
        (X, labels, metadata) matching run()'s signature.
        X: (n_cells, d) float64 array.
        labels: integer label array or None.
        metadata: dict with optional keys for run().
    """
    if obsm_key is not None:
        if obsm_key not in adata.obsm:
            available = list(adata.obsm.keys())
            raise KeyError(
                f"obsm_key={obsm_key!r} not found. Available: {available}"
            )
        X = _to_dense(adata.obsm[obsm_key])
    else:
        if adata.X is None:
            raise ValueError("adata.X is None and no obsm_key was specified")
        if n_hvg is not None and "highly_variable" in adata.var.columns:
            hvg_mask = adata.var["highly_variable"].values
            n_available = int(hvg_mask.sum())
            if n_available < n_hvg:
                warnings.warn(
                    f"Requested {n_hvg} HVGs but only {n_available} available"
                )
            X = _to_dense(adata[:, hvg_mask].X)
        else:
            X = _to_dense(adata.X)

    labels = None
    if obs_label_key is not None:
        if obs_label_key not in adata.obs.columns:
            available = list(adata.obs.columns)
            raise KeyError(
                f"obs_label_key={obs_label_key!r} not found. Available: {available}"
            )
        raw_series = adata.obs[obs_label_key]
        if hasattr(raw_series, "cat"):
            labels = raw_series.cat.codes.values.copy()
        else:
            try:
                is_int = np.issubdtype(raw_series.dtype, np.integer)
            except TypeError:
                is_int = False
            if is_int:
                labels = np.asarray(raw_series.values)
            else:
                raw_labels = list(raw_series)
                unique_vals = sorted(set(raw_labels))
                label_map = {v: i for i, v in enumerate(unique_vals)}
                labels = np.array([label_map[v] for v in raw_labels])

    metadata: dict[str, Any] = {}
    if obs_batch_key is not None:
        if obs_batch_key not in adata.obs.columns:
            available = list(adata.obs.columns)
            raise KeyError(
                f"obs_batch_key={obs_batch_key!r} not found. Available: {available}"
            )
        metadata["batch"] = np.asarray(adata.obs[obs_batch_key].values)

    metadata["n_cells"] = adata.n_obs
    metadata["n_features"] = X.shape[1]
    if hasattr(adata, "obs") and len(adata.obs.columns) > 0:
        metadata["obs_columns"] = list(adata.obs.columns)

    return X, labels, metadata


def load_h5ad(
    path: str | Path,
    obsm_key: str | None = None,
    obs_label_key: str | None = None,
    obs_batch_key: str | None = None,
    n_hvg: int | None = None,
    backed: bool = False,
) -> tuple[np.ndarray, np.ndarray | None, dict[str, Any]]:
    """Load an .h5ad file and extract arrays for run().

    Args:
        path: Path to the .h5ad file.
        obsm_key: Key in obsm for embeddings. None uses adata.X.
        obs_label_key: Column in obs for labels.
        obs_batch_key: Column in obs for batch/site.
        n_hvg: Subset to top HVGs if using adata.X.
        backed: If True, load in backed mode (memory-efficient for large files).

    Returns:
        (X, labels, metadata) matching run()'s signature.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if not path.suffix == ".h5ad":
        raise ValueError(f"Expected .h5ad file, got: {path.suffix}")

    if backed:
        adata = ad.read_h5ad(path, backed="r")
    else:
        adata = ad.read_h5ad(path)

    return extract_arrays(
        adata,
        obsm_key=obsm_key,
        obs_label_key=obs_label_key,
        obs_batch_key=obs_batch_key,
        n_hvg=n_hvg,
    )


def anndata_from_synthetic(
    counts: np.ndarray,
    labels: np.ndarray,
    gene_names: list[str] | None = None,
    cell_type_key: str = "cell_type",
) -> ad.AnnData:
    """Build an AnnData from synthetic count matrices (for testing).

    Args:
        counts: (n_cells, n_genes) array.
        labels: integer cell-type labels.
        gene_names: optional gene name list (auto-generated if None).
        cell_type_key: column name for labels in obs.

    Returns:
        AnnData with .X, .obs[cell_type_key], and .var index.
    """
    n_cells, n_genes = counts.shape
    if gene_names is None:
        gene_names = [f"gene_{i}" for i in range(n_genes)]

    import pandas as pd

    obs = pd.DataFrame(
        {cell_type_key: [f"type_{l}" for l in labels]},
        index=[f"cell_{i}" for i in range(n_cells)],
    )
    obs[cell_type_key] = obs[cell_type_key].astype("category")

    var = pd.DataFrame(index=gene_names)

    adata = ad.AnnData(
        X=counts.astype(np.float32),
        obs=obs,
        var=var,
    )
    return adata
