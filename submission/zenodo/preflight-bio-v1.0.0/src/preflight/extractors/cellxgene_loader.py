"""CELLxGENE Census loader — pull real biology data into the Preflight pipeline.

Queries the CELLxGENE Census for a tissue/cell-type slice, optionally with
hosted scFM embeddings (Geneformer, scVI, UCE, scGPT), and returns arrays
matching run()'s contract.

Requires the `cellxgene-census` package (not in the core install):
    pip install cellxgene-census

Usage:
    from preflight.extractors.cellxgene_loader import query_census

    X, labels, metadata = query_census(
        census_version="2023-12-15",
        organism="homo_sapiens",
        obs_value_filter="tissue_ontology_term_id == 'UBERON:0002048'",
        embedding="geneformer",
        obs_label_key="cell_type",
    )
"""
from typing import Any

import numpy as np

from preflight.extractors.anndata_loader import extract_arrays


def _import_census():
    try:
        import cellxgene_census
        import cellxgene_census.experimental
    except ImportError:
        raise ImportError(
            "cellxgene-census is required for CELLxGENE queries. "
            "Install it with: pip install cellxgene-census"
        )
    return cellxgene_census


DEFAULT_OBS_COLUMNS = [
    "cell_type",
    "tissue",
    "assay",
    "disease",
    "dataset_id",
    "is_primary_data",
    "donor_id",
]


def query_census(
    census_version: str,
    organism: str = "homo_sapiens",
    obs_value_filter: str | None = None,
    embedding: str | None = None,
    obs_label_key: str = "cell_type",
    obs_batch_key: str | None = None,
    obs_columns: list[str] | None = None,
    max_cells: int | None = None,
) -> tuple[np.ndarray, np.ndarray | None, dict[str, Any]]:
    """Query CELLxGENE Census and return arrays for run().

    Args:
        census_version: Pinned Census version (e.g. "2023-12-15"). Never
            use "latest" for preregistered runs.
        organism: "homo_sapiens" or "mus_musculus".
        obs_value_filter: SOMA filter string, e.g.
            "tissue_ontology_term_id == 'UBERON:0002048' and is_primary_data == True".
        embedding: Name of hosted embedding to load (e.g. "geneformer",
            "scvi", "scgpt", "uce"). If None, uses raw counts from adata.X.
        obs_label_key: Column in obs for cell-type labels.
        obs_batch_key: Column in obs for batch/site metadata.
        obs_columns: Columns to fetch in obs. Defaults to a standard set.
        max_cells: If set, randomly subsample to this many cells.

    Returns:
        (X, labels, metadata) matching run()'s signature.
    """
    cellxgene_census = _import_census()

    if obs_columns is None:
        obs_columns = list(DEFAULT_OBS_COLUMNS)

    for key in [obs_label_key, obs_batch_key]:
        if key is not None and key not in obs_columns:
            obs_columns.append(key)

    obs_embeddings = [embedding] if embedding else []

    with cellxgene_census.open_soma(census_version=census_version) as census:
        adata = cellxgene_census.get_anndata(
            census,
            organism=organism,
            obs_value_filter=obs_value_filter,
            obs_column_names=obs_columns,
            obs_embeddings=obs_embeddings,
        )

    if max_cells is not None and adata.n_obs > max_cells:
        rng = np.random.default_rng(42)
        idx = rng.choice(adata.n_obs, size=max_cells, replace=False)
        idx.sort()
        adata = adata[idx].copy()

    obsm_key = embedding if embedding else None
    X, labels, meta = extract_arrays(
        adata,
        obsm_key=obsm_key,
        obs_label_key=obs_label_key,
        obs_batch_key=obs_batch_key,
    )

    meta["census_version"] = census_version
    meta["organism"] = organism
    meta["obs_value_filter"] = obs_value_filter
    meta["embedding"] = embedding

    return X, labels, meta


def build_dataset_spec(
    meta: dict[str, Any],
    name: str,
    description: str,
) -> "DatasetSpec":
    """Build a DatasetSpec from Census query metadata for preregistration.

    Args:
        meta: Metadata dict returned by query_census.
        name: Short name for the dataset.
        description: Description of this data slice.

    Returns:
        DatasetSpec with Census version and query baked in.
    """
    from preflight.core.preregister import DatasetSpec

    return DatasetSpec(
        name=name,
        source_description=description,
        target_description=description,
        n_source=meta.get("n_cells"),
        n_target=meta.get("n_cells"),
        d=meta.get("n_features"),
        extra={
            "census_version": meta.get("census_version"),
            "organism": meta.get("organism"),
            "obs_value_filter": meta.get("obs_value_filter"),
            "embedding": meta.get("embedding"),
        },
    )


def list_available_embeddings(census_version: str) -> list[dict[str, Any]]:
    """List all hosted embeddings for a Census version."""
    cellxgene_census = _import_census()
    return cellxgene_census.experimental.get_all_available_embeddings(census_version)
