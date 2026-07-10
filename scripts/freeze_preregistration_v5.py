#!/usr/bin/env python3
"""Freeze preregistration SHA-256 for all 6 experiments in preregistration_v5.

Computes one SHA per experiment (different dataset specs, same scorer code).
Saves JSON records to docs/frozen_prereg_v5/.
"""
import json
import subprocess
from pathlib import Path

from preflight.core.preregister import (
    DatasetSpec,
    compute_preregistration_sha,
    save_preregistration,
)
from preflight.core.runner import ALL_MODULE_SOURCES, resolve_hyperparameters

OUTPUT_DIR = Path("docs/frozen_prereg_v5")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

hp, weights = resolve_hyperparameters()
canonical_hp = {**hp, "weights": weights}

git_sha = subprocess.check_output(
    ["git", "rev-parse", "HEAD"], text=True
).strip()

CENSUS_VERSION = "2023-12-15"

EXPERIMENTS = [
    DatasetSpec(
        name="exp0_composite_validation",
        source_description="~25 source->target pairs across 4 pair types (cross-assay, cross-tissue, cross-dataset, negative control)",
        target_description="Same pairs — target side",
        n_source=2000,
        n_target=2000,
        extra={
            "census_version": CENSUS_VERSION,
            "embedding": "geneformer",
            "tissues": ["lung", "heart", "liver", "kidney", "brain"],
            "pair_types": ["cross_assay", "cross_tissue", "cross_dataset", "negative_control"],
            "n_pairs_approx": 25,
            "min_shared_cell_types": 3,
            "primary_endpoint": "partial_spearman_rho_composite_vs_relative_degradation",
        },
    ),
    DatasetSpec(
        name="exp1_bag_of_genes_baseline",
        source_description="Lung 10x 3' v3 (geneformer + scvi + bag-of-genes PCA 512d)",
        target_description="Lung 10x 5' v2 (geneformer + scvi + bag-of-genes PCA 512d)",
        n_source=2000,
        n_target=2000,
        extra={
            "census_version": CENSUS_VERSION,
            "embeddings": ["geneformer", "scvi", "bag_of_genes_pca512"],
            "tissue": "lung",
        },
    ),
    DatasetSpec(
        name="exp2_multi_model_sweep",
        source_description="5 tissues x 3 models, cross-assay shift (10x 3'v3 -> 10x 5'v2)",
        target_description="Same tissues, target assay",
        n_source=2000,
        n_target=2000,
        extra={
            "census_version": CENSUS_VERSION,
            "embeddings": ["geneformer", "scvi", "bag_of_genes_pca512"],
            "tissues": ["lung", "heart", "liver", "kidney", "brain"],
        },
    ),
    DatasetSpec(
        name="exp3_sample_size_sensitivity",
        source_description="Lung 10x 3' v3 (geneformer), varying cell counts",
        target_description="Lung 10x 5' v2 (geneformer), varying cell counts",
        n_source=None,
        n_target=None,
        extra={
            "census_version": CENSUS_VERSION,
            "embedding": "geneformer",
            "cell_counts": [200, 500, 1000, 2000, 5000, 10000],
            "replicates_per_count": 3,
            "tissue": "lung",
        },
    ),
    DatasetSpec(
        name="exp4_negative_control_full",
        source_description="Lung 10x 3' v3, random split A (seed=42)",
        target_description="Lung 10x 3' v3, random split B (seed=99)",
        n_source=2000,
        n_target=2000,
        extra={
            "census_version": CENSUS_VERSION,
            "embeddings": ["geneformer", "scvi", "bag_of_genes_pca512"],
            "tissue": "lung",
            "seed_source": 42,
            "seed_target": 99,
        },
    ),
    DatasetSpec(
        name="exp5_gene_level_probes",
        source_description="Geneformer gene token embeddings (from model weights)",
        target_description="N/A (gene-level, not cell-level)",
        n_source=None,
        n_target=None,
        extra={
            "model": "geneformer",
            "regulatory_source": "OmniPath (DoRothEA + CollecTRI)",
            "probes": ["tf_target", "hub_tf", "rsa"],
        },
    ),
]

print(f"Git commit: {git_sha}")
print(f"Scorer modules: {[m.__name__ for m in ALL_MODULE_SOURCES]}")
print(f"Hyperparameters: {json.dumps(canonical_hp, indent=2)}")
print()

all_records = {}
for ds in EXPERIMENTS:
    record = compute_preregistration_sha(ALL_MODULE_SOURCES, canonical_hp, ds)
    path = OUTPUT_DIR / f"{ds.name}.json"
    save_preregistration(record, path)
    all_records[ds.name] = record.sha256
    print(f"  {ds.name}: {record.sha256[:16]}...")

summary = {
    "git_commit": git_sha,
    "preregistration_version": "v4",
    "preregistration_doc": "docs/preregistration_v5.md",
    "scorer_modules": [m.__name__ for m in ALL_MODULE_SOURCES],
    "hyperparameters": canonical_hp,
    "experiment_shas": all_records,
}
summary_path = OUTPUT_DIR / "FROZEN_SUMMARY.json"
with open(summary_path, "w") as f:
    json.dump(summary, f, indent=2)

print(f"\nFrozen summary: {summary_path}")
print(f"All {len(EXPERIMENTS)} experiment SHAs saved to {OUTPUT_DIR}/")
