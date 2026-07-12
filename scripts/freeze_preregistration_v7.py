#!/usr/bin/env python3
"""Freeze preregistration SHA-256 for v7 extended validation experiments.

Hashes: scorer source code + hyperparameters + preregistration document + Census version.
The v7 experiments use the SAME v6 scorer (no modifications). This freeze covers the
experiment design document; experiment scripts will be written post-freeze but must
conform to the frozen design.
"""
import hashlib
import json
import subprocess
from pathlib import Path

from preflight.core.preregister import (
    DatasetSpec,
    compute_preregistration_sha,
    save_preregistration,
)
from preflight.core.runner import ALL_MODULE_SOURCES, resolve_hyperparameters

OUTPUT_DIR = Path("docs/frozen_prereg_v7")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PREREG_DOC = Path("docs/preregistration_v7_new_experiments_v2.md")
CENSUS_VERSION = "2023-12-15"

hp, weights = resolve_hyperparameters()
canonical_hp = {**hp, "weights": weights}

git_sha = subprocess.check_output(
    ["git", "rev-parse", "HEAD"], text=True
).strip()

prereg_doc_sha = hashlib.sha256(PREREG_DOC.read_bytes()).hexdigest()

EXPERIMENTS = [
    DatasetSpec(
        name="exp4_metric_comparison",
        source_description="Same 10 pairs from Exp 0 composite validation",
        target_description="Same 10 pairs — target side",
        n_source=2000,
        n_target=2000,
        extra={
            "census_version": CENSUS_VERSION,
            "embedding": "geneformer",
            "competing_metrics": ["rankme", "mmd", "c2st"],
            "n_pairs": 10,
            "primary_hypothesis": "H4.1_false_certification_rate",
            "false_cert_threshold": {"tier_ge": 5, "degradation_gt": 0.30},
            "reuses_exp0_embeddings": True,
        },
    ),
    DatasetSpec(
        name="exp4b_degeneracy_check",
        source_description="Re-analysis of Exp 1 + Exp 2 M4 values (no new data pull)",
        target_description="N/A (counterfactual simulation)",
        n_source=None,
        n_target=None,
        extra={
            "census_version": CENSUS_VERSION,
            "part_a": "confirmatory_reanalysis_H4b1_H4b3",
            "part_b": "prospective_counterfactual_H4b2",
            "gate_rule": "any_module_tier_le_1_caps_overall_to_le_3",
            "data_source": [
                "results/modal_results/bag_of_genes_v6/bag_of_genes_baseline/summary_20260711_122910.json",
                "results/modal_results/sweep_v6/sweep/summary_20260711_221653.json",
            ],
        },
    ),
    DatasetSpec(
        name="exp5_cross_tissue",
        source_description="6 cross-tissue pairs (lung/blood/brain/kidney/liver), Geneformer, 10x 3'v3",
        target_description="Different tissue, same assay (10x 3'v3)",
        n_source=2000,
        n_target=2000,
        extra={
            "census_version": CENSUS_VERSION,
            "embedding": "geneformer",
            "assay": "EFO:0009922",
            "pairs": [
                ["lung", "brain"],
                ["lung", "kidney"],
                ["blood", "lung"],
                ["blood", "brain"],
                ["liver", "kidney"],
                ["blood", "liver"],
            ],
            "negative_controls": ["lung", "blood", "brain"],
            "blood_ontology": "UBERON:0000178",
        },
    ),
    DatasetSpec(
        name="exp6_cross_disease_exploratory",
        source_description="4 tissue x disease pairs, Geneformer + 10x 3'v3, healthy source",
        target_description="Same tissue, diseased target",
        n_source=2000,
        n_target=2000,
        extra={
            "census_version": CENSUS_VERSION,
            "embedding": "geneformer",
            "status": "EXPLORATORY",
            "pairs": [
                {"tissue": "lung", "disease": "pulmonary fibrosis"},
                {"tissue": "lung", "disease": "lung adenocarcinoma"},
                {"tissue": "brain", "disease": "Alzheimer disease"},
                {"tissue": "liver", "disease": "hepatocellular carcinoma"},
            ],
            "negative_controls": 2,
            "min_cells_per_condition": 500,
        },
    ),
    DatasetSpec(
        name="exp7_scgpt",
        source_description="4 tissues x 4 embeddings (Geneformer, scVI, BoG-PCA-512, scGPT), cross-assay",
        target_description="Same tissues, target assay (10x 5'v2)",
        n_source=2000,
        n_target=2000,
        extra={
            "census_version": CENSUS_VERSION,
            "embeddings": ["geneformer", "scvi", "bag_of_genes_pca512", "scgpt"],
            "tissues": ["lung", "liver", "kidney", "brain"],
            "source_assay": "EFO:0009922",
            "target_assay": "EFO:0009900",
            "contingency": "drop_tissue_if_scgpt_unavailable",
        },
    ),
]

print(f"Git commit: {git_sha}")
print(f"Prereg doc: {PREREG_DOC} (SHA: {prereg_doc_sha[:16]}...)")
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
    "preregistration_version": "v7",
    "preregistration_doc": str(PREREG_DOC),
    "preregistration_doc_sha256": prereg_doc_sha,
    "scorer_version": "v6 (unchanged)",
    "scorer_modules": [m.__name__ for m in ALL_MODULE_SOURCES],
    "hyperparameters": canonical_hp,
    "census_version": CENSUS_VERSION,
    "experiment_shas": all_records,
}
summary_path = OUTPUT_DIR / "FROZEN_SUMMARY.json"
with open(summary_path, "w") as f:
    json.dump(summary, f, indent=2)

print(f"\nFrozen summary: {summary_path}")
print(f"All {len(EXPERIMENTS)} experiment SHAs saved to {OUTPUT_DIR}/")
print(f"\nPreregistration document SHA-256: {prereg_doc_sha}")
