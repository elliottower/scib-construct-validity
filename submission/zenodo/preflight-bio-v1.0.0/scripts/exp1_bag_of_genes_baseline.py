"""Experiment 1: Bag-of-genes baseline comparison.

Compares Geneformer (512d), scVI (50d), and bag-of-genes PCA (512d, 50d)
embeddings on the lung cross-assay shift task (10x 3'v3 -> 10x 5'v2).
Tests whether scFM embeddings outperform a simple PCA baseline on
transportability metrics and cell-type classification.

Preregistered hypotheses (from preregistration_v4):
  H1.1: Geneformer cell-type probe macro F1 > bag-of-genes F1
  H1.2: Bag-of-genes M2 domain shift AUC < Geneformer M2 AUC
  H1.3: Bag-of-genes M1 Grassmannian distance <= Geneformer M1 distance

Usage (from the .venv-census environment):
    .venv-census/bin/python scripts/exp1_bag_of_genes_baseline.py
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import anndata as ad
import networkx as nx
import numpy as np
from scipy import sparse as sp
from scipy.spatial.distance import cdist
from tqdm import tqdm

# ======================================================================
# Constants
# ======================================================================

CENSUS_VERSION = "2023-12-15"
ORGANISM = "homo_sapiens"
MAX_CELLS = 2000

SOURCE_FILTER = (
    "tissue_ontology_term_id == 'UBERON:0002048' "
    "and is_primary_data == True "
    "and assay_ontology_term_id == 'EFO:0009922'"
)
TARGET_FILTER = (
    "tissue_ontology_term_id == 'UBERON:0002048' "
    "and is_primary_data == True "
    "and assay_ontology_term_id == 'EFO:0011025'"
)

OBS_COLUMNS = [
    "cell_type", "tissue", "assay", "disease",
    "dataset_id", "donor_id", "is_primary_data",
]

# Hosted embeddings to pull from Census
HOSTED_EMBEDDINGS = ["geneformer", "scvi"]

FROZEN_PREREG_PATH = Path("docs/frozen_prereg_v6/exp1_bag_of_genes_baseline.json")
OUTPUT_DIR = Path("results/bag_of_genes_baseline")


# ======================================================================
# Helpers
# ======================================================================

def _ts():
    """UTC timestamp for log lines."""
    return datetime.now(timezone.utc).strftime("%H:%M:%S")


def _build_m7_records(source_adata, target_adata):
    """Build M7 records from Census obs: dataset_id as site, binarized disease as outcome."""
    records = []
    for adata in [source_adata, target_adata]:
        for _, row in adata.obs.iterrows():
            records.append({
                "site": str(row["dataset_id"]),
                "outcome": 0 if str(row.get("disease", "normal")) == "normal" else 1,
            })
    return records


def _build_m6_knn_graph(X_source, X_target, k=10, max_nodes=500):
    """Build k-NN graph on combined embeddings for M6 curvature."""
    X = np.vstack([X_source, X_target])
    n = X.shape[0]
    if n > max_nodes:
        rng = np.random.default_rng(0)
        idx = rng.choice(n, size=max_nodes, replace=False)
        X = X[idx]
        n = max_nodes
    D = cdist(X, X, metric="cosine")
    np.fill_diagonal(D, np.inf)
    k_actual = min(k, n - 1)
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for i in range(n):
        neighbors = np.argsort(D[i])[:k_actual]
        for j in neighbors:
            G.add_edge(i, int(j))
    return G


def _query_slice(census, cellxgene_census, filter_str, max_cells, seed):
    """Query cell IDs first (fast), subsample, then pull embeddings + raw expression."""
    print(f"  {_ts()} Querying IDs: {filter_str[:70]}...")
    obs_df = cellxgene_census.get_obs(
        census, ORGANISM,
        value_filter=filter_str,
        column_names=["soma_joinid"],
    )
    n_total = len(obs_df)
    print(f"  {_ts()} Total matching cells: {n_total}")

    joinids = obs_df["soma_joinid"].values
    if n_total > max_cells:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n_total, size=max_cells, replace=False)
        idx.sort()
        joinids = joinids[idx]
        print(f"  {_ts()} Subsampled to: {len(joinids)} cells")

    print(f"  {_ts()} Pulling embeddings + raw expression for {len(joinids)} cells...")
    adata = cellxgene_census.get_anndata(
        census,
        organism=ORGANISM,
        obs_value_filter=filter_str,
        obs_coords=joinids,
        obs_column_names=OBS_COLUMNS,
        obs_embeddings=HOSTED_EMBEDDINGS,
    )
    print(f"  {_ts()} AnnData: {adata.n_obs} cells, "
          f"obsm keys: {list(adata.obsm.keys())}, "
          f"X shape: {adata.X.shape}")
    return adata


# ======================================================================
# Main
# ======================================================================

def main():
    try:
        import cellxgene_census
        import cellxgene_census.experimental
    except ImportError:
        print("ERROR: cellxgene-census not installed.")
        print("Run: uv pip install --python .venv-census/bin/python --only-binary :all: cellxgene-census")
        return 1

    from preflight.core.preregister import (
        DatasetSpec,
        compute_preregistration_sha,
        load_preregistration,
        verify_preregistration,
    )
    from preflight.core.runner import (
        ALL_MODULE_SOURCES,
        resolve_hyperparameters,
        run,
    )
    from preflight.extractors.anndata_loader import extract_arrays
    from preflight.extractors.bag_of_genes import (
        extract_bag_of_genes,
        extract_bag_of_genes_with_variance,
    )
    from preflight.probes import run_probes

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    print(f"{_ts()} Experiment 1: Bag-of-genes baseline")
    print(f"{_ts()} Census: {CENSUS_VERSION}, Max cells: {MAX_CELLS}")
    print(f"{_ts()} Embeddings: geneformer (512d), scvi (50d), "
          f"bag-of-genes PCA (512d, 50d)")

    # ==================================================================
    # STEP 1: Load and verify frozen preregistration
    # ==================================================================
    print(f"\n[1/9] {_ts()} Loading frozen preregistration...")

    if not FROZEN_PREREG_PATH.exists():
        print(f"ERROR: Frozen prereg not found: {FROZEN_PREREG_PATH}")
        print("Run: .venv-census/bin/python scripts/freeze_preregistration_v4.py")
        return 1

    frozen_record = load_preregistration(FROZEN_PREREG_PATH)
    frozen_sha = frozen_record.sha256
    print(f"  {_ts()} Loaded SHA: {frozen_sha[:32]}...")

    # Reconstruct dataset_spec (must match freeze_preregistration_v4.py exactly)
    dataset_spec = DatasetSpec(
        name="exp1_bag_of_genes_baseline",
        source_description="Lung 10x 3' v3 (geneformer + scvi + bag-of-genes PCA 512d)",
        target_description="Lung 10x 5' v2 (geneformer + scvi + bag-of-genes PCA 512d)",
        n_source=MAX_CELLS,
        n_target=MAX_CELLS,
        extra={
            "census_version": CENSUS_VERSION,
            "embeddings": ["geneformer", "scvi", "bag_of_genes_pca512"],
            "tissue": "lung",
        },
    )

    hp, weights = resolve_hyperparameters({"k": 5})
    canonical_hp = {**hp, "weights": weights}

    match, msg = verify_preregistration(
        frozen_record, ALL_MODULE_SOURCES, canonical_hp, dataset_spec,
    )
    if not match:
        print(f"  WARNING: SHA mismatch -- {msg}")
        print(f"  This means scorer code or hyperparameters changed since freezing.")
        print(f"  Proceeding anyway, but results are NOT preregistered.")
        prereg_verified = False
    else:
        print(f"  {_ts()} VERIFIED: {msg}")
        prereg_verified = True

    # ==================================================================
    # STEP 2: Pull source from Census (embeddings + raw expression)
    # ==================================================================
    print(f"\n[2/9] {_ts()} Querying source data (Lung 10x 3' v3)...")
    with cellxgene_census.open_soma(census_version=CENSUS_VERSION) as census:
        source_adata = _query_slice(
            census, cellxgene_census, SOURCE_FILTER, MAX_CELLS, seed=42,
        )

    X_src_gf, labels_src, _ = extract_arrays(
        source_adata, obsm_key="geneformer", obs_label_key="cell_type",
    )
    X_src_scvi, _, _ = extract_arrays(source_adata, obsm_key="scvi")
    print(f"  {_ts()} Source geneformer: {X_src_gf.shape}, scvi: {X_src_scvi.shape}")

    # ==================================================================
    # STEP 3: Pull target from Census
    # ==================================================================
    print(f"\n[3/9] {_ts()} Querying target data (Lung 10x 5' v2)...")
    with cellxgene_census.open_soma(census_version=CENSUS_VERSION) as census:
        target_adata = _query_slice(
            census, cellxgene_census, TARGET_FILTER, MAX_CELLS, seed=43,
        )

    X_tgt_gf, labels_tgt, _ = extract_arrays(
        target_adata, obsm_key="geneformer", obs_label_key="cell_type",
    )
    X_tgt_scvi, _, _ = extract_arrays(target_adata, obsm_key="scvi")
    print(f"  {_ts()} Target geneformer: {X_tgt_gf.shape}, scvi: {X_tgt_scvi.shape}")

    # ==================================================================
    # STEP 4: Extract bag-of-genes embeddings (joint PCA on source+target)
    # ==================================================================
    print(f"\n[4/9] {_ts()} Extracting bag-of-genes embeddings...")

    # Combine raw expression for joint PCA (so source and target share
    # the same principal component space for fair comparison)
    X_raw_src = source_adata.X
    X_raw_tgt = target_adata.X
    if sp.issparse(X_raw_src):
        X_raw_combined = sp.vstack([X_raw_src, X_raw_tgt])
    else:
        X_raw_combined = np.vstack([X_raw_src, X_raw_tgt])
    combined_for_pca = ad.AnnData(X=X_raw_combined)

    n_src = source_adata.n_obs
    print(f"  {_ts()} Combined expression matrix: {X_raw_combined.shape}")

    # 512-dimensional PCA (matches geneformer dimensionality)
    print(f"  {_ts()} Fitting PCA (512d)...")
    X_bog512_all, bog512_info = extract_bag_of_genes_with_variance(
        combined_for_pca, n_components=512,
    )
    X_src_bog512 = X_bog512_all[:n_src]
    X_tgt_bog512 = X_bog512_all[n_src:]
    print(f"  {_ts()} bag-of-genes 512d: {X_bog512_all.shape}, "
          f"variance explained: {bog512_info['total_variance_explained']:.3f}")

    # 50-dimensional PCA (matches scVI dimensionality)
    print(f"  {_ts()} Fitting PCA (50d)...")
    X_bog50_all, bog50_info = extract_bag_of_genes_with_variance(
        combined_for_pca, n_components=50,
    )
    X_src_bog50 = X_bog50_all[:n_src]
    X_tgt_bog50 = X_bog50_all[n_src:]
    print(f"  {_ts()} bag-of-genes 50d: {X_bog50_all.shape}, "
          f"variance explained: {bog50_info['total_variance_explained']:.3f}")

    # Free the large combined matrix
    del combined_for_pca, X_raw_combined, X_bog512_all, X_bog50_all

    # ==================================================================
    # STEP 5: Run Preflight pipeline for each embedding type
    # ==================================================================
    print(f"\n[5/9] {_ts()} Running Preflight pipeline (4 embedding types)...")

    embedding_configs = {
        "geneformer": {"src": X_src_gf, "tgt": X_tgt_gf},
        "scvi": {"src": X_src_scvi, "tgt": X_tgt_scvi},
        "bag_of_genes_512": {"src": X_src_bog512, "tgt": X_tgt_bog512},
        "bag_of_genes_50": {"src": X_src_bog50, "tgt": X_tgt_bog50},
    }

    # Build M7 records once (same obs data for all embeddings)
    m7_records = _build_m7_records(source_adata, target_adata)
    n_sites = len(set(r["site"] for r in m7_records))
    has_m7 = n_sites >= 2
    if has_m7:
        print(f"  {_ts()} M7: {len(m7_records)} records across {n_sites} sites")
    else:
        print(f"  {_ts()} M7 skipped: only {n_sites} site(s)")

    reports = {}
    for emb_name, arrays in tqdm(
        embedding_configs.items(), desc="Pipeline runs", unit="embedding",
    ):
        X_src = arrays["src"]
        X_tgt = arrays["tgt"]

        metadata = {}
        if labels_src is not None:
            metadata["source_labels"] = labels_src
        if labels_tgt is not None:
            metadata["target_labels"] = labels_tgt

        # M7: ecological bias (same for all embeddings)
        if has_m7:
            metadata["records"] = m7_records
            metadata["site_key"] = "site"
            metadata["outcome_key"] = "outcome"

        # M6: curvature (embedding-specific k-NN graph)
        metadata["graph"] = _build_m6_knn_graph(X_src, X_tgt)

        report = run(
            source=X_src,
            target=X_tgt,
            metadata=metadata,
            hyperparameters={"k": 5},
        )
        reports[emb_name] = report
        _inc_path = OUTPUT_DIR / f"incremental_{timestamp}.jsonl"
        with open(_inc_path, "a") as _jf:
            _jf.write(json.dumps({"emb": emb_name, **report.to_dict()}, default=str) + "\n")
            _jf.flush()
        print(f"  {_ts()} {emb_name}: tier={report.overall_tier}/7, "
              f"score={report.overall_score:.3f}")

    # ==================================================================
    # STEP 6: Run linear probes (multi-model comparison)
    # ==================================================================
    print(f"\n[6/9] {_ts()} Running linear probes...")

    probe_embeddings = {}
    for emb_name, arrays in embedding_configs.items():
        probe_embeddings[emb_name] = np.vstack([arrays["src"], arrays["tgt"]])

    combined_labels = np.concatenate([
        source_adata.obs["cell_type"].values,
        target_adata.obs["cell_type"].values,
    ])
    combined_donors = np.concatenate([
        source_adata.obs["donor_id"].values,
        target_adata.obs["donor_id"].values,
    ])

    n_donors = len(np.unique(combined_donors))
    probe_k = min(5, n_donors)
    probe_results = None
    if probe_k >= 2:
        probe_results = run_probes(
            embeddings=probe_embeddings,
            labels=combined_labels,
            batch_ids=combined_donors,
            k=probe_k,
        )
        print(probe_results["summary"])
    else:
        print(f"  {_ts()} Skipped: only {n_donors} donor(s), need >= 2 for CV")

    # ==================================================================
    # STEP 7: Test hypotheses
    # ==================================================================
    print(f"\n[7/9] {_ts()} Testing preregistered hypotheses...")

    hypotheses = {}

    # --- H1.1: Geneformer cell-type probe macro F1 > bag-of-genes F1 ---
    if probe_results and "cell_type" in probe_results.get("results_by_probe", {}):
        ct_results = probe_results["results_by_probe"]["cell_type"]
        gf_f1 = ct_results["geneformer"].mean
        bog512_f1 = ct_results["bag_of_genes_512"].mean

        # Find the paired comparison
        h1_1_p = None
        h1_1_delta = None
        for comp in probe_results["comparisons"]:
            if comp.probe_name != "cell_type":
                continue
            pair = {comp.model_a, comp.model_b}
            if pair == {"geneformer", "bag_of_genes_512"}:
                h1_1_p = comp.p_value
                # delta is model_a - model_b; adjust sign
                if comp.model_a == "geneformer":
                    h1_1_delta = comp.mean_delta
                else:
                    h1_1_delta = -comp.mean_delta
                break

        hypotheses["H1.1"] = {
            "description": "Geneformer cell-type probe macro F1 > bag-of-genes F1",
            "geneformer_f1": float(gf_f1),
            "bag_of_genes_512_f1": float(bog512_f1),
            "delta": float(h1_1_delta) if h1_1_delta is not None else None,
            "p_value": float(h1_1_p) if h1_1_p is not None else None,
            "supported": gf_f1 > bog512_f1,
        }
        status = "SUPPORTED" if gf_f1 > bog512_f1 else "NOT SUPPORTED"
        p_str = f", p={h1_1_p:.4f}" if h1_1_p is not None else ""
        print(f"  H1.1 [{status}]: geneformer F1={gf_f1:.3f} vs "
              f"bag-of-genes F1={bog512_f1:.3f}{p_str}")
    else:
        hypotheses["H1.1"] = {
            "description": "Geneformer cell-type probe macro F1 > bag-of-genes F1",
            "error": "Probes did not run (insufficient donors for CV)",
            "supported": None,
        }
        print(f"  H1.1 [UNTESTABLE]: probes did not run")

    # --- H1.2: Bag-of-genes M2 domain shift AUC < Geneformer M2 AUC ---
    gf_m2 = reports["geneformer"].module_results.get("m2_domain_shift")
    bog_m2 = reports["bag_of_genes_512"].module_results.get("m2_domain_shift")
    if gf_m2 and bog_m2:
        gf_auc = gf_m2.details["domain_classifier_auc"]
        bog_auc = bog_m2.details["domain_classifier_auc"]
        hypotheses["H1.2"] = {
            "description": "Bag-of-genes M2 domain shift AUC < Geneformer M2 AUC",
            "geneformer_m2_auc": float(gf_auc),
            "bag_of_genes_512_m2_auc": float(bog_auc),
            "supported": bog_auc < gf_auc,
        }
        status = "SUPPORTED" if bog_auc < gf_auc else "NOT SUPPORTED"
        print(f"  H1.2 [{status}]: bag-of-genes AUC={bog_auc:.3f} vs "
              f"geneformer AUC={gf_auc:.3f}")
    else:
        hypotheses["H1.2"] = {
            "description": "Bag-of-genes M2 domain shift AUC < Geneformer M2 AUC",
            "error": "M2 module did not produce results",
            "supported": None,
        }
        print(f"  H1.2 [UNTESTABLE]: M2 results missing")

    # --- H1.3: Bag-of-genes M1 Grassmannian distance <= Geneformer M1 distance ---
    gf_m1 = reports["geneformer"].module_results.get("m1_grassmannian")
    bog_m1 = reports["bag_of_genes_512"].module_results.get("m1_grassmannian")
    if gf_m1 and bog_m1:
        gf_dist = gf_m1.details["geodesic_distance"]
        bog_dist = bog_m1.details["geodesic_distance"]
        hypotheses["H1.3"] = {
            "description": "Bag-of-genes M1 Grassmannian distance <= Geneformer M1 distance",
            "geneformer_m1_geodesic": float(gf_dist),
            "bag_of_genes_512_m1_geodesic": float(bog_dist),
            "supported": bog_dist <= gf_dist,
        }
        status = "SUPPORTED" if bog_dist <= gf_dist else "NOT SUPPORTED"
        print(f"  H1.3 [{status}]: bag-of-genes dist={bog_dist:.3f} vs "
              f"geneformer dist={gf_dist:.3f}")
    else:
        hypotheses["H1.3"] = {
            "description": "Bag-of-genes M1 Grassmannian distance <= Geneformer M1 distance",
            "error": "M1 module did not produce results",
            "supported": None,
        }
        print(f"  H1.3 [UNTESTABLE]: M1 results missing")

    # ==================================================================
    # STEP 8: Save all artifacts
    # ==================================================================
    print(f"\n[8/9] {_ts()} Saving artifacts to {OUTPUT_DIR}/...")

    # Per-embedding pipeline reports
    for emb_name, report in reports.items():
        report_json = OUTPUT_DIR / f"report_{emb_name}_{timestamp}.json"
        report.save_json(report_json)

        report_md = OUTPUT_DIR / f"report_{emb_name}_{timestamp}.md"
        report_md.write_text(report.to_markdown())
        print(f"  {_ts()} {emb_name}: {report_json.name}")

    # Probe results
    if probe_results:
        probe_path = OUTPUT_DIR / f"probes_{timestamp}.json"
        probe_data = {
            "summary": probe_results["summary"],
            "results": {},
            "comparisons": [],
        }
        for key, pr in probe_results["results"].items():
            probe_data["results"][key] = {
                "model_name": pr.model_name,
                "probe_name": pr.probe_name,
                "mean": pr.mean,
                "ci_lo": pr.ci_lo,
                "ci_hi": pr.ci_hi,
                "scores": pr.scores,
                "n_items": pr.n_items,
                "details": pr.details,
            }
        for comp in probe_results["comparisons"]:
            probe_data["comparisons"].append({
                "model_a": comp.model_a,
                "model_b": comp.model_b,
                "probe_name": comp.probe_name,
                "n_pairs": comp.n_pairs,
                "mean_delta": comp.mean_delta,
                "ci_lo": comp.ci_lo,
                "ci_hi": comp.ci_hi,
                "p_value": comp.p_value,
            })
        with open(probe_path, "w") as f:
            json.dump(probe_data, f, indent=2, default=str)
        print(f"  {_ts()} Probes: {probe_path.name}")

    # Hypothesis results
    hyp_path = OUTPUT_DIR / f"hypotheses_{timestamp}.json"
    with open(hyp_path, "w") as f:
        json.dump(hypotheses, f, indent=2)
    print(f"  {_ts()} Hypotheses: {hyp_path.name}")

    # PCA variance diagnostics
    pca_info_path = OUTPUT_DIR / f"pca_variance_{timestamp}.json"
    with open(pca_info_path, "w") as f:
        json.dump({"pca_512d": bog512_info, "pca_50d": bog50_info}, f, indent=2)
    print(f"  {_ts()} PCA variance: {pca_info_path.name}")

    # Summary
    summary = {
        "experiment": "exp1_bag_of_genes_baseline",
        "timestamp": timestamp,
        "census_version": CENSUS_VERSION,
        "source_cells": int(source_adata.n_obs),
        "target_cells": int(target_adata.n_obs),
        "prereg_sha": frozen_sha,
        "prereg_verified": prereg_verified,
        "embedding_results": {},
        "hypotheses": hypotheses,
    }
    for emb_name, report in reports.items():
        emb_summary = {
            "overall_tier": report.overall_tier,
            "overall_score": float(report.overall_score),
            "module_scores": {
                name: {"score": float(r.score), "tier": r.tier}
                for name, r in report.module_results.items()
            },
        }
        # Add probe results for this embedding
        if probe_results and "cell_type" in probe_results.get("results_by_probe", {}):
            ct = probe_results["results_by_probe"]["cell_type"]
            if emb_name in ct:
                pr = ct[emb_name]
                emb_summary["probe_macro_f1"] = {
                    "mean": pr.mean,
                    "ci_lo": pr.ci_lo,
                    "ci_hi": pr.ci_hi,
                    "n_items": pr.n_items,
                }
        summary["embedding_results"][emb_name] = emb_summary

    summary_path = OUTPUT_DIR / f"summary_{timestamp}.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  {_ts()} Summary: {summary_path.name}")

    # ==================================================================
    # STEP 9: Print final report
    # ==================================================================
    print(f"\n[9/9] {_ts()} Final report")
    print("=" * 70)
    print(f"Preregistration SHA: {frozen_sha[:32]}... "
          f"({'VERIFIED' if prereg_verified else 'MISMATCH'})")
    print()

    print(f"{'Embedding':<22} | {'Tier':>6} | {'Score':>6} | "
          f"{'M1 dist':>8} | {'M2 AUC':>8} | {'Probe F1':>8}")
    print("-" * 70)
    for emb_name, report in reports.items():
        m1_dist = report.module_results.get("m1_grassmannian", None)
        m1_str = f"{m1_dist.details['geodesic_distance']:.3f}" if m1_dist else "n/a"
        m2_auc = report.module_results.get("m2_domain_shift", None)
        m2_str = f"{m2_auc.details['domain_classifier_auc']:.3f}" if m2_auc else "n/a"
        probe_str = "n/a"
        if probe_results and "cell_type" in probe_results.get("results_by_probe", {}):
            ct = probe_results["results_by_probe"]["cell_type"]
            if emb_name in ct:
                probe_str = f"{ct[emb_name].mean:.3f}"
        print(f"{emb_name:<22} | {report.overall_tier:>4}/7 | "
              f"{report.overall_score:>6.3f} | {m1_str:>8} | "
              f"{m2_str:>8} | {probe_str:>8}")

    print()
    print("Hypotheses:")
    for h_name, h_data in hypotheses.items():
        supported = h_data.get("supported")
        if supported is None:
            status = "UNTESTABLE"
        elif supported:
            status = "SUPPORTED"
        else:
            status = "NOT SUPPORTED"
        print(f"  {h_name} [{status}]: {h_data['description']}")

    print("=" * 70)
    print(f"\n{_ts()} Done. Artifacts in {OUTPUT_DIR}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
