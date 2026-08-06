"""Exp 2: Multi-model sweep across 5 tissues x 3 embeddings.

Preregistered experiment: for each tissue, pull cross-assay source/target
(10x 3'v3 -> 10x 5'v2), extract three embeddings (Geneformer, scVI,
bag-of-genes PCA 512d), run the full pipeline + probes, and test four
hypotheses about cross-tissue ranking consistency and domain shift.

Fallback: if a tissue lacks both assays with >=500 cells, use
cross-dataset split (different dataset_ids, same assay).

Usage:
    .venv-census/bin/python scripts/exp2_multi_model_sweep.py
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import networkx as nx
import numpy as np
from scipy import stats
from scipy.spatial.distance import cdist
from tqdm import tqdm

CENSUS_VERSION = "2023-12-15"
ORGANISM = "homo_sapiens"
MAX_CELLS = 2000
MIN_CELLS_FOR_ASSAY = 500

TISSUES = {
    "lung": "UBERON:0002048",
    "heart": "UBERON:0000948",
    "liver": "UBERON:0002107",
    "kidney": "UBERON:0002113",
    "brain": "UBERON:0000955",
}

SOURCE_ASSAY = "EFO:0009922"  # 10x 3' v3
TARGET_ASSAY = "EFO:0011025"  # 10x 5' v2

EMBEDDINGS = ["geneformer", "scvi", "bag_of_genes_pca512"]

OBS_COLUMNS = [
    "cell_type", "tissue", "assay", "disease",
    "dataset_id", "donor_id", "is_primary_data",
]

OUTPUT_DIR = Path("results/sweep")
PREREG_DIR = Path("docs/frozen_prereg_v6")


def _build_m7_records(source_adata, target_adata):
    records = []
    for adata in [source_adata, target_adata]:
        for _, row in adata.obs.iterrows():
            records.append({
                "site": str(row["dataset_id"]),
                "outcome": 0 if str(row.get("disease", "normal")) == "normal" else 1,
            })
    return records


def _build_m6_knn_graph(X_source, X_target, k=10, max_nodes=500):
    rng = np.random.default_rng(0)
    X = np.vstack([X_source, X_target])
    n = X.shape[0]
    if n > max_nodes:
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
    obs_df = cellxgene_census.get_obs(
        census, ORGANISM,
        value_filter=filter_str,
        column_names=["soma_joinid"],
    )
    n_total = len(obs_df)
    joinids = obs_df["soma_joinid"].values
    if n_total > max_cells:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n_total, size=max_cells, replace=False)
        idx.sort()
        joinids = joinids[idx]
    return joinids, n_total


def _pull_adata_with_embeddings(census, cellxgene_census, filter_str, joinids):
    adata = cellxgene_census.get_anndata(
        census,
        organism=ORGANISM,
        obs_value_filter=filter_str,
        obs_coords=joinids,
        obs_column_names=OBS_COLUMNS,
        obs_embeddings=["geneformer", "scvi"],
    )
    return adata


def _count_cells(census, cellxgene_census, filter_str):
    obs_df = cellxgene_census.get_obs(
        census, ORGANISM,
        value_filter=filter_str,
        column_names=["soma_joinid"],
    )
    return len(obs_df)


def _determine_shift_type(census, cellxgene_census, tissue_id):
    src_filter = (
        f"tissue_ontology_term_id == '{tissue_id}' "
        f"and is_primary_data == True "
        f"and assay_ontology_term_id == '{SOURCE_ASSAY}'"
    )
    tgt_filter = (
        f"tissue_ontology_term_id == '{tissue_id}' "
        f"and is_primary_data == True "
        f"and assay_ontology_term_id == '{TARGET_ASSAY}'"
    )

    n_src = _count_cells(census, cellxgene_census, src_filter)
    n_tgt = _count_cells(census, cellxgene_census, tgt_filter)

    if n_src >= MIN_CELLS_FOR_ASSAY and n_tgt >= MIN_CELLS_FOR_ASSAY:
        return "cross_assay", src_filter, tgt_filter

    # Fallback: cross-dataset split within the larger assay
    print(f"    Fallback: cross-assay insufficient (src={n_src}, tgt={n_tgt})")
    fallback_assay = SOURCE_ASSAY if n_src >= n_tgt else TARGET_ASSAY
    base_filter = (
        f"tissue_ontology_term_id == '{tissue_id}' "
        f"and is_primary_data == True "
        f"and assay_ontology_term_id == '{fallback_assay}'"
    )

    obs_df = cellxgene_census.get_obs(
        census, ORGANISM,
        value_filter=base_filter,
        column_names=["soma_joinid", "dataset_id"],
    )
    datasets = obs_df["dataset_id"].value_counts()
    viable = datasets[datasets >= MIN_CELLS_FOR_ASSAY]

    if len(viable) >= 2:
        top2 = viable.index[:2]
        src_filter_fb = base_filter + f" and dataset_id == '{top2[0]}'"
        tgt_filter_fb = base_filter + f" and dataset_id == '{top2[1]}'"
        return "cross_dataset", src_filter_fb, tgt_filter_fb

    print(f"    WARNING: No viable cross-dataset split either")
    return "cross_dataset", base_filter, base_filter


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
        load_preregistration,
        verify_preregistration,
    )
    from preflight.core.runner import (
        ALL_MODULE_SOURCES,
        resolve_hyperparameters,
        run,
    )
    from preflight.extractors.anndata_loader import extract_arrays
    from preflight.probes import run_probes

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    print(f"[{datetime.now(timezone.utc).isoformat()}] Exp 2: Multi-model sweep")
    print(f"Census: {CENSUS_VERSION}, Max cells: {MAX_CELLS}")
    print(f"Tissues: {list(TISSUES.keys())}")
    print(f"Embeddings: {EMBEDDINGS}")

    # ==================================================================
    # STEP 1: Load and verify frozen preregistration
    # ==================================================================
    print(f"\n[{datetime.now(timezone.utc).isoformat()}] [1] Loading frozen preregistration...")

    prereg_path = PREREG_DIR / "exp2_multi_model_sweep.json"
    if not prereg_path.exists():
        print(f"ERROR: Frozen prereg not found at {prereg_path}")
        print("Run: python scripts/freeze_preregistration_v4.py")
        return 1

    prereg_record = load_preregistration(prereg_path)
    frozen_sha = prereg_record.sha256
    print(f"  Frozen SHA: {frozen_sha[:32]}...")

    hp, weights = resolve_hyperparameters({"k": 5})
    canonical_hp = {**hp, "weights": weights}

    dataset_spec = DatasetSpec(
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
    )

    match, msg = verify_preregistration(prereg_record, ALL_MODULE_SOURCES, canonical_hp, dataset_spec)
    if not match:
        print(f"  VERIFICATION FAILED: {msg}")
        return 1
    print(f"  VERIFIED: {msg}")

    # ==================================================================
    # STEP 2: For each tissue, determine shift type and pull data
    # ==================================================================
    print(f"\n[{datetime.now(timezone.utc).isoformat()}] [2] Pulling data for each tissue...")

    tissue_data = {}
    tissue_shift_types = {}

    with cellxgene_census.open_soma(census_version=CENSUS_VERSION) as census:
        for tissue_name, tissue_id in tqdm(TISSUES.items(), desc="Tissues"):
            print(f"\n  --- {tissue_name} ({tissue_id}) ---")
            print(f"  [{datetime.now(timezone.utc).isoformat()}] Checking assay availability...")

            shift_type, src_filter, tgt_filter = _determine_shift_type(
                census, cellxgene_census, tissue_id
            )
            tissue_shift_types[tissue_name] = shift_type
            print(f"    Shift type: {shift_type}")

            # Pull source
            print(f"  [{datetime.now(timezone.utc).isoformat()}] Pulling source cells...")
            src_joinids, n_src_total = _query_slice(
                census, cellxgene_census, src_filter, MAX_CELLS, seed=42
            )
            print(f"    Source: {len(src_joinids)} / {n_src_total} cells")

            src_adata = _pull_adata_with_embeddings(
                census, cellxgene_census, src_filter, src_joinids
            )

            # Pull target
            print(f"  [{datetime.now(timezone.utc).isoformat()}] Pulling target cells...")
            tgt_joinids, n_tgt_total = _query_slice(
                census, cellxgene_census, tgt_filter, MAX_CELLS, seed=43
            )
            print(f"    Target: {len(tgt_joinids)} / {n_tgt_total} cells")

            tgt_adata = _pull_adata_with_embeddings(
                census, cellxgene_census, tgt_filter, tgt_joinids
            )

            if src_adata.n_obs == 0 or tgt_adata.n_obs == 0:
                print(f"    SKIPPING {tissue_name}: src={src_adata.n_obs}, tgt={tgt_adata.n_obs} cells")
                continue

            tissue_data[tissue_name] = {
                "src_adata": src_adata,
                "tgt_adata": tgt_adata,
                "src_filter": src_filter,
                "tgt_filter": tgt_filter,
            }
            print(f"    Source obsm: {list(src_adata.obsm.keys())}")
            print(f"    Target obsm: {list(tgt_adata.obsm.keys())}")

    # ==================================================================
    # STEP 3: Extract bag-of-genes embeddings
    # ==================================================================
    print(f"\n[{datetime.now(timezone.utc).isoformat()}] [3] Extracting bag-of-genes PCA embeddings...")

    from preflight.extractors.bag_of_genes import extract_bag_of_genes

    for tissue_name in tqdm(tissue_data, desc="Bag-of-genes"):
        td = tissue_data[tissue_name]
        td["bog_src"] = extract_bag_of_genes(td["src_adata"], n_components=512, seed=0)
        td["bog_tgt"] = extract_bag_of_genes(td["tgt_adata"], n_components=512, seed=0)
        print(f"  {tissue_name}: src={td['bog_src'].shape}, tgt={td['bog_tgt'].shape}")

    # ==================================================================
    # STEP 4: Run pipeline for each tissue x embedding
    # ==================================================================
    print(f"\n[{datetime.now(timezone.utc).isoformat()}] [4] Running pipeline: 5 tissues x 3 embeddings...")

    # score_matrix[tissue][embedding] = {report, probe_results, ...}
    all_results = {}
    n_conditions = len(tissue_data) * len(EMBEDDINGS)
    condition_idx = 0

    for tissue_name in tqdm(tissue_data, desc="Pipeline runs"):
        all_results[tissue_name] = {}
        td = tissue_data[tissue_name]
        src_adata = td["src_adata"]
        tgt_adata = td["tgt_adata"]

        for emb_name in EMBEDDINGS:
            condition_idx += 1
            print(f"\n  [{datetime.now(timezone.utc).isoformat()}] "
                  f"Condition {condition_idx}/{n_conditions}: {tissue_name} / {emb_name}")

            # Extract embeddings
            if emb_name == "bag_of_genes_pca512":
                X_src = td["bog_src"]
                X_tgt = td["bog_tgt"]
            else:
                obsm_key = emb_name
                X_src, _, _ = extract_arrays(src_adata, obsm_key=obsm_key, obs_label_key="cell_type")
                X_tgt, _, _ = extract_arrays(tgt_adata, obsm_key=obsm_key, obs_label_key="cell_type")

            print(f"    Embeddings: src={X_src.shape}, tgt={X_tgt.shape}")

            if X_src.shape[1] != X_tgt.shape[1]:
                print(f"    SKIP: dimension mismatch src={X_src.shape[1]} vs tgt={X_tgt.shape[1]}")
                all_results[tissue_name][emb_name] = None
                continue

            # Build metadata
            metadata = {}
            metadata["source_labels"] = src_adata.obs["cell_type"].values
            metadata["target_labels"] = tgt_adata.obs["cell_type"].values

            m7_records = _build_m7_records(src_adata, tgt_adata)
            n_sites = len(set(r["site"] for r in m7_records))
            if n_sites >= 2:
                metadata["records"] = m7_records
                metadata["site_key"] = "site"
                metadata["outcome_key"] = "outcome"

            metadata["graph"] = _build_m6_knn_graph(X_src, X_tgt)

            # Run pipeline
            report = run(
                source=X_src,
                target=X_tgt,
                metadata=metadata,
                hyperparameters={"k": 5},
                preregister=True,
                preregister_path=str(prereg_path),
                dataset_spec=dataset_spec,
            )
            print(f"    Tier: {report.overall_tier}/7, Score: {report.overall_score:.3f}")

            # Run probes
            combined_X = np.vstack([X_src, X_tgt])
            combined_labels = np.concatenate([
                src_adata.obs["cell_type"].values,
                tgt_adata.obs["cell_type"].values,
            ])
            combined_donors = np.concatenate([
                src_adata.obs["donor_id"].values,
                tgt_adata.obs["donor_id"].values,
            ])
            n_donors = len(np.unique(combined_donors))
            probe_k = min(5, n_donors)

            probe_results = None
            if probe_k >= 2:
                probe_results = run_probes(
                    embeddings={emb_name: combined_X},
                    labels=combined_labels,
                    batch_ids=combined_donors,
                    k=probe_k,
                )

            all_results[tissue_name][emb_name] = {
                "report": report,
                "probe_results": probe_results,
                "shift_type": tissue_shift_types[tissue_name],
            }
            _inc_path = OUTPUT_DIR / f"incremental_{timestamp}.jsonl"
            _inc_rec = {
                "tissue": tissue_name, "embedding": emb_name,
                "shift_type": tissue_shift_types[tissue_name],
                **report.to_dict(),
            }
            with open(_inc_path, "a") as _jf:
                _jf.write(json.dumps(_inc_rec, default=str) + "\n")
                _jf.flush()

    # ==================================================================
    # STEP 5: Build score matrix and test hypotheses
    # ==================================================================
    print(f"\n[{datetime.now(timezone.utc).isoformat()}] [5] Testing hypotheses...")

    # NxE score matrices (N = tissues that had cells)
    tissues_list = list(tissue_data.keys())
    n_tissues = len(tissues_list)
    probe_f1_matrix = np.full((n_tissues, 3), np.nan)    # tissue x embedding
    m2_auc_matrix = np.full((n_tissues, 3), np.nan)       # tissue x embedding
    m1_dist_matrix = np.full((n_tissues, 3), np.nan)      # tissue x embedding
    tier_matrix = np.full((n_tissues, 3), np.nan)          # tissue x embedding

    for i, tissue in enumerate(tissues_list):
        for j, emb in enumerate(EMBEDDINGS):
            res = all_results[tissue].get(emb)
            if res is None:
                continue
            report = res["report"]
            tier_matrix[i, j] = report.overall_tier

            # M1 Grassmannian distance
            m1_details = report.module_results.get("m1_grassmannian")
            if m1_details:
                m1_dist_matrix[i, j] = m1_details.details.get("geodesic_distance", np.nan)

            # M2 domain AUC
            m2_details = report.module_results.get("m2_domain_shift")
            if m2_details:
                m2_auc_matrix[i, j] = m2_details.details.get("domain_classifier_auc", np.nan)

            # Probe F1
            pr = res.get("probe_results")
            if pr and "cell_type" in pr.get("results_by_probe", {}):
                ct_result = pr["results_by_probe"]["cell_type"].get(emb)
                if ct_result:
                    probe_f1_matrix[i, j] = ct_result.mean

    print("\n  --- Score matrices ---")
    print(f"  Probe F1:\n{probe_f1_matrix}")
    print(f"  M2 domain AUC:\n{m2_auc_matrix}")
    print(f"  M1 geodesic distance:\n{m1_dist_matrix}")
    print(f"  Overall tier:\n{tier_matrix}")

    # H2.1: Model rankings on cell-type probe F1 consistent across >=3 tissues
    print("\n  H2.1: Probe F1 ranking consistency across tissues")
    n_consistent = 0
    rankings = []
    for i in range(n_tissues):
        row = probe_f1_matrix[i]
        if np.all(~np.isnan(row)):
            rank = np.argsort(-row)  # descending
            rankings.append(rank)
    if len(rankings) >= 2:
        # Check how many tissues share the same top model
        top_models = [r[0] for r in rankings]
        from collections import Counter
        top_counts = Counter(top_models)
        most_common_top, most_common_count = top_counts.most_common(1)[0]
        n_consistent = most_common_count
        h2_1_pass = n_consistent >= 3
        print(f"    Top model across tissues: {EMBEDDINGS[most_common_top]} in {n_consistent}/{n_tissues} tissues")
        print(f"    H2.1 {'PASS' if h2_1_pass else 'FAIL'}: "
              f"consistency in {n_consistent} tissues (need >= 3)")
    else:
        h2_1_pass = False
        print(f"    H2.1 SKIP: insufficient data ({len(rankings)} tissues with all embeddings)")

    # H2.2: M2 domain shift AUC varies by tissue (>=0.1 spread) for at least one model
    print("\n  H2.2: M2 domain AUC tissue variation")
    h2_2_pass = False
    for j, emb in enumerate(EMBEDDINGS):
        col = m2_auc_matrix[:, j]
        valid = col[~np.isnan(col)]
        if len(valid) >= 2:
            spread = float(np.max(valid) - np.min(valid))
            print(f"    {emb}: spread = {spread:.3f} (range [{np.min(valid):.3f}, {np.max(valid):.3f}])")
            if spread >= 0.1:
                h2_2_pass = True
    print(f"    H2.2 {'PASS' if h2_2_pass else 'FAIL'}: >= 0.1 spread in at least one model")

    # H2.3: M1 Grassmannian distance and M2 domain AUC positively correlated (Spearman rho > 0.3)
    print("\n  H2.3: M1 vs M2 correlation across 15 conditions")
    m1_flat = m1_dist_matrix.flatten()
    m2_flat = m2_auc_matrix.flatten()
    valid_mask = ~np.isnan(m1_flat) & ~np.isnan(m2_flat)
    n_valid = int(valid_mask.sum())
    if n_valid >= 5:
        rho, p_val = stats.spearmanr(m1_flat[valid_mask], m2_flat[valid_mask])
        h2_3_pass = rho > 0.3
        print(f"    Spearman rho = {rho:.3f}, p = {p_val:.4f} (n={n_valid})")
        print(f"    H2.3 {'PASS' if h2_3_pass else 'FAIL'}: rho > 0.3")
    else:
        h2_3_pass = False
        rho, p_val = np.nan, np.nan
        print(f"    H2.3 SKIP: insufficient valid pairs ({n_valid})")

    # H2.4: At least one tissue shows Tier >= 5 for Geneformer
    print("\n  H2.4: Geneformer tier >= 5 in at least one tissue")
    gf_idx = EMBEDDINGS.index("geneformer")
    gf_tiers = tier_matrix[:, gf_idx]
    valid_gf = gf_tiers[~np.isnan(gf_tiers)]
    h2_4_pass = bool(np.any(valid_gf >= 5))
    for i, tissue in enumerate(tissues_list):
        if not np.isnan(gf_tiers[i]):
            print(f"    {tissue}: Tier {int(gf_tiers[i])}")
    print(f"    H2.4 {'PASS' if h2_4_pass else 'FAIL'}: Tier >= 5 in at least one tissue")

    # ==================================================================
    # STEP 6: Save results
    # ==================================================================
    print(f"\n[{datetime.now(timezone.utc).isoformat()}] [6] Saving results...")

    # Save individual reports
    for tissue in tissues_list:
        for emb in EMBEDDINGS:
            res = all_results[tissue].get(emb)
            if res is None or res.get("report") is None:
                continue
            report_path = OUTPUT_DIR / f"report_{tissue}_{emb}_{timestamp}.json"
            res["report"].save_json(report_path)

    # Save aggregate summary
    summary = {
        "timestamp": timestamp,
        "frozen_prereg_sha": frozen_sha,
        "census_version": CENSUS_VERSION,
        "tissues": tissues_list,
        "embeddings": EMBEDDINGS,
        "shift_types": tissue_shift_types,
        "probe_f1_matrix": probe_f1_matrix.tolist(),
        "m2_auc_matrix": m2_auc_matrix.tolist(),
        "m1_distance_matrix": m1_dist_matrix.tolist(),
        "tier_matrix": tier_matrix.tolist(),
        "hypotheses": {
            "H2.1_ranking_consistency": {
                "pass": h2_1_pass,
                "n_consistent": n_consistent,
                "threshold": 3,
            },
            "H2.2_tissue_variation": {
                "pass": h2_2_pass,
                "threshold_spread": 0.1,
            },
            "H2.3_m1_m2_correlation": {
                "pass": h2_3_pass,
                "spearman_rho": float(rho) if not np.isnan(rho) else None,
                "p_value": float(p_val) if not np.isnan(p_val) else None,
                "n_conditions": n_valid,
                "threshold_rho": 0.3,
            },
            "H2.4_geneformer_tier5": {
                "pass": h2_4_pass,
                "geneformer_tiers": {t: int(gf_tiers[i]) for i, t in enumerate(tissues_list) if not np.isnan(gf_tiers[i])},
            },
        },
        "per_condition": {},
    }

    for tissue in tissues_list:
        summary["per_condition"][tissue] = {}
        for emb in EMBEDDINGS:
            res = all_results[tissue].get(emb)
            if res is None:
                summary["per_condition"][tissue][emb] = None
                continue
            report = res["report"]
            entry = {
                "overall_tier": report.overall_tier,
                "overall_score": float(report.overall_score),
                "shift_type": res["shift_type"],
                "module_scores": {
                    name: {"score": float(r.score), "tier": r.tier}
                    for name, r in report.module_results.items()
                },
            }
            pr = res.get("probe_results")
            if pr and "cell_type" in pr.get("results_by_probe", {}):
                ct_res = pr["results_by_probe"]["cell_type"].get(emb)
                if ct_res:
                    entry["probe_f1"] = {
                        "mean": ct_res.mean,
                        "ci": [ct_res.ci_lo, ct_res.ci_hi],
                        "n_items": ct_res.n_items,
                    }
            summary["per_condition"][tissue][emb] = entry

    summary_path = OUTPUT_DIR / f"summary_{timestamp}.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"  Summary: {summary_path}")

    # ==================================================================
    # Final report
    # ==================================================================
    print(f"\n{'=' * 60}")
    print("EXP 2 RESULTS: Multi-model sweep")
    print(f"{'=' * 60}")
    print(f"  Prereg SHA: {frozen_sha[:32]}...")
    print(f"  H2.1 ranking consistency: {'PASS' if h2_1_pass else 'FAIL'}")
    print(f"  H2.2 tissue variation:    {'PASS' if h2_2_pass else 'FAIL'}")
    print(f"  H2.3 M1-M2 correlation:   {'PASS' if h2_3_pass else 'FAIL'}")
    print(f"  H2.4 Geneformer tier>=5:  {'PASS' if h2_4_pass else 'FAIL'}")
    n_pass = sum([h2_1_pass, h2_2_pass, h2_3_pass, h2_4_pass])
    print(f"  Total: {n_pass}/4 hypotheses confirmed")
    print(f"\n[{datetime.now(timezone.utc).isoformat()}] Done. Artifacts in {OUTPUT_DIR}/")

    return 0


if __name__ == "__main__":
    sys.exit(main())
