"""Experiment 7: scGPT Embeddings.

Re-runs Exp 2 multi-model sweep with scGPT added as a fourth embedding.
Tests whether findings generalize across foundation models.

Usage:
    python scripts/exp7_scgpt.py
"""
import json
import networkx as nx
import numpy as np
from datetime import datetime, timezone
from pathlib import Path
from scipy.spatial.distance import cdist
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tqdm import tqdm

CENSUS_VERSION = "2023-12-15"
ORGANISM = "homo_sapiens"
MAX_CELLS = 2000

OBS_COLUMNS = [
    "cell_type", "tissue", "assay", "disease",
    "dataset_id", "donor_id", "is_primary_data",
]

TISSUES = {
    "lung":   "UBERON:0002048",
    "liver":  "UBERON:0002107",
    "kidney": "UBERON:0002113",
    "brain":  "UBERON:0000955",
}

EMBEDDINGS = ["geneformer", "scvi", "scgpt"]
SOURCE_ASSAY = "EFO:0009922"
TARGET_ASSAY = "EFO:0011025"

FROZEN_PREREG_PATH = Path("docs/frozen_prereg_v7/exp7_scgpt.json")
OUTPUT_DIR = Path("results/sweep_v7")


def _ts():
    return datetime.now(timezone.utc).strftime("%H:%M:%S")


def _make_filter(tissue_id, assay_id):
    return (
        f"tissue_ontology_term_id == '{tissue_id}' "
        f"and is_primary_data == True "
        f"and assay_ontology_term_id == '{assay_id}'"
    )


def _build_knn_graph(X_source, X_target, k=10, max_nodes=500):
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


def run_probes(X_src, labels_src, X_tgt, labels_tgt):
    le = LabelEncoder()
    shared = set(labels_src) & set(labels_tgt)
    if len(shared) < 2:
        return {"f1_source": 0.0, "f1_target": 0.0, "n_shared": len(shared)}

    mask_src = np.isin(labels_src, list(shared))
    mask_tgt = np.isin(labels_tgt, list(shared))
    X_s = X_src[mask_src]
    X_t = X_tgt[mask_tgt]
    y_s = labels_src[mask_src]
    y_t = labels_tgt[mask_tgt]

    le.fit(list(shared))
    y_s_enc = le.transform(y_s)
    y_t_enc = le.transform(y_t)

    scaler = StandardScaler()
    X_s_sc = scaler.fit_transform(X_s)
    X_t_sc = scaler.transform(X_t)

    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_s_sc, y_s_enc)

    f1_src = f1_score(y_s_enc, clf.predict(X_s_sc), average="macro")
    f1_tgt = f1_score(y_t_enc, clf.predict(X_t_sc), average="macro")

    return {"f1_source": float(f1_src), "f1_target": float(f1_tgt), "n_shared": len(shared)}


def main():
    import cellxgene_census
    from preflight.core.runner import run
    from preflight.extractors.bag_of_genes import extract_bag_of_genes_with_variance
    import anndata as ad
    from scipy import sparse as sp

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    print(f"{_ts()} Experiment 7: scGPT + Multi-Model Sweep")
    print(f"{_ts()} Census: {CENSUS_VERSION}")
    print(f"{_ts()} Embeddings: {EMBEDDINGS} + bag_of_genes_pca512")

    results = {}
    inc_path = OUTPUT_DIR / f"incremental_{timestamp}.jsonl"
    tissues_with_scgpt = []

    with cellxgene_census.open_soma(census_version=CENSUS_VERSION) as census:
        for tissue_name, tissue_id in tqdm(TISSUES.items(), desc="Tissues"):
            print(f"\n{_ts()} === {tissue_name} ===")

            src_filter = _make_filter(tissue_id, SOURCE_ASSAY)
            tgt_filter = _make_filter(tissue_id, TARGET_ASSAY)

            # Pull source
            obs_df = cellxgene_census.get_obs(
                census, ORGANISM, value_filter=src_filter,
                column_names=["soma_joinid"],
            )
            src_joinids = obs_df["soma_joinid"].values
            if len(src_joinids) > MAX_CELLS:
                rng = np.random.default_rng(42)
                idx = rng.choice(len(src_joinids), size=MAX_CELLS, replace=False)
                idx.sort()
                src_joinids = src_joinids[idx]

            # Check which embeddings are available
            try:
                src_adata = cellxgene_census.get_anndata(
                    census, organism=ORGANISM,
                    obs_value_filter=src_filter,
                    obs_coords=src_joinids,
                    obs_column_names=OBS_COLUMNS,
                    obs_embeddings=EMBEDDINGS,
                )
                available_embeddings = [e for e in EMBEDDINGS if e in src_adata.obsm]
                print(f"  Source: {src_adata.n_obs} cells, available embeddings: {available_embeddings}")
            except Exception as e:
                print(f"  ERROR pulling source with all embeddings: {e}")
                # Retry without scgpt
                available_embeddings = [e for e in EMBEDDINGS if e != "scgpt"]
                src_adata = cellxgene_census.get_anndata(
                    census, organism=ORGANISM,
                    obs_value_filter=src_filter,
                    obs_coords=src_joinids,
                    obs_column_names=OBS_COLUMNS,
                    obs_embeddings=available_embeddings,
                )
                print(f"  Source (retry): {src_adata.n_obs} cells, embeddings: {available_embeddings}")

            if "scgpt" in available_embeddings:
                tissues_with_scgpt.append(tissue_name)

            # Pull target
            obs_df = cellxgene_census.get_obs(
                census, ORGANISM, value_filter=tgt_filter,
                column_names=["soma_joinid"],
            )
            tgt_joinids = obs_df["soma_joinid"].values
            if len(tgt_joinids) == 0:
                print(f"  SKIPPING {tissue_name}: no target cells")
                continue
            if len(tgt_joinids) > MAX_CELLS:
                rng = np.random.default_rng(43)
                idx = rng.choice(len(tgt_joinids), size=MAX_CELLS, replace=False)
                idx.sort()
                tgt_joinids = tgt_joinids[idx]

            tgt_adata = cellxgene_census.get_anndata(
                census, organism=ORGANISM,
                obs_value_filter=tgt_filter,
                obs_coords=tgt_joinids,
                obs_column_names=OBS_COLUMNS,
                obs_embeddings=available_embeddings,
            )
            print(f"  Target: {tgt_adata.n_obs} cells")

            labels_src = src_adata.obs["cell_type"].values
            labels_tgt = tgt_adata.obs["cell_type"].values

            # Bag-of-genes PCA-512
            X_raw_src = src_adata.X
            X_raw_tgt = tgt_adata.X
            if sp.issparse(X_raw_src):
                X_raw_combined = sp.vstack([X_raw_src, X_raw_tgt])
            else:
                X_raw_combined = np.vstack([X_raw_src, X_raw_tgt])
            combined_for_pca = ad.AnnData(X=X_raw_combined)
            n_src = src_adata.n_obs

            X_bog_all, _ = extract_bag_of_genes_with_variance(combined_for_pca, n_components=512)
            X_src_bog = X_bog_all[:n_src]
            X_tgt_bog = X_bog_all[n_src:]

            # Run pipeline for each embedding
            tissue_results = {}
            all_embs = available_embeddings + ["bag_of_genes_pca512"]
            for emb_name in all_embs:
                if emb_name == "bag_of_genes_pca512":
                    X_src = X_src_bog
                    X_tgt = X_tgt_bog
                else:
                    X_src = src_adata.obsm[emb_name]
                    X_tgt = tgt_adata.obsm[emb_name]

                metadata = {
                    "source_labels": labels_src,
                    "target_labels": labels_tgt,
                    "graph": _build_knn_graph(X_src, X_tgt),
                }

                report = run(source=X_src, target=X_tgt, metadata=metadata, hyperparameters={"k": 5})
                probe_results = run_probes(X_src, labels_src, X_tgt, labels_tgt)

                tissue_results[emb_name] = {
                    "overall_tier": report.overall_tier,
                    "overall_score": report.overall_score,
                    "module_scores": {k: {"score": v.score, "tier": v.tier} for k, v in report.module_results.items()},
                    "probe_f1_source": probe_results["f1_source"],
                    "probe_f1_target": probe_results["f1_target"],
                }
                print(f"    {emb_name}: Tier {report.overall_tier}, "
                      f"F1 {probe_results['f1_source']:.3f}->{probe_results['f1_target']:.3f}")

            results[tissue_name] = tissue_results
            with open(inc_path, "a") as f:
                f.write(json.dumps({"tissue": tissue_name, **tissue_results}) + "\n")
                f.flush()

    # Contingency check
    print(f"\n{'=' * 60}")
    if len(tissues_with_scgpt) < 2:
        print(f"CONTINGENCY: scGPT available in only {len(tissues_with_scgpt)} tissues.")
        print("Experiment dropped per contingency plan.")
        summary = {
            "experiment": "exp7_scgpt",
            "timestamp": timestamp,
            "status": "DROPPED",
            "reason": f"scGPT embeddings available in only {len(tissues_with_scgpt)} tissues",
            "tissues_with_scgpt": tissues_with_scgpt,
        }
    else:
        print(f"scGPT available in {len(tissues_with_scgpt)} tissues: {tissues_with_scgpt}")
        print("\nHYPOTHESIS TESTING")

        # H7.1: scGPT achieves intermediate tiers (3-4)
        scgpt_tiers = [results[t]["scgpt"]["overall_tier"] for t in tissues_with_scgpt if "scgpt" in results[t]]
        h71 = 3 <= np.mean(scgpt_tiers) <= 4 if scgpt_tiers else False
        print(f"\nH7.1: scGPT mean tier in [3,4]: {np.mean(scgpt_tiers):.2f} -> {'SUPPORTED' if h71 else 'REJECTED'}")

        # H7.2: scGPT F1 within 0.10 of Geneformer
        f1_diffs = []
        for t in tissues_with_scgpt:
            if "scgpt" in results[t] and "geneformer" in results[t]:
                gf_f1 = results[t]["geneformer"]["probe_f1_target"]
                sc_f1 = results[t]["scgpt"]["probe_f1_target"]
                f1_diffs.append(abs(gf_f1 - sc_f1))
        h72 = all(d <= 0.10 for d in f1_diffs) if f1_diffs else False
        print(f"H7.2: scGPT F1 within 0.10 of Geneformer: diffs={[f'{d:.3f}' for d in f1_diffs]} -> {'SUPPORTED' if h72 else 'REJECTED'}")

        # H7.4: Tier ordering consistent across tissues
        consistent = 0
        for t in results:
            tiers = {}
            for emb in ["geneformer", "scgpt", "scvi", "bag_of_genes_pca512"]:
                if emb in results[t]:
                    tiers[emb] = results[t][emb]["overall_tier"]
            if all(k in tiers for k in ["geneformer", "scvi", "bag_of_genes_pca512"]):
                if tiers["geneformer"] <= tiers.get("scgpt", 99) <= tiers["scvi"]:
                    consistent += 1
        h74 = consistent >= 3
        print(f"H7.4: Tier ordering consistent in >= 3 tissues: {consistent}/{len(results)} -> {'SUPPORTED' if h74 else 'REJECTED'}")

        summary = {
            "experiment": "exp7_scgpt",
            "timestamp": timestamp,
            "status": "COMPLETED",
            "prereg_sha": json.load(open(FROZEN_PREREG_PATH))["sha256"] if FROZEN_PREREG_PATH.exists() else "N/A",
            "census_version": CENSUS_VERSION,
            "tissues_with_scgpt": tissues_with_scgpt,
            "results": results,
            "hypotheses": {
                "H7.1": {"supported": bool(h71), "scgpt_tiers": scgpt_tiers},
                "H7.2": {"supported": bool(h72), "f1_diffs": f1_diffs},
                "H7.4": {"supported": bool(h74), "consistent_tissues": consistent},
            },
        }

    output_path = OUTPUT_DIR / "summary_v7.json"
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
